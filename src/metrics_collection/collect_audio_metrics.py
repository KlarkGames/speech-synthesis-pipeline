import dataclasses
from typing import List, NamedTuple, Dict, Any

import io
import click
import numpy as np
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from joblib import Parallel, delayed
from pydub import AudioSegment
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from sqlalchemy.engine.base import Engine
from tqdm import tqdm

from src.metrics_collection.models import AudioMetrics, AudioToDataset, Base
from src.utils import read_metadata_and_calculate_hash, signal_to_noise, update_bar_on_ending
from src.data_managers import LocalFileSystemManager, LakeFSFileSystemManager, AbstractFileSystemManager

load_dotenv()


@click.group()
@click.option("--overwrite", is_flag=True, help="Is to overwrite existing metrics or not.", default=False)
@click.option("--database-address", type=str, help="Address of the database", envvar="POSTGRES_ADDRESS")
@click.option("--database-port", type=int, help="Port of the database", envvar="POSTGRES_PORT")
@click.option("--database-user", type=str, help="Username to use for database authentication", envvar="POSTGRES_USER")
@click.option(
    "--database-password", type=str, help="Password to use for database authentication", envvar="POSTGRES_PASSWORD"
)
@click.option("--database-name", type=str, help="Name of the database", envvar="POSTGRES_DB")
@click.option(
    "--n-jobs", type=int, default=-1, help="Number of parallel jobs to use while processing. -1 means to use all cores."
)
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .csv file with metadata.",
    default=None,
)
@click.pass_context
def cli(
    context: click.Context,
    overwrite: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
    metadata_path: str | None,
):
    context.ensure_object(dict)

    context.obj["overwrite"] = overwrite
    context.obj["n_jobs"] = n_jobs

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )
    context.obj["engine"] = engine
    context.obj["metadata_path"] = metadata_path


@cli.command()
@click.option("--dataset-path", type=click.Path(exists=True), help="Path to dataset")
@click.pass_context
def local(context: click.Context, dataset_path: str):
    file_system_manager = LocalFileSystemManager(dataset_path)

    calculate_and_load_metrics_to_db(
        file_system_manager=file_system_manager,
        engine=context.obj["engine"],
        overwrite=context.obj["overwrite"],
        n_jobs=context.obj["n_jobs"],
        metadata_path=context.obj["metadata_path"],
    )


@cli.command()
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--repository-name", type=str, help="Name of LakeFS repository")
@click.option("--branch-name", type=str, help="Name of the branch.", default="main")
@click.pass_context
def s3(
    context: click.Context,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    repository_name: str,
    branch_name: str,
):
    file_system_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=repository_name,
        lakefs_branch_name=branch_name,
    )

    calculate_and_load_metrics_to_db(
        file_system_manager=file_system_manager,
        engine=context.obj["engine"],
        overwrite=context.obj["overwrite"],
        n_jobs=context.obj["n_jobs"],
        metadata_path=context.obj["metadata_path"],
    )


def calculate_and_load_metrics_to_db(
    file_system_manager: AbstractFileSystemManager,
    engine: Engine,
    overwrite: bool,
    n_jobs: int = -1,
    metadata_path: str | None = None,
) -> None:
    dataset_name = file_system_manager.directory_name

    if metadata_path is not None:
        metadata_df = read_metadata_and_calculate_hash(metadata_path, file_system_manager, n_jobs=n_jobs)
    else:
        with file_system_manager.get_buffered_reader("metadata.csv") as metadata_reader:
            metadata_df = read_metadata_and_calculate_hash(metadata_reader, file_system_manager, n_jobs=n_jobs)

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        existing_in_db_hashes_of_audio = session.scalars(select(AudioMetrics.audio_md5_hash)).all()
        existing_in_db_hashes_of_datasets_info = session.scalars(
            select(AudioToDataset.audio_md5_hash).where(AudioToDataset.dataset_name == dataset_name)
        ).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        samples_audio_info = get_audio_metrics_from_selected_samples(
            dataframe=samples_to_add, file_system_manager=file_system_manager, n_jobs=n_jobs
        )
        session.add_all(samples_audio_info)

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_datasets_info)]
        samples_dataset_info = get_datasets_info_from_selected_samples(
            dataframe=samples_to_add, file_system_manager=file_system_manager, n_jobs=n_jobs
        )
        session.add_all(samples_dataset_info)

        if overwrite:
            print("Overwriting others samples")
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
            samples_audio_info = get_audio_metrics_from_selected_samples(
                dataframe=samples_to_add, file_system_manager=file_system_manager, n_jobs=n_jobs
            )
            samples_audio_info = [dataclasses.asdict(info) for info in samples_audio_info if info is not None]
            session.execute(update(AudioMetrics), samples_audio_info)

            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_datasets_info)]
            samples_dataset_info = get_datasets_info_from_selected_samples(
                dataframe=samples_to_update, file_system_manager=file_system_manager, n_jobs=n_jobs
            )
            samples_dataset_info = [dataclasses.asdict(info) for info in samples_dataset_info if info is not None]
            session.execute(update(AudioToDataset), samples_dataset_info)

        session.commit()


def get_audio_metrics_from_selected_samples(
    dataframe: pd.DataFrame, file_system_manager: AbstractFileSystemManager, n_jobs: int = -1
) -> List[AudioMetrics]:
    def process_file(sample: NamedTuple, tqdm_bar: tqdm):
        with file_system_manager.get_buffered_reader(sample.path_to_wav) as reader:
            sf_results = get_audio_info_sf(reader)

        with file_system_manager.get_buffered_reader(sample.path_to_wav) as reader:
            pydub_results = get_audio_info_pydub(reader)

        return AudioMetrics(
            audio_md5_hash=sample.hash,
            sample_rate=sf_results["sample_rate"],
            channels=sf_results["channels"],
            pcm_format=sf_results["pcm_format"],
            duration_seconds=pydub_results["duration_seconds"],
            SNR=pydub_results["SNR"],
            dBFS=pydub_results["dBFS"],
        )

    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting audio metrics")
    samples_audio_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(process_file))(sample, tqdm_bar) for sample in dataframe.itertuples()
    )

    samples_audio_info = [info for info in samples_audio_info if info is not None]
    return samples_audio_info


def get_datasets_info_from_selected_samples(
    dataframe: pd.DataFrame, file_system_manager: AbstractFileSystemManager, n_jobs: int
) -> List[AudioToDataset]:
    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting dataset info")
    samples_dataset_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(AudioToDataset))(
            audio_md5_hash=sample.hash,
            dataset_name=file_system_manager.directory_name,
            path_to_file=sample.path_to_wav,
            speaker_id=sample.speaker_id,
        )
        for sample in dataframe.itertuples()
    )
    samples_dataset_info = [info for info in samples_dataset_info if info is not None]
    return samples_dataset_info


def get_audio_info_sf(path_to_audio: str | io.BufferedReader) -> Dict[str, Any] | None:
    try:
        audio_info = sf.info(path_to_audio)
        sample_rate = audio_info.samplerate
        channels = audio_info.channels
        pcm_format = audio_info.subtype

        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "pcm_format": pcm_format,
        }

    except FileNotFoundError:
        print(f"File {path_to_audio} not found. Skipping this sample.")
        return None


def get_audio_info_pydub(path_to_audio: str | io.BufferedReader) -> Dict[str, Any] | None:
    try:
        audio_segment = AudioSegment.from_file(path_to_audio)
        duration_seconds = audio_segment.duration_seconds
        dBFS = audio_segment.dBFS

        numpy_audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio_segment.channels)
        )
        SNR = signal_to_noise(numpy_audio)[0]

        return {
            "duration_seconds": duration_seconds,
            "SNR": SNR,
            "dBFS": dBFS,
        }

    except FileNotFoundError:
        print(f"File {path_to_audio} not found. Skipping this sample.")
        return None


if __name__ == "__main__":
    cli()
