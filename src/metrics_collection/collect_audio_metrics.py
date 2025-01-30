import dataclasses
import os
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from joblib import Parallel, delayed
from pydub import AudioSegment
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.metrics_collection.models import AudioMetrics, AudioToDataset, Base
from src.utils import read_metadata_and_calculate_hash, signal_to_noise, update_bar_on_ending

load_dotenv()


@click.command()
@click.option("--dataset-path", type=click.Path(exists=True), help="Path to dataset")
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .csv file with metadata.",
    callback=lambda context, _, value: value if value else os.path.join(context.params["dataset_path"], "metadata.csv"),
)
@click.option("--overwrite", type=bool, help="Is to overwrite existing metrics or not.", default=False)
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
def main(
    dataset_path: str,
    metadata_path: str,
    overwrite: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
):
    dataset_name = Path(dataset_path).stem
    metadata_df = read_metadata_and_calculate_hash(metadata_path, dataset_path, n_jobs=n_jobs)
    metadata_df = metadata_df.drop_duplicates(subset=["hash"])

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        existing_in_db_hashes_of_audio = session.scalars(select(AudioMetrics.audio_md5_hash)).all()
        existing_in_db_hashes_of_datasets_info = session.scalars(
            select(AudioToDataset.audio_md5_hash).where(AudioToDataset.dataset_name == dataset_name)
        ).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        samples_audio_info = get_audio_metrics_from_selected_samples(
            dataframe=samples_to_add, dataset_path=dataset_path, n_jobs=n_jobs
        )
        session.add_all(samples_audio_info)

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_datasets_info)]
        samples_dataset_info = get_datasets_info_from_selected_samples(
            dataframe=samples_to_add, dataset_path=dataset_path, dataset_name=dataset_name, n_jobs=n_jobs
        )
        session.add_all(samples_dataset_info)

        if overwrite:
            print("Overwriting others samples")
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
            samples_audio_info = get_audio_metrics_from_selected_samples(
                dataframe=samples_to_add, dataset_path=dataset_path, n_jobs=n_jobs
            )
            samples_audio_info = [dataclasses.asdict(info) for info in samples_audio_info if info is not None]
            session.execute(update(AudioMetrics), samples_audio_info)

            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_datasets_info)]
            samples_dataset_info = get_datasets_info_from_selected_samples(
                dataframe=samples_to_update, dataset_path=dataset_path, dataset_name=dataset_name, n_jobs=n_jobs
            )
            samples_dataset_info = [dataclasses.asdict(info) for info in samples_dataset_info if info is not None]
            session.execute(update(AudioToDataset), samples_dataset_info)

        session.commit()


def get_audio_metrics_from_selected_samples(
    dataframe: pd.DataFrame, dataset_path: str, n_jobs: int
) -> List[AudioMetrics]:
    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting audio metrics")
    samples_audio_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(get_audio_info))(
            sample.hash, os.path.join(dataset_path, sample.path_to_wav)
        )
        for sample in dataframe.itertuples()
    )
    samples_audio_info = [info for info in samples_audio_info if info is not None]
    return samples_audio_info


def get_datasets_info_from_selected_samples(
    dataframe: pd.DataFrame, dataset_path: str, dataset_name: str, n_jobs: int
) -> List[AudioToDataset]:
    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting dataset info")
    samples_dataset_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(AudioToDataset))(
            audio_md5_hash=sample.hash,
            dataset_name=dataset_name,
            path_to_file=sample.path_to_wav,
            speaker_id=sample.speaker_id,
        )
        for sample in dataframe.itertuples()
    )
    samples_dataset_info = [info for info in samples_dataset_info if info is not None]
    return samples_dataset_info


def get_audio_info(hash: str, path_to_audio: str) -> AudioMetrics | None:
    try:
        audio_segment = AudioSegment.from_file(path_to_audio)
        duration_seconds = audio_segment.duration_seconds
        dBFS = audio_segment.dBFS

        audio_info = sf.info(path_to_audio)
        sample_rate = audio_info.samplerate
        channels = audio_info.channels
        pcm_format = audio_info.subtype

        numpy_audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio_segment.channels)
        )
        SNR = signal_to_noise(numpy_audio)[0]

        return AudioMetrics(
            audio_md5_hash=hash,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            pcm_format=pcm_format,
            SNR=SNR,
            dBFS=dBFS,
        )

    except FileNotFoundError:
        print(f"File {path_to_audio} not found. Skipping this sample.")
        return None


if __name__ == "__main__":
    main()
