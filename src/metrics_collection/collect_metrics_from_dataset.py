import dataclasses
import functools
import hashlib
import os
from pathlib import Path
from typing import Any, Callable

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

from src.metrics_collection.models import (
    AudioMetrics,
    AudioToDataset,
    Base,
)

load_dotenv()


def update_bar_on_ending(status_bar: tqdm) -> Callable:
    def run_function(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            status_bar.update()
            return result

        return wrapper

    return run_function


@click.command()
@click.option("--dataset-path", type=click.Path(exists=True), help="Path to dataset")
@click.option("--overwrite", type=bool, help="Is to overwrite existing metrics or not.", default=False)
@click.option(
    "--get-asr-texts",
    type=bool,
    help="Is it needed to check for ASR texts and load them in the database.",
    default=True,
)
@click.option(
    "--get-mfa-textgrids",
    type=bool,
    help="Is it needed to check for MFA textgrids and load them in the databese",
    default=True,
)
@click.option(
    "--calculate-wer-cer",
    type=bool,
    help="Is it needed to calculate WER and CER for samples where ASR and original text exist",
    default=True,
)
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
    overwrite: bool,
    get_asr_texts: bool,
    get_mfa_textgrids: bool,
    calculate_wer_cer: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
):
    path_to_metadata = os.path.join(dataset_path, "metadata.csv")
    metadata_df = pd.read_csv(path_to_metadata, sep="|").drop_duplicates()
    dataset_name = Path(dataset_path).stem

    assert "path_to_wav" in metadata_df.columns
    assert "text" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}",
        echo=True,
    )

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        # Calculate hashes
        tqdm_bar = tqdm(total=len(metadata_df), desc="Getting hashes from audio files")
        metadata_df["hash"] = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(update_bar_on_ending(tqdm_bar)(get_hash_of_file))(os.path.join(dataset_path, sample.path_to_wav))
            for sample in metadata_df.itertuples()
        )

        existing_in_db_hashes_of_audio = session.scalars(select(AudioMetrics.audio_md5_hash)).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        tqdm_bar = tqdm(total=len(samples_to_add), desc="Collecting audio metrics")
        samples_audio_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(update_bar_on_ending(tqdm_bar)(get_audio_info))(
                sample.hash, os.path.join(dataset_path, sample.path_to_wav)
            )
            for sample in samples_to_add.itertuples()
        )
        samples_audio_info = [info for info in samples_audio_info if info is not None]
        session.add_all(samples_audio_info)

        if overwrite:
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
            tqdm_bar = tqdm(total=len(samples_to_update), desc="Collecting audio metrics (Overwrite)")
            samples_audio_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(update_bar_on_ending(tqdm_bar)(get_audio_info))(
                    sample.hash, os.path.join(dataset_path, sample.path_to_wav)
                )
                for sample in samples_to_update.itertuples()
            )
            samples_audio_info = [dataclasses.asdict(info) for info in samples_audio_info if info is not None]
            session.execute(update(AudioMetrics), samples_audio_info)

        # Dataset Info
        existing_in_db_hashes_of_audio = session.scalars(
            select(AudioToDataset.audio_md5_hash).where(AudioToDataset.dataset_name == dataset_name)
        ).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        tqdm_bar = tqdm(total=len(samples_to_add), desc="Collecting dataset info")
        samples_audio_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(update_bar_on_ending(tqdm_bar)(AudioToDataset))(
                audio_md5_hash=sample.hash,
                dataset_name=dataset_name,
                path_to_file=sample.path_to_wav,
                speaker_id=sample.speaker_id,
            )
            for sample in samples_to_add.itertuples()
        )
        samples_audio_info = [info for info in samples_audio_info if info is not None]
        session.add_all(samples_audio_info)

        if overwrite:
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
            tqdm_bar = tqdm(total=len(samples_to_update), desc="Collecting dataset info (Overwrite)")
            samples_audio_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(update_bar_on_ending(tqdm_bar)(AudioToDataset))(
                    audio_md5_hash=sample.hash,
                    dataset_name=dataset_name,
                    path_to_file=sample.path_to_wav,
                    speaker_id=sample.speaker_id,
                )
                for sample in samples_to_update.itertuples()
            )
            samples_audio_info = [dataclasses.asdict(info) for info in samples_audio_info if info is not None]
            session.execute(update(AudioToDataset), samples_audio_info)

        session.commit()


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


def signal_to_noise(a, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.

    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.

    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.

    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.

    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def get_hash_of_file(path_to_file: str) -> str:
    return hashlib.md5(open(path_to_file, "rb").read()).hexdigest()


if __name__ == "__main__":
    main()
