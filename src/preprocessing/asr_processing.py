"""
Automatic Speech Recognition (ASR) Processing Pipeline

This module provides a CLI for batch processing audio files through an ASR model using Triton Inference Server.
It handles both local filesystem and LakeFS/S3 storage, with parallel processing capabilities and database integration
to track processing results.

Key Features:
- Processes audio files in parallel batches using Triton Inference Server
- Supports local and cloud storage via LakeFS/S3
- Maintains processing state in PostgreSQL database
- Generates text transcripts and stores them in appropriate storage
- Provides idempotent processing with overwrite capability

CLI Commands:
    local: Process audio files from local filesystem
    s3: Process audio files from LakeFS/S3 storage

Example usage:
    python script.py local --dataset-path ./audio_data \
        --database-address localhost --database-port 5432 \
        --database-user postgres --database-password secret
"""

import asyncio
import dataclasses

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from pydub import AudioSegment
from pytriton.client import AsyncioModelClient
from sqlalchemy import create_engine, select, update
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.data_managers import AbstractFileSystemManager, LakeFSFileSystemManager, LocalFileSystemManager
from src.metrics_collection.models import AudioToASRText, Base
from src.utils import read_metadata_and_calculate_hash, update_bar_on_ending

load_dotenv()


async def get_texts_from_audio_by_asr(triton_address, triton_port, file_system_manager, input_batch):
    """Process audio batch through ASR model and save recognized texts.

    Args:
        - triton_address: Triton Inference Server host address
        - triton_port: Triton Inference Server port number
        - file_system_manager: File system manager for input/output operations
        - input_batch: List of audio file paths to process

    Returns:
        dict: Mapping of audio file paths to their recognized texts

    Note:
        - Skips processing if text file already exists
        - Converts audio to mono 16kHz format before processing
        - Stores results as UTF-8 encoded text files
        - Uses async TaskGroup for concurrent inference requests
    """

    results = {}
    pending_responses = {}

    client = AsyncioModelClient(f"{triton_address}:{triton_port}", "ensemble_english_stt", inference_timeout_s=600)

    async with asyncio.TaskGroup() as tg:
        for input_file in input_batch:
            txt_path = input_file.replace("/wavs", "/asr_recognized_texts").replace(".wav", ".txt")

            if file_system_manager.is_path_exists(file_system_manager.get_absolute_path(txt_path)):
                with file_system_manager.get_buffered_reader(txt_path) as text_file:
                    text = text_file.read().decode("UTF-8")
                results[input_file] = text
            else:
                with file_system_manager.get_buffered_reader(input_file) as audio_file:
                    audio = AudioSegment.from_wav(audio_file).set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float64).reshape(1, -1)

                task = tg.create_task(client.infer_batch(audio_data))
                pending_responses[input_file] = task

    await client.close()

    for input_file, response in pending_responses.items():
        txt_path = input_file.replace("/wavs", "/asr_recognized_texts").replace(".wav", ".txt")

        text = response.result()["decoded_texts"].base[0].decode()

        with file_system_manager.get_buffered_writer(txt_path) as text_file:
            text_file.write(text.encode("UTF-8"))

        results[input_file] = text

    return results


def process_audios(input_batch, file_system_manager, triton_address, triton_port):
    """Orchestrate async processing of audio batch through ASR model.

    Args:
        - input_batch: List of audio file paths to process
        - file_system_manager: File system manager for input/output
        - triton_address: Triton server address
        - triton_port: Triton server port

    Returns:
        dict: Recognized texts mapping from audio paths to transcripts

    Note:
        Creates new event loop for async processing
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    recognized_texts = loop.run_until_complete(
        get_texts_from_audio_by_asr(
            input_batch=input_batch,
            file_system_manager=file_system_manager,
            triton_address=triton_address,
            triton_port=triton_port,
        )
    )

    return recognized_texts


@click.group()
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
@click.option("--triton-address", help="Address of the Triton Inference Server.", envvar="ASR_TRITON_ADDRESS")
@click.option("--triton-port", type=int, help="Port of the Triton Inference Server.", envvar="ASR_TRITON_HTTP_PORT")
@click.option("--batch-size", type=int, default=10, help="Batch size for processing audio files.")
@click.pass_context
def cli(
    ctx: click.Context,
    overwrite: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
    triton_address: str,
    triton_port: int,
    batch_size: int,
):
    """Main CLI entry point for ASR processing pipeline.

    Configures database connection, processing parameters, and Triton settings.
    """

    ctx.ensure_object(dict)

    ctx.obj["overwrite"] = overwrite
    ctx.obj["n_jobs"] = n_jobs

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )
    ctx.obj["engine"] = engine

    ctx.obj["triton_address"] = triton_address
    ctx.obj["triton_port"] = triton_port
    ctx.obj["batch_size"] = batch_size


@cli.command()
@click.option("--dataset-path", type=click.Path(exists=True), help="Path to dataset")
@click.pass_context
def local(ctx: click.Context, dataset_path: str):
    """Process audio files from local filesystem.

    Args:
        dataset_path: Local directory containing audio files and metadata.csv

    Requires:
        Metadata CSV with 'path_to_wav' column pointing to audio files
    """

    file_system_manager = LocalFileSystemManager(dataset_path)

    process_dataset(
        file_system_manager=file_system_manager,
        engine=ctx.obj["engine"],
        overwrite=ctx.obj["overwrite"],
        n_jobs=ctx.obj["n_jobs"],
        triton_address=ctx.obj["triton_address"],
        triton_port=ctx.obj["triton_port"],
        batch_size=ctx.obj["batch_size"],
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
    ctx: click.Context,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    repository_name: str,
    branch_name: str,
):
    """Process audio files from LakeFS/S3 storage.

    Requires LakeFS credentials and repository configuration.
    """

    file_system_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=repository_name,
        lakefs_branch_name=branch_name,
    )

    process_dataset(
        file_system_manager=file_system_manager,
        engine=ctx.obj["engine"],
        overwrite=ctx.obj["overwrite"],
        n_jobs=ctx.obj["n_jobs"],
        triton_address=ctx.obj["triton_address"],
        triton_port=ctx.obj["triton_port"],
        batch_size=ctx.obj["batch_size"],
    )


def process_dataset(
    file_system_manager: AbstractFileSystemManager,
    engine: Engine,
    overwrite: bool,
    n_jobs: int,
    triton_address: str,
    triton_port: int,
    batch_size: int,
):
    """Main processing workflow for ASR pipeline.

    Args:
        - file_system_manager: Storage manager for audio/text files
        - engine: SQLAlchemy database engine
        - overwrite: Flag to regenerate existing entries
        - n_jobs: Number of parallel processing jobs
        - triton_address: Triton server address
        - triton_port: Triton server port
        - batch_size: Number of files per processing batch

    Workflow:
        1. Load and validate metadata with audio hashes
        2. Create database schema if missing
        3. Process new samples and add to database
        4. Optionally overwrite existing entries
        5. Commit all changes to database
    """

    metadata_path = "metadata.csv"

    with file_system_manager.get_buffered_reader(metadata_path) as metadata_reader:
        metadata_df = read_metadata_and_calculate_hash(metadata_reader, file_system_manager, n_jobs=n_jobs)

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        existing_in_db_hashes_of_audio = session.scalars(select(AudioToASRText.audio_md5_hash)).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        samples_asr_text_info = process_selected_samples(
            dataframe=samples_to_add,
            file_system_manager=file_system_manager,
            batch_size=batch_size,
            triton_address=triton_address,
            triton_port=triton_port,
            n_jobs=n_jobs,
        )
        session.add_all(samples_asr_text_info)

        if overwrite:
            print("Overwriting others samples")
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
            samples_asr_text_info = process_selected_samples(
                dataframe=samples_to_update,
                file_system_manager=file_system_manager,
                batch_size=batch_size,
                triton_address=triton_address,
                triton_port=triton_port,
                n_jobs=n_jobs,
            )
            samples_asr_text_info = [dataclasses.asdict(info) for info in samples_asr_text_info if info is not None]
            session.execute(update(AudioToASRText), samples_asr_text_info)

        session.commit()


def process_selected_samples(
    dataframe: pd.DataFrame,
    file_system_manager: AbstractFileSystemManager,
    batch_size: int,
    triton_address: str,
    triton_port: int,
    n_jobs: int,
):
    """Process selected audio samples and prepare database records.

    Args:
        - dataframe: DataFrame containing audio metadata
        - file_system_manager: Storage manager for audio files
        - batch_size: Number of files per processing batch
        - triton_address: Triton server address
        - triton_port: Triton server port
        - n_jobs: Number of parallel jobs

    Returns:
        list: AudioToASRText objects ready for database insertion

    Note:
        - Uses parallel processing with progress tracking
        - Filters out samples with empty ASR results
        - Converts results to database model instances
    """

    files = dataframe["path_to_wav"].values
    status_bar = tqdm(files, total=len(files), desc=f"Processing audio files in batches. (Batch size: {batch_size})")

    batches = [files[i : min(i + batch_size, len(files))] for i in range(0, len(files), batch_size)]

    list_of_recognized_texts = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(status_bar, batch_size)(process_audios))(
            input_batch=batch,
            file_system_manager=file_system_manager,
            triton_address=triton_address,
            triton_port=triton_port,
        )
        for batch in batches
    )

    recognized_texts = {path: text for batch in list_of_recognized_texts for path, text in batch.items()}
    dataframe["recognized_text"] = dataframe["path_to_wav"].map(recognized_texts)

    recognized_text_empty_mask = dataframe["recognized_text"].isnull()
    print(f"Found {sum(recognized_text_empty_mask)} samples for which ASR did not recognize any speech.")
    dataframe = dataframe[~recognized_text_empty_mask]

    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting ASR texts data")
    samples_asr_text_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(AudioToASRText))(audio_md5_hash=sample.hash, text=sample.recognized_text)
        for sample in dataframe.itertuples()
    )

    return samples_asr_text_info


if __name__ == "__main__":
    cli(obj={})