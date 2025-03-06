"""
Audio Enhancement Pipeline with Triton Inference Server

This script provides a CLI for processing audio files through an enhancer model deployed on
Triton Inference Server. It supports multiple storage backends (local filesystem, LakeFS/S3)
and parallel processing of audio files.

The pipeline works as follows:
1. Reads metadata to get list of audio files
2. Processes files in parallel batches using Triton Inference Server
3. Saves enhanced audio files to specified output location
4. Copies metadata file to destanation storage.

Example usage:
    python script.py local-to-local --input-path ./input --output-path ./output
    --triton-address localhost --triton-port 8000

Classes:
    TritonInfo: Holds configuration for connecting to Triton Inference Server

Functions:
    - send_data_to_enhancer: Async function to process batch through Triton
    - process_audio_file: Wrapper for async processing
    - process_dataset: Main parallel processing logic
    - CLI commands: Handle different input/output storage configurations
"""

import asyncio
from contextlib import asynccontextmanager
from random import randint

import click
import numpy as np
import pandas as pd
import soundfile as sf
from dotenv import load_dotenv
from dataclasses import dataclass
from joblib import Parallel, delayed
from pydub import AudioSegment
from pytriton.client import AsyncioModelClient
from tqdm import tqdm

from src.data_managers import AbstractFileSystemManager, LakeFSFileSystemManager, LocalFileSystemManager
from src.utils import update_bar_on_ending

load_dotenv()


@dataclass
class TritonInfo:
    """Configuration container for Triton Inference Server connection parameters.

    Attributes:
        model_name: Name of the Triton model to use for inference
        chunk_duration: Length (seconds) of audio chunks to process
        chunk_overlap: Overlap (seconds) between consecutive chunks
        triton_address: IP address or hostname of Triton server
        triton_port: Port number for Triton server
    """

    model_name: str
    chunk_duration: float
    chunk_overlap: float
    triton_address: str
    triton_port: int

    @asynccontextmanager
    async def get_client(self):
        """Async ctx manager for Triton client connection.

        Yields:
            AsyncioModelClient: Connected Triton client instance

        Example:
            async with triton_info.get_client() as client:
                await client.infer_sample(...)
        """

        client = AsyncioModelClient(
            f"{self.triton_address}:{self.triton_port}", self.model_name, inference_timeout_s=600
        )
        try:
            yield client
        finally:
            await client.close()


async def send_data_to_enhancer(
    input_batch,
    input_manager: AbstractFileSystemManager,
    output_manager: AbstractFileSystemManager,
    triton_info: TritonInfo,
):
    """Processes a batch of audio files through Triton Inference Server.

    Args:
        input_batch: List of audio file paths to process
        input_manager: File system manager for input files
        output_manager: File system manager for output files
        triton_info: Triton connection configuration

    Returns:
        dict: Mapping of file paths to their enhancement tasks

    Note:
        Uses async TaskGroup for concurrent inference requests
        Skips files that already exist in output location
    """

    await asyncio.sleep(randint(10, 90) / 100)  # To prevent processes beeing intersected
    results = {}

    async with triton_info.get_client() as client, asyncio.TaskGroup() as tg:
        for file in input_batch:
            if output_manager.is_path_exists(output_manager.get_absolute_path(file)):
                continue

            with input_manager.get_buffered_reader(file) as reader:
                audio = AudioSegment.from_wav(reader)

            audio = audio.set_channels(1)
            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)

            results[file] = tg.create_task(
                client.infer_sample(
                    INPUT_AUDIO=audio_data,
                    SAMPLE_RATE=np.asarray([audio.frame_rate]),
                    CHUNK_DURATION_S=np.asarray([triton_info.chunk_duration], dtype=np.float32),
                    CHUNK_OVERLAP_S=np.asarray([triton_info.chunk_overlap], dtype=np.float32),
                )
            )

    return results


def process_audio_files(
    input_batch,
    input_manager: AbstractFileSystemManager,
    output_manager: AbstractFileSystemManager,
    triton_info: TritonInfo,
):
    """Processes audio batch and saves enhanced results.

    Args:
        input_batch: List of audio file paths to process
        input_manager: File system manager for input files
        output_manager: File system manager for output files
        triton_info: Triton connection configuration

    Note:
        Creates new event loop for async processing
        Writes enhanced audio as 44.1kHz mono WAV files
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(send_data_to_enhancer(input_batch, input_manager, output_manager, triton_info))

    for file, triton_output in results.items():
        with output_manager.get_buffered_writer(file) as writer:
            sf.write(writer, triton_output.result()["OUTPUT_AUDIO"], 44100)


@click.group()
@click.option(
    "--chunk-duration",
    type=float,
    default=30.0,
    show_default=True,
    help="The duration in seconds by which the enhancer will divide your sample.",
)
@click.option(
    "--chunk-overlap",
    type=float,
    default=1.0,
    show_default=True,
    help="The duration of overlap between adjacent samples. Does not enlarge chunk_duration.",
)
@click.option(
    "--model-name", default="enhancer_ensemble", show_default=True, help="The name of Triton Inference Server model."
)
@click.option(
    "--batch-size",
    type=int,
    default=10,
    show_default=True,
    help="The size of the batch of async tasks every job will process",
)
@click.option("--triton-address", help="The Triton Inference Server address", envvar="ENHANCER_TRITON_ADDRESS")
@click.option("--triton-port", type=int, help="The Triton Inference Server port", envvar="ENHANCER_TRITON_HTTP_PORT")
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Number of parallel jobs. If -1 specified, use all available CPU cores.",
)
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .csv file with metadata.",
    default=None,
)
@click.pass_context
def cli(
    ctx: click.Context,
    chunk_duration: float,
    chunk_overlap: float,
    model_name: str,
    batch_size: int,
    triton_address: str,
    triton_port: int,
    n_jobs: int,
    metadata_path: str,
):
    """Main CLI entry point for audio enhancement pipeline.

    Configures processing parameters and Triton connection info.
    """

    ctx.ensure_object(dict)

    ctx.obj["triton_info"] = TritonInfo(
        model_name=model_name,
        chunk_duration=chunk_duration,
        chunk_overlap=chunk_overlap,
        triton_address=triton_address,
        triton_port=triton_port,
    )

    ctx.obj["batch_size"] = batch_size
    ctx.obj["n_jobs"] = n_jobs
    ctx.obj["metadata_path"] = metadata_path


@cli.command()
@click.option("--input-path", help="Path to processing dataset.")
@click.option("--output-path", help="Path where the enhanced dataset will be saved.")
@click.pass_context
def local_to_local(ctx: click.Context, input_path: str, output_path: str):
    """Process files from local filesystem to local output.

    Args:
        input_path: Local directory containing input audio
        output_path: Local directory to save enhanced audio

    Uses metadata.csv from input directory unless specified.
    """

    input_manager = LocalFileSystemManager(input_path)
    output_manager = LocalFileSystemManager(output_path)

    if ctx.obj["metadata_path"] is not None:
        metadata_df = pd.read_csv(ctx.obj["metadata_path"], sep="|")
    else:
        with input_manager.get_buffered_reader("metadata.csb") as reader:
            metadata_df = pd.read_csv(reader, sep="|")

    process_dataset(
        batch_size=ctx.obj["batch_size"],
        n_jobs=ctx.obj["n_jobs"],
        triton_info=ctx.obj["triton_info"],
        input_manager=input_manager,
        output_manager=output_manager,
        metadata_df=metadata_df,
    )


@cli.command()
@click.option("--input-path", help="Path to processing dataset.")
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--output-repository-name", type=str, help="Name of LakeFS repository where to store Enhanced data.")
@click.option("--output-branch-name", type=str, help="Name of the branch where to store Enhanced data.", default="main")
@click.pass_context
def local_to_s3(
    ctx: click.Context,
    input_path: str,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    output_repository_name: str,
    output_branch_name: str,
):
    """Process files from local filesystem to LakeFS/S3 storage.

    Requires LakeFS connection credentials and repository info.
    """

    input_manager = LocalFileSystemManager(input_path)
    output_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=output_repository_name,
        lakefs_branch_name=output_branch_name,
    )

    if ctx.obj["metadata_path"] is not None:
        metadata_df = pd.read_csv(ctx.obj["metadata_path"], sep="|")
    else:
        with input_manager.get_buffered_reader("metadata.csb") as reader:
            metadata_df = pd.read_csv(reader, sep="|")

    process_dataset(
        batch_size=ctx.obj["batch_size"],
        n_jobs=ctx.obj["n_jobs"],
        triton_info=ctx.obj["triton_info"],
        input_manager=input_manager,
        output_manager=output_manager,
        metadata_df=metadata_df,
    )


@cli.command()
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--output-repository-name", type=str, help="Name of LakeFS repository where to store Enhanced data.")
@click.option("--output-branch-name", type=str, help="Name of the branch where to store Enhanced data.", default="main")
@click.option("--input-repository-name", type=str, help="Name of LakeFS repository where processing dataset is stored.")
@click.option(
    "--input-branch-name", type=str, help="Name of the branch where processing dataset is stored.", default="main"
)
@click.pass_context
def s3_to_s3(
    ctx: click.Context,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    output_repository_name: str,
    output_branch_name: str,
    input_repository_name: str,
    input_branch_name: str,
):
    """Process files between LakeFS/S3 repositories.

    Uses same credentials for input and output repositories.
    """

    input_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=input_repository_name,
        lakefs_branch_name=input_branch_name,
    )
    output_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=output_repository_name,
        lakefs_branch_name=output_branch_name,
    )

    if ctx.obj["metadata_path"] is not None:
        metadata_df = pd.read_csv(ctx.obj["metadata_path"], sep="|")
    else:
        with input_manager.get_buffered_reader("metadata.csv") as reader:
            metadata_df = pd.read_csv(reader, sep="|")

    process_dataset(
        batch_size=ctx.obj["batch_size"],
        n_jobs=ctx.obj["n_jobs"],
        triton_info=ctx.obj["triton_info"],
        input_manager=input_manager,
        output_manager=output_manager,
        metadata_df=metadata_df,
    )


@cli.command()
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--input-repository-name", type=str, help="Name of LakeFS repository where processing dataset is stored.")
@click.option(
    "--input-branch-name", type=str, help="Name of the branch where processing dataset is stored.", default="main"
)
@click.option("--output-path", help="Path where the enhanced dataset will be saved.")
@click.pass_context
def s3_to_local(
    ctx: click.Context,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    input_repository_name: str,
    input_branch_name: str,
    output_path: str,
):
    """Process files from LakeFS/S3 to local filesystem.

    Requires LakeFS credentials for input repository.
    """
    input_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=input_repository_name,
        lakefs_branch_name=input_branch_name,
    )
    output_manager = LocalFileSystemManager(output_path)

    if ctx.obj["metadata_path"] is not None:
        metadata_df = pd.read_csv(ctx.obj["metadata_path"], sep="|")
    else:
        with input_manager.get_buffered_reader("metadata.csb") as reader:
            metadata_df = pd.read_csv(reader, sep="|")

    process_dataset(
        batch_size=ctx.obj["batch_size"],
        n_jobs=ctx.obj["n_jobs"],
        triton_info=ctx.obj["triton_info"],
        input_manager=input_manager,
        output_manager=output_manager,
        metadata_df=metadata_df,
    )


def process_dataset(
    batch_size: int,
    n_jobs: int,
    triton_info: TritonInfo,
    input_manager: AbstractFileSystemManager,
    output_manager: AbstractFileSystemManager,
    metadata_df: pd.DataFrame,
):
    """Orchestrates parallel processing of audio files.

    Args:
        batch_size: Number of files per processing batch
        n_jobs: Number of parallel jobs to use
        triton_info: Triton connection configuration
        input_manager: File system manager for input
        output_manager: File system manager for output
        metadata_df: DataFrame containing audio file metadata

    Note:
        Processes files in random order using sampled DataFrame
        Updates progress bar during processing
        Generates output metadata.csv with original structure
    """

    files = metadata_df["path_to_wav"].sample(frac=1).values

    batches = [files[i : min(i + batch_size, len(files))] for i in range(0, len(files), batch_size)]

    status_bar = tqdm(files, total=len(files), desc="Processing audio files")
    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(status_bar, n=len(batch))(process_audio_files))(
            input_batch=batch, input_manager=input_manager, output_manager=output_manager, triton_info=triton_info
        )
        for batch in batches
    )

    with output_manager.get_buffered_writer("metadata.csv") as writer:
        metadata_df.to_csv(writer, sep="|", index=False)


if __name__ == "__main__":
    cli(obj={})
