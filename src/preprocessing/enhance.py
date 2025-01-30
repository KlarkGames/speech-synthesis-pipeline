import asyncio
import os
from random import randint

import click
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from pydub import AudioSegment
from pytriton.client import AsyncioModelClient
from tqdm import tqdm


async def send_data_to_enhancer(
    input_batch,
    dataset_dir,
    save_dir,
    triton_address,
    triton_port,
    model_name,
    chunk_duration=30.0,
    chunk_overlap=1.0,
):
    await asyncio.sleep(randint(10, 90) / 100)  # To prevent processes beeing intersected
    results = {}
    client = AsyncioModelClient(f"{triton_address}:{triton_port}", model_name, inference_timeout_s=600)

    async with asyncio.TaskGroup() as tg:
        for file in input_batch:
            input_path = os.path.join(dataset_dir, file)
            output_path = os.path.join(save_dir, file)

            if os.path.exists(output_path):
                continue

            audio = AudioSegment.from_wav(input_path)
            audio = audio.set_channels(1)

            audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)

            results[output_path] = tg.create_task(
                client.infer_sample(
                    INPUT_AUDIO=audio_data,
                    SAMPLE_RATE=np.asarray([audio.frame_rate]),
                    CHUNK_DURATION_S=np.asarray([chunk_duration], dtype=np.float32),
                    CHUNK_OVERLAP_S=np.asarray([chunk_overlap], dtype=np.float32),
                )
            )

    await client.close()

    return results


def process_audio_file(
    input_batch,
    dataset_dir,
    save_dir,
    tqdm_bar: tqdm,
    chunk_duration=30.0,
    chunk_overlap=1.0,
    model_name="enhancer_ensemble",
    triton_address="localhost",
    triton_port=8000,
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(
        send_data_to_enhancer(
            input_batch,
            dataset_dir,
            save_dir,
            triton_address,
            triton_port,
            model_name,
            chunk_duration,
            chunk_overlap,
        )
    )

    for output_path, output in results.items():
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sf.write(output_path, output.result()["OUTPUT_AUDIO"], 44100)

    tqdm_bar.update(len(input_batch))


@click.command()
@click.option("--dataset_path", help="Path to processing dataset.")
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .csv file with metadata.",
    callback=lambda context, _, value: value if value else os.path.join(context.params["dataset_path"], "metadata.csv"),
)
@click.option("--output_path", help="Path where the enhanced dataset will be saved.")
@click.option(
    "--chunk_duration",
    type=float,
    default=30.0,
    show_default=True,
    help="The duration in seconds by which the enhancer will divide your sample.",
)
@click.option(
    "--chunk_overlap",
    type=float,
    default=1.0,
    show_default=True,
    help="The duration of overlap between adjacent samples. Does not enlarge chunk_duration.",
)
@click.option(
    "--model_name", default="enhancer_ensemble", show_default=True, help="The name of Triton Inference Server model."
)
@click.option(
    "--batch_size",
    type=int,
    default=10,
    show_default=True,
    help="The size of the batch of async tasks every job will process",
)
@click.option("--triton_address", help="The Triton Inference Server address")
@click.option("--triton_port", type=int, help="The Triton Inference Server port")
@click.option(
    "--n_jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Number of parallel jobs. If -1 specified, use all available CPU cores.",
)
def process_dataset(
    dataset_path,
    metadata_path,
    output_path,
    chunk_duration=30.0,
    chunk_overlap=1.0,
    model_name="enhancer_ensemble",
    batch_size=10,
    triton_address="localhost",
    triton_port=8000,
    n_jobs=-1,
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    metadata_df = pd.read_csv(metadata_path, sep="|")

    files = metadata_df["path_to_wav"].sample(frac=1).values

    status_bar = tqdm(files, total=len(files), desc="Processing audio files")

    batches = [files[i : min(i + batch_size, len(files))] for i in range(0, len(files), batch_size)]

    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(process_audio_file)(
            input_batch=batch,
            dataset_dir=dataset_path,
            save_dir=output_path,
            tqdm_bar=status_bar,
            chunk_duration=chunk_duration,
            chunk_overlap=chunk_overlap,
            model_name=model_name,
            triton_address=triton_address,
            triton_port=triton_port,
        )
        for batch in batches
    )

    metadata_df.to_csv(os.path.join(output_path, "metadata.csv"), sep="|", index=False)


if __name__ == "__main__":
    process_dataset()
