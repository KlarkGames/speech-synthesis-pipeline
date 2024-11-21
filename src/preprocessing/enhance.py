import asyncio
import os
from random import randint

import click
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pytriton.client import AsyncioModelClient
from scipy.io.wavfile import write
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

            audio_data, sample_rate = librosa.load(input_path, sr=None)
            audio_data = audio_data.reshape(1, -1)

            results[output_path] = tg.create_task(
                client.infer_sample(
                    INPUT_AUDIOS=audio_data,
                    SAMPLE_RATE=np.asarray([sample_rate]),
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

        write(output_path, 44100, output.result()["OUTPUT_AUDIOS"])

    tqdm_bar.update(len(input_batch))


@click.command()
@click.option("--dataset_path", help="Path to processing dataset.")
@click.option("--output_path", help="Path where the enhanced dataset will be saved.")
@click.option(
    "--chunk-duration",
    type=float,
    default=30.0,
    show_default=True,
    help="The duration in seconds by which the enhancer will divide your sample.",
)
@click.option("--chunk_overlap", type=float, default=1.0, show_default=True, help="The duration of overlap between adjacent samples. Does not enlarge chunk_duration.")
@click.option("--model_name", default="enhancer_ensemble", show_default=True, help="The name of Triton Inference Server model.")
@click.option("--batch_size", type=int, default=10, show_default=True, help="The size of the batch of async tasks every job will process")
@click.option("--triton_address", help="The Triton Inference Server address")
@click.option("--triton_port", type=int, help="The Triton Inference Server port")
@click.option("--n_jobs", type=int, default=-1, show_default=True, help="Number of parallel jobs. If -1 specified, use all available CPU cores.")
def process_dataset(
    dataset_path,
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

    metadata_df = pd.read_csv(os.path.join(dataset_path, "metadata.csv"), sep="|")

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
