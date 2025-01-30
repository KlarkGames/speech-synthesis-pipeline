import asyncio
import dataclasses
import os

import click
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from pydub import AudioSegment
from pytriton.client import AsyncioModelClient
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.metrics_collection.models import AudioToASRText, Base
from src.utils import read_metadata_and_calculate_hash, update_bar_on_ending

load_dotenv()


async def get_texts_from_audio_by_asr(triton_address, triton_port, dataset_dir, input_batch):
    results = {}
    pending_responces = {}

    client = AsyncioModelClient(f"{triton_address}:{triton_port}", "ensemble_english_stt", inference_timeout_s=600)

    async with asyncio.TaskGroup() as tg:
        for input_file in input_batch:
            input_path = os.path.join(dataset_dir, input_file)
            txt_path = input_path.replace("/wavs", "/asr_recognized_texts").replace(".wav", ".txt")

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="UTF-8") as text_file:
                    text = text_file.read()
                results[input_file] = text
            else:
                audio = AudioSegment.from_wav(input_path).set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float64)

                result = tg.create_task(client.infer_sample(audio_signal=audio_data))
                pending_responces[input_file] = result  # .tolist()[0]

    await client.close()

    for input_file, responce in pending_responces.items():
        input_path = os.path.join(dataset_dir, input_file)
        txt_path = input_path.replace("/wavs", "/asr_recognized_texts").replace(".wav", ".txt")

        text = responce.result()["decoded_texts"].decode("UTF-8")

        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="UTF-8") as text_file:
            text_file.write(text)

        results[input_file] = text

    return results


def process_audios(input_batch, dataset_dir, triton_address, triton_port):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    recognized_texts = loop.run_until_complete(
        get_texts_from_audio_by_asr(
            input_batch=input_batch,
            dataset_dir=dataset_dir,
            triton_address=triton_address,
            triton_port=triton_port,
        )
    )

    return recognized_texts


@click.command()
@click.option("--dataset-path", help="Path to the dataset containing audio files.")
@click.option(
    "--metadata-path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to .csv file with metadata.",
    callback=lambda context, _, value: value if value else os.path.join(context.params["dataset_path"], "metadata.csv"),
)
@click.option("--triton_address", default="localhost", help="Address of the Triton Inference Server.")
@click.option("--triton-port", type=int, default=8000, help="Port of the Triton Inference Server.")
@click.option("--batch-size", type=int, default=10, help="Batch size for processing audio files.")
@click.option("--overwrite", type=bool, help="Is to overwrite existing metrics or not.", default=False)
@click.option("--database-address", type=str, help="Address of the database", envvar="POSTGRES_ADDRESS")
@click.option("--database-port", type=int, help="Port of the database", envvar="POSTGRES_PORT")
@click.option("--database-user", type=str, help="Username to use for database authentication", envvar="POSTGRES_USER")
@click.option(
    "--database-password", type=str, help="Password to use for database authentication", envvar="POSTGRES_PASSWORD"
)
@click.option("--database-name", type=str, help="Name of the database", envvar="POSTGRES_DB")
@click.option(
    "--n_jobs", type=int, default=-1, help="Number of parallel jobs to use while processing. -1 means to use all cores."
)
def process_dataset(
    dataset_path: str,
    metadata_path: str,
    triton_address: str,
    triton_port: int,
    batch_size: int,
    overwrite: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
):
    metadata_df = read_metadata_and_calculate_hash(metadata_path, dataset_path, n_jobs=n_jobs)

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        existing_in_db_hashes_of_audio = session.scalars(select(AudioToASRText.audio_md5_hash)).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        samples_asr_text_info = process_selected_samples(
            dataframe=samples_to_add,
            batch_size=batch_size,
            dataset_path=dataset_path,
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
                batch_size=batch_size,
                dataset_path=dataset_path,
                triton_address=triton_address,
                triton_port=triton_port,
                n_jobs=n_jobs,
            )
            samples_asr_text_info = [dataclasses.asdict(info) for info in samples_asr_text_info if info is not None]
            session.execute(update(AudioToASRText), samples_asr_text_info)

        session.commit()


def process_selected_samples(
    dataframe: pd.DataFrame, batch_size: int, dataset_path: str, triton_address: str, triton_port: int, n_jobs: int
):
    files = dataframe["path_to_wav"].values
    status_bar = tqdm(files, total=len(files), desc=f"Processing audio files in batches. (Batch size: {batch_size})")

    batches = [files[i : min(i + batch_size, len(files))] for i in range(0, len(files), batch_size)]

    list_of_recognized_texts = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(status_bar, batch_size)(process_audios))(
            input_batch=batch,
            dataset_dir=dataset_path,
            triton_address=triton_address,
            triton_port=triton_port,
        )
        for batch in batches
    )

    recognized_texts = {path: text for batch in list_of_recognized_texts for path, text in batch.items()}
    dataframe["recognized_text"] = dataframe["path_to_wav"].map(recognized_texts)

    recognized_text_empty_mask = dataframe["recognized_text"].isnull()
    print(f"Found {sum(recognized_text_empty_mask)} samples for which ASR did not recognized any speech.")
    dataframe = dataframe[~recognized_text_empty_mask]

    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting ASR texts data")
    samples_asr_text_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(AudioToASRText))(audio_md5_hash=sample.hash, text=sample.recognized_text)
        for sample in dataframe.itertuples()
    )

    return samples_asr_text_info


if __name__ == "__main__":
    process_dataset()
