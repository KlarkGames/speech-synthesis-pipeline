import dataclasses
import json
import os
import re
import subprocess
from typing import List

import click
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from textgrid import TextGrid
from tqdm import tqdm

from src.metrics_collection.models import AudioToMFAData, AudioToOriginalText, Base
from src.utils import read_metadata_and_calculate_hash, update_bar_on_ending

load_dotenv()


@click.command("main", context_settings={"show_default": True})
@click.option("--dataset-path", type=click.Path(exists=True))
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
    "--n-jobs",
    type=click.INT,
    help="Number of parallel jobs to use while processing. -1 means to use all cores.",
    default=-1,
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
) -> None:
    metadata_df = read_metadata_and_calculate_hash(metadata_path, dataset_path, n_jobs=n_jobs)

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )

    Base.metadata.create_all(engine, checkfirst=True)

    os.system("mfa model download acoustic english_us_arpa")
    os.system("mfa model download dictionary english_us_arpa")

    with Session(engine) as session:
        existing_in_db_hashes_of_audio = session.scalars(select(AudioToMFAData.audio_md5_hash)).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]

        if "text" not in samples_to_add.columns:
            id_to_text_dict = {
                sample.audio_md5_hash: sample.text
                for sample in session.execute(
                    (select(AudioToOriginalText).where(AudioToOriginalText.audio_md5_hash.in_(samples_to_add["hash"])))
                ).all()
            }

            samples_to_add["text"] = samples_to_add["hash"].map(id_to_text_dict)

        tqdm_bar = tqdm(total=len(samples_to_add), desc="Processing samples by MFA.")
        samples_to_add["mfa_textgrid_data"] = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(update_bar_on_ending(tqdm_bar)(allign_sample))(
                path_to_audio=os.path.join(dataset_path, sample.path_to_wav),
                temp_directory=os.path.join(dataset_path, "temp", str(i)),
                text=sample.text,
            )
            for i, sample in enumerate(samples_to_add.itertuples())
        )

        unprocessed_samples_mask = samples_to_add["mfa_textgrid_data"].isnull()
        print(f"{len(samples_to_add[unprocessed_samples_mask])} samples was not processed.")
        samples_to_add = samples_to_add[~unprocessed_samples_mask]

        tqdm_bar = tqdm(total=len(samples_to_add), desc="Collecting MFA textgrids data.")
        samples_mfa_textgrid_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(update_bar_on_ending(tqdm_bar)(AudioToMFAData))(
                audio_md5_hash=sample.hash, mfa_textgrid_data=sample.mfa_textgrid_data
            )
            for sample in samples_to_add.itertuples()
        )
        session.add_all(samples_mfa_textgrid_info)

        if overwrite:
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]

            if "text" not in samples_to_add.columns:
                id_to_text_dict = {
                    sample.audio_md5_hash: sample.text
                    for sample in session.execute(
                        (
                            select(AudioToOriginalText).where(
                                AudioToOriginalText.audio_md5_hash.in_(samples_to_add["hash"])
                            )
                        )
                    ).all()
                }

                samples_to_update["text"] = samples_to_update["hash"].map(id_to_text_dict)

            tqdm_bar = tqdm(total=len(samples_to_update), desc="Processing samples by MFA. (Overwrite)")
            samples_to_update["mfa_textgrid_data"] = Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(update_bar_on_ending(tqdm_bar)(allign_sample))(
                    path_to_audio=os.path.join(dataset_path, sample.path_to_wav),
                    temp_directory=os.path.join(dataset_path, "temp", str(i)),
                    text=sample.text,
                )
                for i, sample in enumerate(samples_to_update.itertuples())
            )

            unprocessed_samples_mask = samples_to_update["mfa_textgrid_data"].isnull()
            print(f"{len(samples_to_update[unprocessed_samples_mask])} samples was not processed.")
            samples_to_update = samples_to_update[~unprocessed_samples_mask]

            tqdm_bar = tqdm(total=len(samples_to_update), desc="Collecting MFA textgrids data. (Overwrite)")
            samples_mfa_textgrid_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
                delayed(update_bar_on_ending(tqdm_bar)(AudioToMFAData))(
                    audio_md5_hash=sample.hash, mfa_textgrid_data=sample.mfa_textgrid_data
                )
                for sample in samples_to_update.itertuples()
            )
            samples_asr_text_info = [dataclasses.asdict(info) for info in samples_mfa_textgrid_info if info is not None]
            session.execute(update(AudioToMFAData), samples_asr_text_info)

        session.commit()


def allign_sample(path_to_audio: str, temp_directory: str, text: str) -> str:
    text_path = path_to_audio.replace("/wavs/", "/txts/").replace(".wav", ".txt")
    save_text(text_path, text)

    textgrid_path = path_to_audio.replace("/wavs/", "/text_grids/").replace(".wav", ".TextGrid")

    try:
        subprocess.check_output(
            f"mfa align_one --clean -q --num_jobs 1 --temporary_directory {temp_directory} --profile {temp_directory} {path_to_audio} {text_path} english_us_arpa english_us_arpa {textgrid_path}",
            shell=True,
            stderr=subprocess.DEVNULL,
        )

    except Exception:
        if os.path.isfile(textgrid_path):
            print("Exception occurred but .TextGrid file created.")
        else:
            print("Exception occurred and .TextGrid file was faild to create.")
            return None

    try:
        text_grid = TextGrid.fromFile(textgrid_path)
    except FileNotFoundError:
        print(
            f"File {textgrid_path} was not found. It might be because corresponding .wav file "
            + "has mismatch with text provided in metadata.csv. MFA does not create .TextGrid files in "
            + "this scenario. Sample skipped."
        )
        return None

    word_tier = text_grid[0]
    data = [
        {
            "text": interval.mark,
            "min_time": interval.minTime,
            "max_time": interval.maxTime,
            "duration": interval.duration(),
        }
        for interval in word_tier
    ]

    json_dump = json.dumps(data)
    return json_dump


def process_text_grid_files(
    dataset_path: str, metadata: pd.DataFrame, n_jobs=-1, comma_duration: float = 0.15, period_duration: float = 0.3
) -> List[str | None]:
    dataset_path = dataset_path if dataset_path.endswith("/") else dataset_path + "/"
    text_grid_paths = dataset_path + metadata["path_to_wav"].str.replace(".wav", ".TextGrid").str.replace(
        "/wavs/", "/text_grids/"
    )

    texts = Parallel(n_jobs=n_jobs)(
        delayed(get_text_from_text_grid)(text_grid_path, comma_duration, period_duration)
        for text_grid_path in text_grid_paths
    )

    return texts


def get_text_from_text_grid(
    text_grid_path: str, comma_duration: float = 0.15, period_duration: float = 0.3
) -> str | None:
    try:
        text_grid = TextGrid.fromFile(text_grid_path)
    except FileNotFoundError:
        print(
            f"File {text_grid_path} was not found. It might be because corresponding .wav file "
            + "has mismatch with text provided in metadata.csv. MFA does not create .TextGrid files in "
            + "this scenario. Sample skipped."
        )
        return None

    word_tier = text_grid[0]

    result_text = ""

    for i, interval in enumerate(word_tier):
        duration = interval.duration()  # Получаем длительность аннотации
        label = interval.mark  # Получаем текст аннотации

        if label == "":
            if i == 0:
                continue

            if duration > period_duration:
                result_text += ". "
            elif duration > comma_duration:
                result_text += ", "
        else:
            result_text += label + " "

    result_text = result_text.strip()
    result_text += "."

    result_text = re.sub(r"[,.]{2,}", ".", result_text)  # Multiple dots and periods into one period
    result_text = re.sub(r"\s{2,}", " ", result_text)  # Multiple spaces into one space
    result_text = result_text.replace(" .", ".")
    result_text = result_text.replace(" ,", ",")

    return result_text


def save_texts_to_txt(dataset_path: str, metadata: pd.DataFrame, n_jobs=-1) -> None:
    texts = metadata["text"].str.replace("-", " ")
    dataset_path = dataset_path if dataset_path.endswith("/") else dataset_path + "/"
    paths = dataset_path + metadata["path_to_wav"].str.replace(".wav", ".txt").str.replace("/wavs/", "/txts/")
    Parallel(n_jobs=n_jobs)(delayed(save_text)(save_path, text) for save_path, text in zip(paths, texts))


def save_text(save_path: str, text: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="UTF-8") as f:
        f.write(text)


def get_list_of_directories(paths_to_wavs: pd.Series) -> pd.Series:
    return paths_to_wavs.str.rsplit("/", n=1).str[0].unique()


if __name__ == "__main__":
    main()
