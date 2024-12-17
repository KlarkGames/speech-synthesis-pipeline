import os
import re
from typing import List

import click
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from textgrid import TextGrid
from tqdm import tqdm


@click.command("main", context_settings={"show_default": True})
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--n_jobs",
    type=click.INT,
    help="Number of parallel jobs to use while processing. -1 means to use all cores.",
    default=-1,
)
@click.option(
    "--comma_duration",
    type=click.FloatRange(min=0, max_open=True),
    help="Duration of pause which will be indicated with comma.",
    default=0.15,
)
@click.option(
    "--period_duration",
    type=click.FloatRange(min=0, max_open=True),
    help="Duration of pause which will be indicated with period.",
    default=0.3,
)
def main(input_path: str, n_jobs: int, comma_duration: float, period_duration: float) -> None:
    if n_jobs == -1:
        n_jobs = cpu_count()

    metadata_path = os.path.join(input_path, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path, sep="|")

    if not os.path.exists(os.path.join(input_path, "metadata_before_MFA.csv")):
        metadata_df.to_csv(os.path.join(input_path, "metadata_before_MFA.csv"), sep="|", index=False)

    os.system("mfa model download acoustic english_us_arpa")
    os.system("mfa model download dictionary english_us_arpa")

    directories_to_process = get_list_of_directories(metadata_df["path_to_wav"])

    progress_bar = tqdm(total=len(metadata_df))
    for directory in directories_to_process:
        processing_files = metadata_df[metadata_df["path_to_wav"].str.startswith(directory)]
        save_texts_to_txt(dataset_path=input_path, metadata=processing_files, n_jobs=n_jobs)

        wavs_directory_path = os.path.join(input_path, directory)
        txt_directory_path = wavs_directory_path.replace("wavs", "txts")
        text_grid_directory_path = wavs_directory_path.replace("wavs", "text_grids")

        os.system(
            f"mfa align --clean --single_speaker --include_original_text --num_jobs {n_jobs} "
            + f"--audio_directory {wavs_directory_path} {txt_directory_path} english_us_arpa english_us_arpa {text_grid_directory_path}"
        )

        metadata_df.loc[metadata_df["path_to_wav"].str.startswith(directory), "text"] = process_text_grid_files(
            dataset_path=input_path,
            metadata=processing_files,
            comma_duration=comma_duration,
            period_duration=period_duration,
            n_jobs=n_jobs,
        )

        os.system(f"rm -rf {txt_directory_path} {text_grid_directory_path}")

        progress_bar.update(len(processing_files))

    metadata_df.drop(metadata_df[metadata_df["text"] == ""].index)
    metadata_df.to_csv(metadata_path, sep="|", index=False)


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
