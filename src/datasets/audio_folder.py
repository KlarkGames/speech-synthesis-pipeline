import os
import subprocess
from glob import glob
from pathlib import Path
from typing import List

import click
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.utils import update_bar_on_ending

audio_extentions = [
    ".mp3",
    ".mp4",
    ".wav",
    ".aiff",
    ".flac",
    ".opus",
    ".webm",
    ".ogg",
]


def parce_audio_folder_to_dataset(
    folder_path: str, save_path: str, single_speaker=False, unknown_speakers=False, overwrite=False, n_jobs=-1
):
    pd.options.display.max_columns = None

    files = parce_all_audio_files_from_directory(folder_path)
    data = pd.DataFrame(files, columns=["absolute_path_to_files"])
    data["relative_path_to_directory"] = data["absolute_path_to_files"].apply(
        lambda path: (Path(path).parent.relative_to(folder_path))
    )
    data["file_name"] = data["absolute_path_to_files"].apply(lambda path: str(Path(path).stem))
    data["speaker_dir"] = data["relative_path_to_directory"].apply(
        lambda path: Path(path).parts[0] if len(Path(path).parts) > 0 else None
    )

    if single_speaker and not unknown_speakers:
        data["speaker_id"] = 0
        data["path_to_wav"] = data.apply(
            lambda row: os.path.join(
                "speaker_0", "wavs", *Path(row.relative_path_to_directory).parts, row.file_name + ".wav"
            ),
            axis=1,
        )
    elif not single_speaker and unknown_speakers:
        data["speaker_id"] = -1
        data["path_to_wav"] = data.apply(
            lambda row: os.path.join("wavs", *Path(row.relative_path_to_directory).parts, row.file_name + ".wav"),
            axis=1,
        )
    elif not single_speaker and not unknown_speakers:
        unique_speaker_dirs = list(data[data["speaker_dir"].notnull()]["speaker_dir"].unique())
        print(unique_speaker_dirs)
        data.loc[data["speaker_dir"].isnull(), "speaker_id"] = -1
        data.loc[data["speaker_dir"].notnull(), "speaker_id"] = data[data["speaker_dir"].notnull()][
            "speaker_dir"
        ].apply(lambda dir: unique_speaker_dirs.index(dir))
        data["speaker_id"] = data["speaker_id"].astype(int)

        data.loc[data["speaker_dir"].isnull(), "path_to_wav"] = data.apply(
            lambda row: os.path.join("wavs", row.file_name + ".wav"), axis=1
        )
        data.loc[data["speaker_dir"].notnull(), "path_to_wav"] = data.apply(
            lambda row: os.path.join(
                f"speaker_{row.speaker_id}",
                "wavs",
                *Path(row.relative_path_to_directory).parts[1:],
                row.file_name + ".wav",
            ),
            axis=1,
        )
    else:
        ValueError("It cannot be specified Single Speaker and Unknown Speakers in the same time.")

    tqdm_bar = tqdm(total=len(data), desc="Saving audiofiles to structured dataset")
    Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(ffmpeg_processor))(
            audio_path=sample.absolute_path_to_files,
            save_path=os.path.join(save_path, sample.path_to_wav),
            overwrite=overwrite,
        )
        for sample in data.itertuples()
    )

    data[["path_to_wav", "speaker_id"]].to_csv(os.path.join(save_path, "metadata.csv"), sep="|", index=False)
    print(data)


def parce_all_audio_files_from_directory(folder_path: str) -> List[str]:
    files = []
    for extention in audio_extentions:
        mask = os.path.join(folder_path, f"**/*{extention}")
        files.extend(glob(mask, recursive=True))
    return files


def ffmpeg_processor(audio_path: str, save_path: str, overwrite: bool = False) -> None:
    if os.path.exists(save_path) and not overwrite:
        return

    os.makedirs(Path(save_path).parent, exist_ok=True)

    subprocess.check_output(
        f"ffmpeg -i {audio_path} -acodec pcm_s16le -ac 1 {save_path}",
        shell=True,
        stderr=subprocess.DEVNULL,
    )


@click.command("cli")
@click.option(
    "--folder-path", type=click.Path(exists=True, file_okay=False), help="Path to the folder with audio files."
)
@click.option("--single-speaker", is_flag=True, help="Is all files in this folder belongs to one speaker?")
@click.option("--unknown-speaker", is_flag=True, help="Is all files in this folder belongs to unknown speaker?")
@click.option("--overwrite", is_flag=True, help="Is it needed to overwrite existing files?")
@click.option("--save-path", type=click.Path(), help="Path where to save formated dataset.")
@click.option(
    "--n-jobs", type=int, default=-1, help="Number of parallel jobs to use while processing. -1 means to use all cores."
)
def cli(folder_path: str, single_speaker: bool, unknown_speaker: bool, overwrite: bool, save_path: str, n_jobs: int):
    parce_audio_folder_to_dataset(
        folder_path=folder_path,
        single_speaker=single_speaker,
        unknown_speaker=unknown_speaker,
        save_path=save_path,
        overwrite=overwrite,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    cli()
