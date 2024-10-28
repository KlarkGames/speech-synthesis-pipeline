import csv
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Dict

import click
import librosa
import requests
import soundfile as sf
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AudioFile:
    # There are 4 speakers in EmoV_DB dataset: bea, jenie, josh, sam
    speaker_to_speaker_id = {
        "bea": 1,
        "jenie": 2,
        "josh": 3,
        "sam": 4,
    }

    def __init__(self, filepath: str):
        assert filepath.endswith(
            ".wav"
        ), f"{filepath} is not a .wav file. Only .wav files are supported."
        self.filepath = Path(filepath)
        self.speaker, self.emotion, self.audio_id = self.parse_filepath()

    def parse_filepath(self):
        speaker_regex = r"(.*)_.*"
        emotion_audio_id_regex = r"(.+)_[0-9]+[-_][0-9]+_([0-9]+).wav"

        perent_directory = self.filepath.parent.name
        speaker = re.search(speaker_regex, perent_directory).group(1)

        filename = self.filepath.name
        emotion, audio_id = re.search(emotion_audio_id_regex, filename).groups()
        audio_id = int(audio_id)

        return speaker, emotion, audio_id

    @property
    def speaker_id(self):
        try:
            return self.speaker_to_speaker_id[self.speaker]
        except KeyError:
            logger.error(
                f"Speaker {self.speaker} not found in speaker_to_speaker_id dictionary."
            )

    @property
    def output_path_from_dataset_root(self):
        return os.path.join(
            f"speaker_{self.speaker_id}", "wavs", self.emotion, f"{self.audio_id}.wav"
        )

    def save_audio(
        self,
        output_dataset_path: str,
        change_sample_rate: bool = False,
        result_sample_rate: int = 44100,
    ):
        audio, sample_rate = librosa.load(self.filepath)

        if change_sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=result_sample_rate
            )
            sample_rate = result_sample_rate

        save_path = os.path.join(
            output_dataset_path, self.output_path_from_dataset_root
        )
        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        sf.write(save_path, audio, sample_rate)


def preprocess(
    dataset_path: str,
    output_path: str,
    cmuarctic_data_path: str = None,
    cmuarctic_url: str = "http://www.festvox.org/cmu_arctic/cmuarctic.data",
    download_cmuarctic_data: bool = False,
    change_sample_rate: bool = False,
    result_sample_rate: int = 44100,
    n_jobs: int = 4,
):
    if cmuarctic_data_path is None:
        cmuarctic_data_path = os.path.join(dataset_path, "cmuarctic.data")

    if os.path.isfile(cmuarctic_data_path):
        logger.info(
            f"Found cmuarctic.data file in dataset path. Loading it from: {cmuarctic_data_path}"
        )
        with open(cmuarctic_data_path, "r") as f:
            cmuarctic_data = f.read()
    else:
        logger.info(
            f"Not found cmuarctic.data file in dataset path. Downloading it from: {cmuarctic_url}"
        )
        try:
            cmuarctic_data = requests.get(cmuarctic_url).text
            logger.info(f"Downloaded cmuarctic.data file from: {cmuarctic_url}")
        except Exception as e:
            logger.error(f"Failed to download cmuarctic.data file. Error: {e}")
            exit(1)

        if download_cmuarctic_data:
            logger.info(f"Saving cmuarctic.data file to: {cmuarctic_data_path}")
            with open(os.path.join(dataset_path, cmuarctic_data_path), "w") as f:
                f.write(cmuarctic_data)
            logger.info(f"Saved cmuarctic.data file to: {cmuarctic_data_path}")

    audio_id_to_text = get_audio_id_to_text(cmuarctic_data)

    audio_files = [
        AudioFile(filepath)
        for filepath in glob(os.path.join(dataset_path, "*_*", "*.wav"))
    ]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger.info(
        f"Found {len(audio_files)} audio files in dataset path. Saving them to: {output_path}"
    )
    Parallel(n_jobs=n_jobs)(
        delayed(audio_file.save_audio)(
            output_path, change_sample_rate, result_sample_rate
        )
        for audio_file in audio_files
    )

    logger.info(f"Saved {len(audio_files)} audio files to: {output_path}")

    logger.info(f"Saving metadata.csv file to: {output_path}")
    with open(os.path.join(output_path, "metadata.csv"), "w") as f:
        writer = csv.DictWriter(f, ["path_to_wav", "speaker_id", "emotion_id"])

        data = [
            {
                "path_to_wav": audio_file.output_path_from_dataset_root,
                "speaker_id": audio_file.speaker_id,
                "emotion_id": audio_id_to_text[audio_file.audio_id],
            }
            for audio_file in audio_files
        ]

        writer.writerows(data)
    logger.info(f"Saved metadata.csv file to: {output_path}")


def get_audio_id_to_text(cmuarctic_data: str) -> Dict[int, str]:
    """
    Parse 'cmuarctic.data' file and return a dictionary where keys are audio IDs from EmoV_DB dataset and values are corresponding texts.

    There are 2 types of arctics there: arctic_aXXXX and arctic_bXXXX.
    Only arctic_aXXXX is used in EmoV_DB dataset, so arctic_bXXXX is ignored.

    Args:
        cmuarctic_data (str): The contents of 'cmuarctic.data' file.

    Returns:
        Dict[int, str]: A dictionary where keys are audio IDs and values are corresponding texts.
    """
    audio_id_to_text: dict = {}

    regex = r'\( arctic_a([0-9]*?) \"(.*?)" \)'

    for line in cmuarctic_data.split("\n"):
        match = re.search(regex, line)
        if match:
            audio_id, text = match.groups()
            audio_id_to_text[int(audio_id)] = text

    return audio_id_to_text


@click.command("main", context_settings={"show_default": True})
@click.option(
    "--dataset_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to EmoV_DB dataset",
)
@click.option(
    "--output_path", type=click.Path(), required=True, help="Path to output directory"
)
@click.option(
    "--cmuarctic_data_path",
    type=str,
    default=None,
    help="Path to 'cmuarctic.data' file with texts for audiofiles.",
)
@click.option(
    "--cmuarctic_url",
    default="http://www.festvox.org/cmu_arctic/cmuarctic.data",
    help="Path to 'cmuarctic.data' file url to be able to download this file if it doesn't exist.",
)
@click.option(
    "--download_cmuarctic_data",
    default=False,
    help="Download 'cmuarctic.data' file if it doesn't exist to input dataset path.",
)
@click.option(
    "--change_sample_rate",
    default=False,
    help="Resample all audiofiles to specified sample rate.",
)
@click.option(
    "--result_sample_rate",
    default=44100,
    help="The sample rate to resample output audiofiles.",
)
@click.option("--n_jobs", default=4, help="Number of parallel jobs.")
def main(
    dataset_path: str,
    output_path: str,
    cmuarctic_data_path: str,
    cmuarctic_url: str,
    download_cmuarctic_data: bool,
    change_sample_rate: bool,
    result_sample_rate: int,
    n_jobs: int,
):
    preprocess(
        dataset_path=dataset_path,
        output_path=output_path,
        cmuarctic_data_path=cmuarctic_data_path,
        cmuarctic_url=cmuarctic_url,
        download_cmuarctic_data=download_cmuarctic_data,
        change_sample_rate=change_sample_rate,
        result_sample_rate=result_sample_rate,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    main()
