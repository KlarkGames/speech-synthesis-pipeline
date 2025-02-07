import os
import shutil
import string
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from dotenv import load_dotenv
from pydub import AudioSegment

from src.datasets.audio_folder import parce_audio_folder_to_dataset

load_dotenv()


def generate_random_audio(min_s: float, max_s: float, sr=None) -> Tuple[np.ndarray, int]:
    if sr is None:
        sr = np.random.choice([8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000])

    audio = np.random.rand(int(sr * (min_s + np.random.random() * (max_s - min_s))))
    return audio, sr


def generate_random_string(length=None) -> str:
    chars = list(string.ascii_letters + string.digits)
    if length is None:
        length = np.random.randint(5, 25)

    return "".join([np.random.choice(chars) for _ in range(length)])


@pytest.fixture(scope="module")
def raw_audio_dataset():
    path_to_dataset = os.environ.get("TEST_RAW_AUDIO_PATH")

    audio_extentions = [
        ".mp3",
        ".wav",
        ".aiff",
        ".flac",
        ".opus",
        ".ogg",
    ]

    formats_and_subtypes = {
        ".mp3": ("MP3", None),
        ".wav": ("WAV", None),
        ".aiff": ("AIFF", None),
        ".flac": ("FLAC", None),
        ".opus": ("OGG", "OPUS"),
        ".ogg": ("OGG", None),
    }

    folders = [generate_random_string() for _ in range(np.random.randint(2, 4))] + [""]
    audio_files = []

    for extention in audio_extentions:
        num_of_files = np.random.randint(1, 10)
        for _ in range(num_of_files):
            file_name = generate_random_string()

            if extention == ".opus":
                audio, sample_rate = generate_random_audio(
                    1.0, 20.0, sr=np.random.choice([8000, 12000, 16000, 24000, 48000])
                )
            else:
                audio, sample_rate = generate_random_audio(1.0, 20.0)

            selected_folder = np.random.choice(folders)
            subfolders = [generate_random_string() for _ in range(np.random.randint(0, 2))]
            save_path = os.path.join(path_to_dataset, selected_folder, *subfolders, file_name + extention)
            audio_format, audio_subtype = formats_and_subtypes[extention]

            os.makedirs(Path(save_path).parent, exist_ok=True)
            sf.write(save_path, data=audio, samplerate=sample_rate, format=audio_format, subtype=audio_subtype)

            audio_files.append(save_path)

    yield path_to_dataset, audio_files

    shutil.rmtree(path_to_dataset)


def test_single_speaker(raw_audio_dataset):
    path_to_dataset, audio_files = raw_audio_dataset
    path_to_dataset_object = Path(path_to_dataset)

    assert os.path.exists(path_to_dataset)
    for file in audio_files:
        assert os.path.isfile(file)

    processed_dataset_path = path_to_dataset + "_processed"

    parce_audio_folder_to_dataset(
        folder_path=path_to_dataset,
        save_path=processed_dataset_path,
        single_speaker=True,
    )

    assert os.path.exists(processed_dataset_path)
    assert os.path.isfile(os.path.join(processed_dataset_path, "metadata.csv"))

    new_audio_paths = []
    for file in audio_files:
        file_path_object = Path(file)
        if file_path_object.parent == path_to_dataset_object:
            new_audio_paths.append(os.path.join("speaker_0", "wavs", file_path_object.stem + ".wav"))
        else:
            new_audio_paths.append(
                os.path.join(
                    "speaker_0",
                    "wavs",
                    file_path_object.parent.relative_to(path_to_dataset),
                    file_path_object.stem + ".wav",
                )
            )

    metadata_df = pd.read_csv(os.path.join(processed_dataset_path, "metadata.csv"), sep="|")
    assert "path_to_wav" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    assert all(metadata_df["speaker_id"] == 0)

    for previous_audio_path, new_audio_path in zip(audio_files, new_audio_paths):
        absolute_new_audio_path = os.path.join(processed_dataset_path, new_audio_path)
        assert new_audio_path in metadata_df["path_to_wav"].to_list()
        assert os.path.isfile(absolute_new_audio_path)

        previous_segment = AudioSegment.from_file(previous_audio_path)
        new_segment = AudioSegment.from_file(absolute_new_audio_path)

        assert previous_segment.duration_seconds == new_segment.duration_seconds
        assert previous_segment.channels == new_segment.channels
        assert previous_segment.frame_rate == new_segment.frame_rate

        new_audio_info = sf.info(absolute_new_audio_path)

        assert new_audio_info.subtype == "PCM_16"

    shutil.rmtree(processed_dataset_path)


def test_unknown_speakers(raw_audio_dataset):
    path_to_dataset, audio_files = raw_audio_dataset
    path_to_dataset_object = Path(path_to_dataset)
    assert os.path.exists(path_to_dataset)
    for file in audio_files:
        assert os.path.isfile(file)

    processed_dataset_path = path_to_dataset + "_processed"

    parce_audio_folder_to_dataset(
        folder_path=path_to_dataset,
        save_path=processed_dataset_path,
        unknown_speakers=True,
    )

    assert os.path.exists(processed_dataset_path)
    assert os.path.isfile(os.path.join(processed_dataset_path, "metadata.csv"))

    new_audio_paths = []
    for file in audio_files:
        file_path_object = Path(file)
        if file_path_object.parent == path_to_dataset_object:
            new_audio_paths.append(os.path.join("wavs", file_path_object.stem + ".wav"))
        else:
            new_audio_paths.append(
                os.path.join(
                    "wavs", file_path_object.parent.relative_to(path_to_dataset), file_path_object.stem + ".wav"
                )
            )

    metadata_df = pd.read_csv(os.path.join(processed_dataset_path, "metadata.csv"), sep="|")
    assert "path_to_wav" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    assert all(metadata_df["speaker_id"] == -1)

    for previous_audio_path, new_audio_path in zip(audio_files, new_audio_paths):
        absolute_new_audio_path = os.path.join(processed_dataset_path, new_audio_path)
        assert new_audio_path in metadata_df["path_to_wav"].to_list()
        assert os.path.isfile(absolute_new_audio_path)

        previous_segment = AudioSegment.from_file(previous_audio_path)
        new_segment = AudioSegment.from_file(absolute_new_audio_path)

        assert previous_segment.duration_seconds == new_segment.duration_seconds
        assert previous_segment.channels == new_segment.channels
        assert previous_segment.frame_rate == new_segment.frame_rate

        new_audio_info = sf.info(absolute_new_audio_path)

        assert new_audio_info.subtype == "PCM_16"

    shutil.rmtree(processed_dataset_path)


def test_audio_folder_parser(raw_audio_dataset):
    path_to_dataset, audio_files = raw_audio_dataset
    path_to_dataset_object = Path(path_to_dataset)

    assert os.path.exists(path_to_dataset)
    for file in audio_files:
        assert os.path.isfile(file)

    processed_dataset_path = path_to_dataset + "_processed"

    parce_audio_folder_to_dataset(folder_path=path_to_dataset, save_path=processed_dataset_path)

    assert os.path.exists(processed_dataset_path)
    assert os.path.isfile(os.path.join(processed_dataset_path, "metadata.csv"))

    metadata_df = pd.read_csv(os.path.join(processed_dataset_path, "metadata.csv"), sep="|")
    assert "path_to_wav" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    file_name_to_folder = {}
    for file in audio_files:
        file_path_object = Path(file)
        if file_path_object.parent == path_to_dataset_object:
            file_name_to_folder[file_path_object.stem] = "unknown"
        else:
            file_name_to_folder[file_path_object.stem] = file_path_object.relative_to(path_to_dataset).parents[-2]

    folder_to_speaker_id = {"unknown": -1}
    for sample in metadata_df.itertuples():
        file_name = Path(sample.path_to_wav).stem
        speaker_id = sample.speaker_id

        if speaker_id == -1:
            assert "wavs" in sample.path_to_wav
        else:
            assert f"speaker_{speaker_id}/wavs" in sample.path_to_wav

        origin_folder = file_name_to_folder[file_name]
        expected_id = folder_to_speaker_id.get(origin_folder)
        if expected_id is None:
            folder_to_speaker_id[origin_folder] = speaker_id
        else:
            assert expected_id == speaker_id

        absolute_new_audio_path = os.path.join(processed_dataset_path, sample.path_to_wav)
        assert os.path.isfile(absolute_new_audio_path)

        previous_audio_path = next((file for file in audio_files if file_name in file), None)

        previous_segment = AudioSegment.from_file(previous_audio_path)
        new_segment = AudioSegment.from_file(absolute_new_audio_path)

        assert previous_segment.duration_seconds == new_segment.duration_seconds
        assert previous_segment.channels == new_segment.channels
        assert previous_segment.frame_rate == new_segment.frame_rate

        new_audio_info = sf.info(absolute_new_audio_path)

        assert new_audio_info.subtype == "PCM_16"

    shutil.rmtree(processed_dataset_path)
