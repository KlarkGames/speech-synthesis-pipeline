import os
import pickle
from glob import glob
from pathlib import Path
from typing import Dict, List, Self

import click
import librosa
import pandas as pd
import soundfile as sf
from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm


class AudioInfo:
    def __init__(self, path_to_opus: str) -> None:
        self.path_to_opus = Path(path_to_opus)
        self.processed = False
        self._book = self.path_to_opus.parent.name
        self.name = self.path_to_opus.stem
        self.path_from_speaker = os.path.join(self._book, self.name + ".wav")

    def save_audio(self, save_dir: str, change_sample_rate: bool = False, result_sample_rate: int = 44100) -> None:
        audio, sample_rate = librosa.load(self.path_to_opus)
        if change_sample_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=result_sample_rate)
            sample_rate = result_sample_rate

        save_path = os.path.join(save_dir, self.path_from_speaker)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, audio, sample_rate)
        self.processed = True

    def store(self) -> Dict:
        return {"path_to_opus": self.path_to_opus, "processed": self.processed}

    @staticmethod
    def load(data: Dict) -> Self:
        assert "path_to_opus" in data.keys() and "processed" in data.keys()
        audio_info = AudioInfo(data["path_to_opus"])
        audio_info.processed = data["processed"]
        return audio_info


class SpeakerInfo:
    def __init__(self, id: str, path_to_speaker_folder: str) -> None:
        self._id = id
        self._path_to_speaker = Path(path_to_speaker_folder)
        self.split = self._path_to_speaker.parent.parent.name

        assert self.split in [
            "train",
            "dev",
            "test",
        ], f"Split {self.split} not supported. Only train, dev and test splits are supported."

        self.files: List[AudioInfo] = self.find_audios()

    def find_audios(self) -> List[AudioInfo]:
        all_opus_files = glob(os.path.join(self._path_to_speaker, "*", "*.opus"))
        files = [AudioInfo(path_to_opus) for path_to_opus in all_opus_files]
        return files

    @property
    def processed(self) -> bool:
        return all([audio.processed for audio in self.files])

    def process(
        self, save_path: str, n_of_files: int, change_sample_rate: bool = False, result_sample_rate: int = 44100
    ) -> None:
        speaker_dir = os.path.join(save_path, f"speaker_{self._id}")
        audio_path = os.path.join(speaker_dir, "wavs", self.split)
        os.makedirs(audio_path, exist_ok=True)

        if self.processed:
            return

        unprocessed_files = [audio for audio in self.files if not audio.processed]
        if len(unprocessed_files) < n_of_files:
            files_to_process = unprocessed_files
        else:
            files_to_process = unprocessed_files[:n_of_files]

        for audio in files_to_process:
            audio.save_audio(audio_path, change_sample_rate, result_sample_rate)

    def get_metadata(self) -> List[Dict]:
        result = []
        for audio in self.files:
            if audio.processed:
                result.append(
                    {
                        "path_to_wav": os.path.join(f"speaker_{self._id}", "wavs", self.split, audio.path_from_speaker),
                        "audio_name": audio.name,
                        "speaker_id": self._id,
                    }
                )
        return result

    def store(self) -> Dict:
        return {
            "id": self._id,
            "path_to_speaker": self._path_to_speaker,
            "split": self.split,
            "files": [audio.store() for audio in self.files],
        }

    @staticmethod
    def load(data: Dict) -> Self:
        assert (
            "id" in data.keys()
            and "path_to_speaker" in data.keys()
            and "split" in data.keys()
            and "files" in data.keys()
        )
        speaker_info = SpeakerInfo(data["id"], data["path_to_speaker"])
        speaker_info.split = data["split"]
        speaker_info.files = [AudioInfo.load(file) for file in data["files"]]
        return speaker_info


def cache_speakers(speakers: List[SpeakerInfo], cache_file: str) -> None:
    info = [speaker.store() for speaker in speakers]
    with open(cache_file, "wb") as f:
        pickle.dump(info, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_speakers(cache_file: str) -> List[SpeakerInfo]:
    with open(cache_file, "rb") as f:
        info = pickle.load(f)
    speakers = [SpeakerInfo.load(data) for data in info]
    return speakers


def create_speakers(dataset_path: str, n_jobs: int) -> List[SpeakerInfo]:
    all_speakers_folders = glob(os.path.join(dataset_path, "*", "audio", "*"))
    speakers = Parallel(n_jobs=n_jobs)(
        delayed(SpeakerInfo)(i, path_to_speaker) for i, path_to_speaker in enumerate(all_speakers_folders)
    )
    return speakers


def cache_text_by_file_name(cache_file: str, text_by_file_name: Dict[str, str]) -> None:
    with open(os.path.join(cache_file), "wb") as f:
        pickle.dump(text_by_file_name, f)


def load_text_by_file_name(cache_file: str) -> Dict[str, str]:
    with open(os.path.join(cache_file), "rb") as f:
        text_by_file_name = pickle.load(f)
    return text_by_file_name


def create_text_by_file_name(dataset_path: str) -> Dict[str, str]:
    text_by_file_name_dict = {
        **{
            audio: text.strip("\n'\" ")
            for audio, text in [
                line.split("\t") for line in open(os.path.join(dataset_path, "dev", "transcripts.txt")).readlines()
            ]
        },
        **{
            audio: text.strip("\n'\" ")
            for audio, text in [
                line.split("\t") for line in open(os.path.join(dataset_path, "test", "transcripts.txt")).readlines()
            ]
        },
        **{
            audio: text.strip("\n'\" ")
            for audio, text in [
                line.split("\t") for line in open(os.path.join(dataset_path, "train", "transcripts.txt")).readlines()
            ]
        },
    }
    return text_by_file_name_dict


def create_metadata(speakers: List[SpeakerInfo], text_by_file_name_dict: Dict[str, str]):
    metadata = []
    for speaker in speakers:
        metadata.extend(speaker.get_metadata())
    metadata = pd.DataFrame(metadata)
    metadata["text"] = metadata["audio_name"].map(text_by_file_name_dict)
    return metadata[["path_to_wav", "text", "speaker_id"]]


@click.command("main", context_settings={"show_default": True})
@click.option(
    "--dataset_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to MLS dataset",
)
@click.option("--output_path", type=click.Path(), required=True, help="Path to output directory")
@click.option(
    "--change_sample_rate",
    default=False,
    help="Resample all audiofiles to specified sample rate.",
)
@click.option(
    "--result_sample_rate",
    default=44100,
    help="Resample all audiofiles to specified sample rate.",
)
@click.option(
    "--n_jobs",
    default=-1,
    help="Number of parallel jobs. If set to -1, use all available CPU cores.",
)
@click.option(
    "--cache_dir",
    default=".cache",
    help="Directory in output path to store cache files.",
)
@click.option(
    "--n_files",
    default=3600,
    help="Number of files to process. If set to -1, process all files of speaker. "
    + "Mean duration of files is 15s, so if you want to process 1h of speech, set this to 3600. "
    + "If there not enough files, all files will be processed.",
)
@click.option(
    "--cache_every_n_speakers", default=100, help="Number of speakers to be processed before cache is updated."
)
def main(
    dataset_path: str,
    output_path: str,
    change_sample_rate: bool = False,
    result_sample_rate: int = 44100,
    n_jobs: int = -1,
    cache_dir=".cache",
    n_files: int = 3600,
    cache_every_n_speakers: int = 100,
):
    if n_jobs == -1:
        n_jobs = cpu_count()

    cache_path = os.path.join(output_path, cache_dir)
    speakers_cache_file = os.path.join(cache_path, "speakers.pkl")
    text_cache_file = os.path.join(cache_path, "text.pkl")
    os.makedirs(cache_path, exist_ok=True)

    if os.path.isfile(speakers_cache_file):
        speakers = load_speakers(speakers_cache_file)
    else:
        speakers = create_speakers(dataset_path, n_jobs)
        cache_speakers(speakers, speakers_cache_file)

    if os.path.isfile(os.path.join(text_cache_file)):
        text_by_file_name_dict = load_text_by_file_name(text_cache_file)
    else:
        text_by_file_name_dict = create_text_by_file_name(dataset_path)
        cache_text_by_file_name(text_cache_file, text_by_file_name_dict)

    tqdm_bar = tqdm(total=len(speakers), desc="Processing speakers: ")
    next_cache_checkpoint = cache_every_n_speakers
    for speakers_batch in [speakers[i : i + n_jobs] for i in range(0, len(speakers), n_jobs)]:
        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(speaker.process)(output_path, n_files, change_sample_rate, result_sample_rate)
            for speaker in speakers_batch
        )

        metadata = create_metadata(speakers, text_by_file_name_dict)
        metadata.to_csv(os.path.join(output_path, "metadata.csv"), index=False, sep="|")

        if tqdm_bar.n > next_cache_checkpoint:
            cache_speakers(speakers, speakers_cache_file)
            next_cache_checkpoint += cache_every_n_speakers

        tqdm_bar.update(n_jobs)


if __name__ == "__main__":
    main()
