import os
from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd
import yaml
from sqlalchemy import and_, create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from src.metrics_collection.models import (
    AudioMetrics,
    AudioToASRText,
    AudioToDataset,
    AudioToOriginalText,
    TextComparationMetrics,
)


class FiltersGenerator:
    def __init__(self, path_to_config: str, dataset_name: str, config_name: str | None = None):
        self.dataset_name = dataset_name
        self.config = FiltersGenerator.read_yaml_config(path_to_config, config_name)

    @staticmethod
    def read_yaml_config(path_to_config: str, config_name: str | None = None) -> Dict[str, Any]:
        with open(path_to_config, "r", encoding="UTF-8") as file:
            configs = yaml.safe_load(file)

        if config_name is not None:
            try:
                return configs[config_name]
            except KeyError:
                print(f"Config name {config_name} was not found in {path_to_config}." + "Setting configs to default.")

        try:
            return configs["default"]
        except KeyError:
            raise ValueError(f"Default configs was not found in {path_to_config}. ")

    def sample_rate_filter(self) -> ColumnElement | None:
        sample_rate_value = self.config.get("sample_rate")
        if sample_rate_value is not None:
            return AudioMetrics.sample_rate == sample_rate_value
        return None

    def channels_filter(self) -> ColumnElement | None:
        channels_value = self.config.get("channels")
        if channels_value is not None:
            return AudioMetrics.channels == channels_value
        return None

    def duration_filter(self) -> ColumnElement | None:
        duration_settings: Dict[str, int] | None = self.config.get("duration")
        if duration_settings is not None:
            min_border = duration_settings.get("min")
            max_border = duration_settings.get("max")

            if min_border is not None and max_border is not None:
                return and_(min_border <= AudioMetrics.duration_seconds, AudioMetrics.duration_seconds <= max_border)
            elif min_border is None and max_border is not None:
                return AudioMetrics.duration_seconds <= max_border
            elif max_border is None and min_border is not None:
                return AudioMetrics.duration_seconds >= min_border
            else:
                return None
        return None

    def SNR_filter(self) -> ColumnElement | None:
        SNR_settings: Dict[str, int] | None = self.config.get("SNR")
        if SNR_settings is not None:
            min_border = SNR_settings.get("min")
            max_border = SNR_settings.get("max")

            if min_border is not None and max_border is not None:
                return and_(min_border <= AudioMetrics.SNR, AudioMetrics.SNR <= max_border)
            elif min_border is None and max_border is not None:
                return AudioMetrics.SNR <= max_border
            elif max_border is None and min_border is not None:
                return AudioMetrics.SNR >= min_border
            else:
                return None
        return None

    def dBFS_filter(self) -> ColumnElement | None:
        dBFS_settings: Dict[str, int] | None = self.config.get("dBFS")
        if dBFS_settings is not None:
            min_border = dBFS_settings.get("min")
            max_border = dBFS_settings.get("max")

            if min_border is not None and max_border is not None:
                return and_(min_border <= AudioMetrics.dBFS, AudioMetrics.dBFS <= max_border)
            elif min_border is None and max_border is not None:
                return AudioMetrics.dBFS <= max_border
            elif max_border is None and min_border is not None:
                return AudioMetrics.dBFS >= min_border
            else:
                return None
        return None

    def CER_filter(self) -> ColumnElement | None:
        CER_settings: Dict[str, int] | None = self.config.get("CER")
        if CER_settings is not None:
            min_border = CER_settings.get("min")
            max_border = CER_settings.get("max")

            if min_border is not None and max_border is not None:
                return and_(min_border <= TextComparationMetrics.CER, TextComparationMetrics.CER <= max_border)
            elif min_border is None and max_border is not None:
                return TextComparationMetrics.CER <= max_border
            elif max_border is None and min_border is not None:
                return TextComparationMetrics.CER >= min_border
            else:
                return None
        return None

    def WER_filter(self) -> ColumnElement | None:
        WER_settings: Dict[str, int] | None = self.config.get("WER")
        if WER_settings is not None:
            min_border = WER_settings.get("min")
            max_border = WER_settings.get("max")

            if min_border is not None and max_border is not None:
                return and_(min_border <= TextComparationMetrics.WER, TextComparationMetrics.WER <= max_border)
            elif min_border is None and max_border is not None:
                return TextComparationMetrics.WER <= max_border
            elif max_border is None and min_border is not None:
                return TextComparationMetrics.WER >= min_border
            else:
                return None
        return None

    def use_unknown_speakers_filter(self) -> ColumnElement | None:
        use_unknown_speakers = self.config.get("use_unknown_speakers")
        if use_unknown_speakers is not None:
            if not use_unknown_speakers:
                return AudioToDataset.speaker_id != -1
        return None

    def only_with_ASR_texts_filter(self) -> ColumnElement | None:
        only_with_ASR_texts: bool | None = self.config.get("only_with_ASR_texts")
        if only_with_ASR_texts is not None:
            if only_with_ASR_texts:
                return AudioToASRText.text.isnot(None)
        return None

    def only_with_Original_texts_filter(self) -> ColumnElement | None:
        only_with_Original_texts: bool | None = self.config.get("only_with_Original_texts")
        if only_with_Original_texts is not None:
            if only_with_Original_texts:
                return AudioToOriginalText.text.isnot(None)
        return None

    def samples_per_speaker_filter(self, session) -> List[str] | None:
        samples_settings: Dict[str, int] | None = self.config.get("samples_per_speaker")
        if samples_settings is not None:
            max_border = samples_settings.get("max")
            min_border = samples_settings.get("min")

            if max_border is not None:
                # Get speaker counts
                speaker_counts = (
                    session.query(
                        AudioToDataset.speaker_id, func.count(AudioToDataset.audio_md5_hash).label("sample_count")
                    )
                    .filter(AudioToDataset.dataset_name == self.dataset_name)
                    .group_by(AudioToDataset.speaker_id)
                    .all()
                )

                # Filter speakers who exceed the max border
                valid_hashes = []
                for speaker_id, count in speaker_counts:
                    speaker_and_dataset_filter = and_(
                        AudioToDataset.dataset_name == self.dataset_name, AudioToDataset.speaker_id == speaker_id
                    )

                    if count > max_border:
                        # Select a random subset of samples for this speaker
                        samples = (
                            session.query(AudioToDataset.audio_md5_hash)
                            .filter(speaker_and_dataset_filter)
                            .order_by(AudioToDataset.speaker_id)
                            .limit(max_border)
                            .all()
                        )
                        valid_hashes.extend([sample.audio_md5_hash for sample in samples])
                    elif min_border is not None and count < min_border:
                        continue
                    else:
                        samples = session.query(AudioToDataset.audio_md5_hash).filter(speaker_and_dataset_filter).all()
                        valid_hashes.extend([sample.audio_md5_hash for sample in samples])

                return valid_hashes
        return None

    def minutes_per_speaker_filter(self, session) -> List[str] | None:
        minutes_settings: Dict[str, int] | None = self.config.get("minutes_per_speaker")
        if minutes_settings is not None:
            max_border = minutes_settings.get("max")
            min_border = minutes_settings.get("min")

            if max_border is not None:
                # Get speaker durations
                speaker_durations = (
                    session.query(
                        AudioToDataset.speaker_id, func.sum(AudioMetrics.duration_seconds).label("total_duration")
                    )
                    .join(AudioMetrics, AudioToDataset.audio_md5_hash == AudioMetrics.audio_md5_hash)
                    .filter(AudioToDataset.dataset_name == self.dataset_name)
                    .group_by(AudioToDataset.speaker_id)
                    .all()
                )

                # Filter speakers who exceed the max border
                valid_hashes = []
                for speaker_id, duration in speaker_durations:
                    speaker_and_dataset_filter = and_(
                        AudioToDataset.dataset_name == self.dataset_name, AudioToDataset.speaker_id == speaker_id
                    )

                    if duration > max_border * 60:
                        # Select samples until the max border is reached
                        samples_and_durations = (
                            session.query(AudioToDataset.audio_md5_hash, AudioMetrics.duration_seconds)
                            .join(AudioMetrics, AudioToDataset.audio_md5_hash == AudioMetrics.audio_md5_hash)
                            .filter(speaker_and_dataset_filter)
                            .order_by(AudioToDataset.speaker_id)
                            .all()
                        )

                        hashes = []
                        overall_duration = 0
                        for hash, sample_duration in samples_and_durations:
                            hashes.append(hash)
                            overall_duration += sample_duration
                            if overall_duration > max_border * 60:
                                break
                        valid_hashes.extend(hashes)
                    elif min_border is not None and duration < min_border:
                        continue
                    else:
                        samples = session.query(AudioToDataset.audio_md5_hash).filter(speaker_and_dataset_filter).all()
                        valid_hashes.extend([sample.audio_md5_hash for sample in samples])

                return valid_hashes
        return None

    def generate_filters(self, session) -> List[ColumnElement]:
        filters = [
            self.sample_rate_filter(),
            self.channels_filter(),
            self.duration_filter(),
            self.SNR_filter(),
            self.dBFS_filter(),
            self.CER_filter(),
            self.WER_filter(),
            self.use_unknown_speakers_filter(),
            self.only_with_ASR_texts_filter(),
            self.only_with_Original_texts_filter(),
        ]
        return [f for f in filters if f is not None]


def filter_dataset(
    path_to_dataset: str,
    path_to_filter_config: str,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    config_name: str | None = None,
) -> pd.DataFrame:
    dataset_name = Path(path_to_dataset).stem

    generator = FiltersGenerator(path_to_filter_config, dataset_name, config_name)

    engine = create_engine(
        f"postgresql://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )
    Session = sessionmaker(bind=engine)
    session = Session()

    # Apply basic filters
    query = (
        session.query(AudioToDataset.path_to_file, AudioToDataset.speaker_id, AudioToDataset.audio_md5_hash)
        .join(AudioMetrics, AudioMetrics.audio_md5_hash == AudioToDataset.audio_md5_hash, isouter=True)
        .join(
            TextComparationMetrics, TextComparationMetrics.audio_md5_hash == AudioToDataset.audio_md5_hash, isouter=True
        )
        .join(AudioToOriginalText, AudioToOriginalText.audio_md5_hash == AudioToDataset.audio_md5_hash, isouter=True)
        .join(AudioToASRText, AudioToASRText.audio_md5_hash == AudioToDataset.audio_md5_hash, isouter=True)
        .filter(AudioToDataset.dataset_name == dataset_name)
    )

    filters = generator.generate_filters(session)
    for f in filters:
        query = query.filter(f)

    # Apply samples_per_speaker filter
    valid_samples = generator.samples_per_speaker_filter(session)
    if valid_samples is not None:
        query = query.filter(AudioToDataset.audio_md5_hash.in_(valid_samples))

    # Apply minutes_per_speaker filter
    valid_minutes = generator.minutes_per_speaker_filter(session)
    if valid_minutes is not None:
        query = query.filter(AudioToDataset.audio_md5_hash.in_(valid_minutes))

    filtered_data = query.all()
    session.close()

    return pd.DataFrame(filtered_data, columns=["path_to_wav", "speaker_id", "hash"])


@click.command()
@click.option("--dataset-path", type=click.Path(exists=True, file_okay=False), help="Path to dataset.")
@click.option("--path-to-config", type=click.Path(exists=True, dir_okay=False), help="Path to YAML filtration config.")
@click.option("--config-name", type=str, help="Name of config in YAML file to use.", default="default")
@click.option("--database-address", type=str, help="Address of the database", envvar="POSTGRES_ADDRESS")
@click.option("--database-port", type=int, help="Port of the database", envvar="POSTGRES_PORT")
@click.option("--database-user", type=str, help="Username to use for database authentication", envvar="POSTGRES_USER")
@click.option(
    "--database-password", type=str, help="Password to use for database authentication", envvar="POSTGRES_PASSWORD"
)
@click.option("--database-name", type=str, help="Name of the database", envvar="POSTGRES_DB")
@click.option(
    "--save-path",
    type=click.Path(dir_okay=False),
    help="Path where to save metadata Data Frame file.",
    callback=lambda context, _, value: value
    if value
    else os.path.join(context.params["dataset_path"], "filtered_metadata.csv"),
)
def cli(
    dataset_path: str,
    path_to_config: str,
    config_name: str,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    save_path: str,
) -> None:
    filtered_df = filter_dataset(
        path_to_dataset=dataset_path,
        path_to_filter_config=path_to_config,
        config_name=config_name,
        database_address=database_address,
        database_port=database_port,
        database_user=database_user,
        database_password=database_password,
        database_name=database_name,
    )
    filtered_df.to_csv(save_path, sep="|", index=False)


if __name__ == "__main__":
    cli()
