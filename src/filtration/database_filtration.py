"""
Database Filtration Pipeline

This module provides functionality to filter datasets based on various audio and text metrics stored in a PostgreSQL database.
It supports both local filesystem and LakeFS/S3 storage backends.

Key Features:
- Filters audio samples based on metrics like sample rate, channels, duration, SNR, dBFS
- Filters text samples based on CER and WER metrics
- Supports speaker-based filtering (samples per speaker, minutes per speaker)
- Handles both local and cloud storage via LakeFS/S3
- Generates filtered metadata CSV files

Example usage:
    python script.py local --dataset-path ./audio_data \
        --path-to-config ./filters.yaml \
        --database-address localhost --database-port 5432
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd
import yaml
from sqlalchemy import and_, create_engine, func
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql.elements import ColumnElement
from dotenv import load_dotenv

from src.metrics_collection.models import (
    AudioMetrics,
    AudioToASRText,
    AudioToDataset,
    AudioToOriginalText,
    TextComparationMetrics,
)
from src.data_managers import LocalFileSystemManager, LakeFSFileSystemManager, AbstractFileSystemManager

load_dotenv()


class FiltersGenerator:
    """A class to generate database filters based on configuration settings.

    This class handles the generation of SQLAlchemy filter conditions based on a YAML configuration file.
    It supports filtering on various audio metrics (sample rate, channels, duration, SNR, dBFS) and
    text metrics (CER, WER), as well as speaker-based filtering.

    Args:
        path_to_config: Path to the YAML configuration file containing filter settings
        dataset_name: Name of the dataset to filter
        config_name: Optional name of a specific configuration to use from the YAML file
    """

    def __init__(self, path_to_config: str, dataset_name: str, config_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self.config = self.read_yaml_config(path_to_config, config_name)

    @staticmethod
    def read_yaml_config(path_to_config: str, config_name: Optional[str] = None) -> Dict[str, Any]:
        """Read and parse a YAML configuration file.

        Args:
            path_to_config: Path to the YAML configuration file
            config_name: Optional name of a specific configuration to use

        Returns:
            Dictionary containing the configuration settings

        Raises:
            ValueError: If the default configuration is not found
        """
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

    def _create_range_filter(
        self, config_key: str, column: ColumnElement, min_key: str = "min", max_key: str = "max"
    ) -> Optional[ColumnElement]:
        """Create a range filter for a given column.

        Args:
            config_key: Key in config dictionary for the filter settings
            column: SQLAlchemy column to filter on
            min_key: Key for minimum value in config
            max_key: Key for maximum value in config

        Returns:
            SQLAlchemy filter condition if configured, None otherwise
        """
        settings: Optional[Dict[str, int]] = self.config.get(config_key)
        if settings is not None:
            min_border = settings.get(min_key)
            max_border = settings.get(max_key)

            if min_border is not None and max_border is not None:
                return and_(min_border <= column, column <= max_border)
            elif min_border is None and max_border is not None:
                return column <= max_border
            elif max_border is None and min_border is not None:
                return column >= min_border
        return None

    def sample_rate_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for audio sample rate.

        Returns:
            SQLAlchemy filter condition for sample rate if configured, None otherwise
        """
        sample_rate_value = self.config.get("sample_rate")
        if sample_rate_value is not None:
            return AudioMetrics.sample_rate == sample_rate_value
        return None

    def channels_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for audio channels.

        Returns:
            SQLAlchemy filter condition for channels if configured, None otherwise
        """
        channels_value = self.config.get("channels")
        if channels_value is not None:
            return AudioMetrics.channels == channels_value
        return None

    def duration_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for audio duration.

        Returns:
            SQLAlchemy filter condition for duration if configured, None otherwise
        """
        return self._create_range_filter("duration", AudioMetrics.duration_seconds)

    def SNR_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for Signal-to-Noise Ratio (SNR).

        Returns:
            SQLAlchemy filter condition for SNR if configured, None otherwise
        """
        return self._create_range_filter("SNR", AudioMetrics.SNR)

    def dBFS_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for audio dBFS (decibels relative to full scale).

        Returns:
            SQLAlchemy filter condition for dBFS if configured, None otherwise
        """
        return self._create_range_filter("dBFS", AudioMetrics.dBFS)

    def CER_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for Character Error Rate (CER).

        Returns:
            SQLAlchemy filter condition for CER if configured, None otherwise
        """
        return self._create_range_filter("CER", TextComparationMetrics.CER)

    def WER_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for Word Error Rate (WER).

        Returns:
            SQLAlchemy filter condition for WER if configured, None otherwise
        """
        return self._create_range_filter("WER", TextComparationMetrics.WER)

    def use_unknown_speakers_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for unknown speakers.

        Returns:
            SQLAlchemy filter condition for unknown speakers if configured, None otherwise
        """
        use_unknown_speakers = self.config.get("use_unknown_speakers")
        if use_unknown_speakers is not None and not use_unknown_speakers:
            return AudioToDataset.speaker_id != -1
        return None

    def only_with_ASR_texts_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for samples with ASR texts.

        Returns:
            SQLAlchemy filter condition for ASR texts if configured, None otherwise
        """
        only_with_ASR_texts: Optional[bool] = self.config.get("only_with_ASR_texts")
        if only_with_ASR_texts is not None and only_with_ASR_texts:
            return AudioToASRText.text.isnot(None)
        return None

    def only_with_Original_texts_filter(self) -> Optional[ColumnElement]:
        """Generate a filter for samples with original texts.

        Returns:
            SQLAlchemy filter condition for original texts if configured, None otherwise
        """
        only_with_Original_texts: Optional[bool] = self.config.get("only_with_Original_texts")
        if only_with_Original_texts is not None and only_with_Original_texts:
            return AudioToOriginalText.text.isnot(None)
        return None

    def _get_speaker_filter_query(self, session: Session) -> List[tuple]:
        """Get speaker-based query results.

        Args:
            session: SQLAlchemy database session

        Returns:
            List of tuples containing speaker_id and count/duration
        """
        return (
            session.query(
                AudioToDataset.speaker_id,
                func.count(AudioToDataset.audio_md5_hash).label("sample_count"),
            )
            .filter(AudioToDataset.dataset_name == self.dataset_name)
            .group_by(AudioToDataset.speaker_id)
            .all()
        )

    def samples_per_speaker_filter(self, session: Session) -> Optional[List[str]]:
        """Generate a filter for samples per speaker.

        Args:
            session: SQLAlchemy database session

        Returns:
            List of valid audio hashes if configured, None otherwise
        """
        samples_settings: Optional[Dict[str, int]] = self.config.get("samples_per_speaker")
        if samples_settings is not None:
            max_border = samples_settings.get("max")
            min_border = samples_settings.get("min")

            if max_border is not None:
                speaker_counts = self._get_speaker_filter_query(session)
                valid_hashes = []

                for speaker_id, count in speaker_counts:
                    speaker_and_dataset_filter = and_(
                        AudioToDataset.dataset_name == self.dataset_name,
                        AudioToDataset.speaker_id == speaker_id,
                    )

                    if count > max_border:
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

    def minutes_per_speaker_filter(self, session: Session) -> Optional[List[str]]:
        """Generate a filter for minutes per speaker.

        Args:
            session: SQLAlchemy database session

        Returns:
            List of valid audio hashes if configured, None otherwise
        """
        minutes_settings: Optional[Dict[str, int]] = self.config.get("minutes_per_speaker")
        if minutes_settings is not None:
            max_border = minutes_settings.get("max")
            min_border = minutes_settings.get("min")

            if max_border is not None:
                speaker_durations = (
                    session.query(
                        AudioToDataset.speaker_id,
                        func.sum(AudioMetrics.duration_seconds).label("total_duration"),
                    )
                    .join(AudioMetrics, AudioMetrics.audio_md5_hash == AudioToDataset.audio_md5_hash)
                    .filter(AudioToDataset.dataset_name == self.dataset_name)
                    .group_by(AudioToDataset.speaker_id)
                    .all()
                )

                valid_hashes = []
                for speaker_id, duration in speaker_durations:
                    speaker_and_dataset_filter = and_(
                        AudioToDataset.dataset_name == self.dataset_name,
                        AudioToDataset.speaker_id == speaker_id,
                    )

                    if duration > max_border * 60:
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

    def generate_filters(self, session: Session) -> List[ColumnElement]:
        """Generate all configured filters.

        Args:
            session: SQLAlchemy database session

        Returns:
            List of SQLAlchemy filter conditions
        """
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
    dataset_name: str,
    path_to_filter_config: str,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    config_name: Optional[str] = None,
) -> pd.DataFrame:
    """Filter a dataset based on configured criteria and return filtered metadata.

    This function applies various filters to a dataset based on configuration settings,
    querying a PostgreSQL database for metrics and applying the filters. It supports
    both local filesystem and LakeFS/S3 storage backends.

    Args:
        dataset_name: Name of the dataset
        path_to_filter_config: Path to YAML configuration file containing filter settings
        database_address: PostgreSQL database address
        database_port: PostgreSQL database port
        database_user: Database username
        database_password: Database password
        database_name: Name of the database
        config_name: Optional name of a specific configuration to use

    Returns:
        DataFrame containing filtered metadata with columns:
        - path_to_wav: Path to the audio file
        - speaker_id: ID of the speaker
        - hash: MD5 hash of the audio file
        - original_text: Original transcription text
        - asr_text: ASR-generated text
    """
    generator = FiltersGenerator(path_to_filter_config, dataset_name, config_name)

    engine = create_engine(
        f"postgresql://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Apply basic filters
        query = (
            session.query(
                AudioToDataset.path_to_file,
                AudioToDataset.speaker_id,
                AudioToDataset.audio_md5_hash,
                AudioToOriginalText.text,
                AudioToASRText.text,
            )
            .join(AudioMetrics, AudioMetrics.audio_md5_hash == AudioToDataset.audio_md5_hash, isouter=True)
            .join(
                TextComparationMetrics,
                TextComparationMetrics.audio_md5_hash == AudioToDataset.audio_md5_hash,
                isouter=True,
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
        return pd.DataFrame(filtered_data, columns=["path_to_wav", "speaker_id", "hash", "original_text", "asr_text"])

    finally:
        session.close()


def process_filtered_data(df: pd.DataFrame, include_text: bool = False) -> pd.DataFrame:
    """Process filtered data to add text column if requested.

    Args:
        df: Input DataFrame with filtered data
        include_text: Whether to include text column

    Returns:
        Processed DataFrame
    """
    if include_text:
        df["text"] = df["original_text"]
        df["text"] = df["text"].fillna(df["asr_text"])
        df = df.dropna(subset=["text"])
    df = df.drop(columns=["original_text", "asr_text"])
    return df


# CLI Commands
@click.group()
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
    else os.path.join("filtered_metadata.csv"),
)
@click.option("--include-text", type=bool, is_flag=True, help='Is it needed to create "text" column in metadata file.', default=False)
@click.pass_context
def cli(
    context: click.Context,
    path_to_config: str,
    config_name: str,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    save_path: str,
    include_text: bool,
) -> None:
    """Database Filtration CLI.

    This CLI provides commands to filter datasets based on various metrics stored in a PostgreSQL database.
    It supports both local filesystem and LakeFS/S3 storage backends.

    Commands:
        local: Filter dataset from local filesystem
        s3: Filter dataset from LakeFS/S3 storage
    """
    context.ensure_object(dict)
    context.obj["path_to_config"] = path_to_config
    context.obj["config_name"] = config_name
    context.obj["database_address"] = database_address
    context.obj["database_port"] = database_port
    context.obj["database_user"] = database_user
    context.obj["database_password"] = database_password
    context.obj["database_name"] = database_name
    context.obj["save_path"] = save_path
    context.obj["include_text"] = include_text


@cli.command()
@click.option("--dataset-path", type=click.Path(exists=True, file_okay=False), help="Path to dataset.")
@click.pass_context
def local(
    context: click.Context,
    dataset_path: str,
) -> None:
    """Filter dataset from local filesystem.

    Args:
        dataset_path: Local directory containing the dataset
    """
    file_system_manager = LocalFileSystemManager(dataset_path)
    dataset_name = file_system_manager.directory_name

    filtered_df = filter_dataset(
        dataset_name=dataset_name,
        path_to_filter_config=context.obj["path_to_config"],
        config_name=context.obj["config_name"],
        database_address=context.obj["database_address"],
        database_port=context.obj["database_port"],
        database_user=context.obj["database_user"],
        database_password=context.obj["database_password"],
        database_name=context.obj["database_name"],
    )

    filtered_df = process_filtered_data(filtered_df, context.obj["include_text"])
    filtered_df.to_csv(context.obj["save_path"], sep="|", index=False)


@cli.command()
@click.option("--dataset-path", type=str, help="Path to dataset in LakeFS/S3.")
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--repository-name", type=str, help="Name of LakeFS repository")
@click.option("--branch-name", type=str, help="Name of the branch.", default="main")
@click.pass_context
def s3(
    context: click.Context,
    dataset_path: str,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    repository_name: str,
    branch_name: str,
) -> None:
    """Filter dataset from LakeFS/S3 storage.

    Args:
        dataset_path: Path to dataset in LakeFS/S3
        lakefs_address: LakeFS server address
        lakefs_port: LakeFS server port
        access_key_id: LakeFS access key ID
        secret_key: LakeFS secret key
        repository_name: Name of LakeFS repository
        branch_name: Name of LakeFS branch
    """
    file_system_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=repository_name,
        lakefs_branch_name=branch_name,
    )
    dataset_name = file_system_manager.directory_name

    filtered_df = filter_dataset(
        dataset_name=dataset_name,
        path_to_filter_config=context.obj["path_to_config"],
        config_name=context.obj["config_name"],
        database_address=context.obj["database_address"],
        database_port=context.obj["database_port"],
        database_user=context.obj["database_user"],
        database_password=context.obj["database_password"],
        database_name=context.obj["database_name"],
    )

    filtered_df = process_filtered_data(filtered_df, context.obj["include_text"])
    filtered_df.to_csv(context.obj["save_path"], sep="|", index=False)


if __name__ == "__main__":
    cli()
