from sqlalchemy import ForeignKey, Identity
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


class Base(MappedAsDataclass, DeclarativeBase):
    """subclasses will be converted to dataclasses"""


class AudioMetrics(Base):
    __tablename__ = "audio_metrics"

    audio_md5_hash: Mapped[str] = mapped_column(primary_key=True)
    duration_seconds: Mapped[float]
    sample_rate: Mapped[int]
    channels: Mapped[int]
    pcm_format: Mapped[str]
    SNR: Mapped[float]
    dBFS: Mapped[float]


class AudioToDataset(Base):
    __tablename__ = "audio_to_dataset"

    id: Mapped[int] = mapped_column(Identity(), primary_key=True, autoincrement=True, init=False)
    audio_md5_hash: Mapped[str] = mapped_column(ForeignKey("audio_metrics.audio_md5_hash"))
    dataset_name: Mapped[str]
    path_to_file: Mapped[str]
    speaker_id: Mapped[int] = mapped_column(default=-1)


class AudioToOriginalText(Base):
    __tablename__ = "audio_to_original_text"

    audio_md5_hash: Mapped[str] = mapped_column(ForeignKey("audio_metrics.audio_md5_hash"), primary_key=True)
    text: Mapped[str]


class AudioToASRText(Base):
    __tablename__ = "audio_to_asr_text"

    audio_md5_hash: Mapped[str] = mapped_column(ForeignKey("audio_metrics.audio_md5_hash"), primary_key=True)
    text: Mapped[str]


class TextComparationMetrics(Base):
    __tablename__ = "text_comparation_metrics"

    audio_md5_hash: Mapped[str] = mapped_column(ForeignKey("audio_metrics.audio_md5_hash"), primary_key=True)
    WER: Mapped[float]
    CER: Mapped[float]


class AudioToMFAData(Base):
    __tablename__ = "audio_to_mfa_data"

    audio_md5_hash: Mapped[str] = mapped_column(ForeignKey("audio_metrics.audio_md5_hash"), primary_key=True)
    mfa_textgrid_data: Mapped[str]
