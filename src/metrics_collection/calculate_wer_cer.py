import dataclasses
from typing import Dict, List

import click
from dotenv import load_dotenv
from jiwer import cer, wer
from joblib import Parallel, delayed
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from sqlalchemy.engine.base import Engine
from tqdm import tqdm

from src.metrics_collection.models import AudioToASRText, AudioToOriginalText, Base, TextComparationMetrics
from src.utils import read_metadata_and_calculate_hash, update_bar_on_ending
from src.data_managers import LocalFileSystemManager, LakeFSFileSystemManager, AbstractFileSystemManager

load_dotenv()


@click.group()
@click.option("--overwrite", is_flag=True, help="Is to overwrite existing metrics or not.", default=False)
@click.option("--database-address", type=str, help="Address of the database", envvar="POSTGRES_ADDRESS")
@click.option("--database-port", type=int, help="Port of the database", envvar="POSTGRES_PORT")
@click.option("--database-user", type=str, help="Username to use for database authentication", envvar="POSTGRES_USER")
@click.option(
    "--database-password", type=str, help="Password to use for database authentication", envvar="POSTGRES_PASSWORD"
)
@click.option("--database-name", type=str, help="Name of the database", envvar="POSTGRES_DB")
@click.option(
    "--n-jobs", type=int, default=-1, help="Number of parallel jobs to use while processing. -1 means to use all cores."
)
@click.pass_context
def cli(
    context: click.Context,
    overwrite: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
):
    context.ensure_object(dict)

    context.obj["overwrite"] = overwrite
    context.obj["n_jobs"] = n_jobs

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )
    context.obj["engine"] = engine


@cli.command()
@click.option("--dataset-path", type=click.Path(exists=True), help="Path to dataset")
@click.pass_context
def local(context: click.Context, dataset_path: str):
    file_system_manager = LocalFileSystemManager(dataset_path)

    process_text_comparison_metrics_to_db(
        file_system_manager=file_system_manager,
        engine=context.obj["engine"],
        overwrite=context.obj["overwrite"],
        n_jobs=context.obj["n_jobs"],
    )


@cli.command()
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--repository-name", type=str, help="Name of LakeFS repository")
@click.option("--branch-name", type=str, help="Name of the branch.", default="main")
@click.pass_context
def s3(
    context: click.Context,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    repository_name: str,
    branch_name: str,
):
    file_system_manager = LakeFSFileSystemManager(
        lakefs_address=lakefs_address,
        lakefs_port=lakefs_port,
        lakefs_ACCESS_KEY=access_key_id,
        lakefs_SECRET_KEY=secret_key,
        lakefs_repository_name=repository_name,
        lakefs_branch_name=branch_name,
    )

    process_text_comparison_metrics_to_db(
        file_system_manager=file_system_manager,
        engine=context.obj["engine"],
        overwrite=context.obj["overwrite"],
        n_jobs=context.obj["n_jobs"],
    )


def process_text_comparison_metrics_to_db(
    file_system_manager: AbstractFileSystemManager,
    engine: Engine,
    overwrite: bool,
    n_jobs: int = -1,
) -> None:
    with file_system_manager.get_buffered_reader("metadata.csv") as metadata_reader:
        metadata_df = read_metadata_and_calculate_hash(metadata_reader, file_system_manager, n_jobs=n_jobs)

    dataset_name = file_system_manager.directory_name

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        dataset_hashes = set(metadata_df["hash"])

        existing_in_db_hashes_query = select(TextComparationMetrics.audio_md5_hash).where(
            TextComparationMetrics.audio_md5_hash.in_(dataset_hashes)
        )
        existing_in_db_hashes = set(session.scalars(existing_in_db_hashes_query).all())

        original_text_query = select(AudioToOriginalText).where(AudioToOriginalText.audio_md5_hash.in_(dataset_hashes))
        asr_text_query = select(AudioToASRText).where(AudioToASRText.audio_md5_hash.in_(dataset_hashes))

        original_texts = {row.audio_md5_hash: row.text for row in session.scalars(original_text_query).all()}
        asr_texts = {row.audio_md5_hash: row.text for row in session.scalars(asr_text_query).all()}

        if len(original_texts) == 0:
            print(f"No Original Texts was found for dataset {dataset_name}.")
            return

        if len(asr_texts) == 0:
            print(f"No ASR Texts was found for dataset {dataset_name}.")
            return

        intersected_hashes = set(original_texts.keys()) & set(asr_texts.keys())
        print(
            f"Found {len(intersected_hashes)} ({len(intersected_hashes) / len(dataset_hashes) * 100:.2f}%) samples which have ASR and Original texts for comparation."
        )

        hashes_to_add = intersected_hashes - existing_in_db_hashes
        print(
            f"{len(hashes_to_add)} ({len(hashes_to_add) / len(dataset_hashes) * 100:.2f}%) samples are to be added into database."
        )

        samples_wer_cer_info = calculate_samples_wer_cer(
            hashes=hashes_to_add, original_texts=original_texts, asr_texts=asr_texts, n_jobs=n_jobs
        )
        session.add_all(samples_wer_cer_info)

        if overwrite:
            hashes_to_update = intersected_hashes & existing_in_db_hashes
            print(
                f"{len(hashes_to_update)} ({len(hashes_to_update) / len(dataset_hashes) * 100:.2f}%) samples are to be updated in database."
            )

            samples_wer_cer_info = calculate_samples_wer_cer(
                hashes=hashes_to_update, original_texts=original_texts, asr_texts=asr_texts, n_jobs=n_jobs
            )
            samples_wer_cer_info = [dataclasses.asdict(info) for info in samples_wer_cer_info if info is not None]
            session.execute(update(TextComparationMetrics), samples_wer_cer_info)

        session.commit()


def calculate_samples_wer_cer(
    hashes: List[str], original_texts: Dict[str, str], asr_texts: Dict[str, str], n_jobs: int
) -> List[TextComparationMetrics]:
    tqdm_bar = tqdm(total=len(hashes), desc="Calculating WER/CER metrics")
    samples_wer_cer_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(TextComparationMetrics))(
            audio_md5_hash=hash,
            WER=wer(original_texts[hash], asr_texts[hash]),
            CER=cer(original_texts[hash], asr_texts[hash]),
        )
        for hash in list(hashes)
    )
    samples_wer_cer_info = [info for info in samples_wer_cer_info if info is not None]
    return samples_wer_cer_info


if __name__ == "__main__":
    cli()
