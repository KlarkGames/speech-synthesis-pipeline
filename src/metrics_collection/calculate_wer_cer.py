import dataclasses
from pathlib import Path
from typing import Dict, List
import os

import click
from dotenv import load_dotenv
from jiwer import cer, wer
from joblib import Parallel, delayed
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.metrics_collection.models import AudioToASRText, AudioToOriginalText, Base, TextComparationMetrics
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
):
    dataset_name = Path(dataset_path).stem
    metadata_df = read_metadata_and_calculate_hash(metadata_path, dataset_path, n_jobs=n_jobs)

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )

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
    main()
