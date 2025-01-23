import dataclasses
from typing import List

import click
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.metrics_collection.models import AudioToOriginalText, Base
from src.utils import read_metadata_and_calculate_hash, update_bar_on_ending

load_dotenv()


@click.command()
@click.option("--dataset-path", type=click.Path(exists=True), help="Path to dataset")
@click.option("--overwrite", type=bool, help="Is to overwrite existing metrics or not.", default=False)
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
def main(
    dataset_path: str,
    overwrite: bool,
    database_address: str,
    database_port: int,
    database_user: str,
    database_password: str,
    database_name: str,
    n_jobs: int,
):
    metadata_df = read_metadata_and_calculate_hash(dataset_path, n_jobs=n_jobs)

    assert "text" in metadata_df.columns

    engine = create_engine(
        f"postgresql+psycopg://{database_user}:{database_password}@{database_address}:{database_port}/{database_name}"
    )

    Base.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        existing_in_db_hashes_of_audio = session.scalars(select(AudioToOriginalText.audio_md5_hash)).all()

        samples_to_add = metadata_df[~metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
        samples_text_info = get_original_texts_from_selected_samples(dataframe=samples_to_add, n_jobs=n_jobs)
        session.add_all(samples_text_info)

        if overwrite:
            print("Overwriting others samples")
            samples_to_update = metadata_df[metadata_df["hash"].isin(existing_in_db_hashes_of_audio)]
            samples_text_info = get_original_texts_from_selected_samples(dataframe=samples_to_update, n_jobs=n_jobs)
            samples_text_info = [dataclasses.asdict(info) for info in samples_text_info if info is not None]
            session.execute(update(AudioToOriginalText), samples_text_info)

        session.commit()


def get_original_texts_from_selected_samples(dataframe: pd.DataFrame, n_jobs: int) -> List[AudioToOriginalText]:
    tqdm_bar = tqdm(total=len(dataframe), desc="Collecting dataset info")
    samples_text_info = Parallel(n_jobs=n_jobs, require="sharedmem")(
        delayed(update_bar_on_ending(tqdm_bar)(AudioToOriginalText))(audio_md5_hash=sample.hash, text=sample.text)
        for sample in dataframe.itertuples()
    )
    samples_text_info = [info for info in samples_text_info if info is not None]


if __name__ == "__main__":
    main()
