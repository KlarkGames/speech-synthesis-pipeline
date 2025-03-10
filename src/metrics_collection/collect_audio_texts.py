import dataclasses
from typing import List

import click
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from sqlalchemy.engine.base import Engine
from tqdm import tqdm

from src.metrics_collection.models import AudioToOriginalText, Base
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

    process_text_metrics_to_db(
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

    process_text_metrics_to_db(
        file_system_manager=file_system_manager,
        engine=context.obj["engine"],
        overwrite=context.obj["overwrite"],
        n_jobs=context.obj["n_jobs"],
    )


def process_text_metrics_to_db(
    file_system_manager: AbstractFileSystemManager,
    engine: Engine,
    overwrite: bool,
    n_jobs: int = -1,
) -> None:
    with file_system_manager.get_buffered_reader("metadata.csv") as metadata_reader:
        metadata_df = read_metadata_and_calculate_hash(metadata_reader, file_system_manager, n_jobs=n_jobs)

    assert "text" in metadata_df.columns

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
    return samples_text_info


if __name__ == "__main__":
    cli()
