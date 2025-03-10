import os

import click
from dotenv import load_dotenv
from lakefs_spec import LakeFSFileSystem
from tqdm import tqdm

load_dotenv()


@click.command()
@click.option("--path", type=click.Path(exists=True))
@click.option("--LakeFS-address", type=str, help="LakeFS address", envvar="LAKEFS_ADDRESS")
@click.option("--LakeFS-port", type=str, help="LakeFS port", envvar="LAKEFS_PORT")
@click.option("--ACCESS-KEY-ID", type=str, help="Access key id of LakeFS", envvar="LAKEFS_ACCESS_KEY_ID")
@click.option("--SECRET-KEY", type=str, help="Secret key of LakeFS", envvar="LAKEFS_SECRET_KEY")
@click.option("--repository-name", type=str, help="Name of LakeFS repository")
@click.option("--branch-name", type=str, help="Name of the branch.", default="main")
@click.option("--overwrite", is_flag=True, help="Is to overwrite existing metrics or not.", default=False)
def cli(
    path: str,
    lakefs_address: str,
    lakefs_port: str,
    access_key_id: str,
    secret_key: str,
    repository_name: str,
    branch_name: str,
    overwrite: bool,
):
    file_system = LakeFSFileSystem(host=f"{lakefs_address}:{lakefs_port}", username=access_key_id, password=secret_key)

    for file_or_dir in tqdm(os.listdir(path)):
        file_system.put(
            os.path.join(path, file_or_dir),
            os.path.join(repository_name, branch_name) + "/",
            recursive=True,
            use_blockstore=True,
        )


if __name__ == "__main__":
    cli()
