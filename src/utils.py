import functools
import hashlib
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import io
from src.data_managers import AbstractFileSystemManager


def update_bar_on_ending(status_bar: tqdm, n: int = 1) -> Callable:
    def run_function(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            status_bar.update(n)
            return result

        return wrapper

    return run_function


def signal_to_noise(a: Iterable, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.

    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.

    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.

    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.

    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def get_hash_of_file(reader: io.BufferedReader) -> str:
    return hashlib.md5(reader.read()).hexdigest()


def read_metadata_and_calculate_hash(
    path_to_metadata: str | io.BufferedReader, file_manager: AbstractFileSystemManager, n_jobs=-1
) -> pd.DataFrame:
    metadata_df = pd.read_csv(path_to_metadata, sep="|").drop_duplicates()

    assert "path_to_wav" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    if "hash" not in metadata_df.columns:

        def process_file(sample: NamedTuple, tqdm_bar: tqdm):
            with file_manager.get_buffered_reader(sample.path_to_wav) as reader:
                return update_bar_on_ending(tqdm_bar)(get_hash_of_file)(reader)

        tqdm_bar = tqdm(total=len(metadata_df), desc="Getting hashes from audio files")
        metadata_df["hash"] = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(process_file)(sample, tqdm_bar) for sample in metadata_df.itertuples()
        )

        with file_manager.get_buffered_writer("metadata.csv") as writer:
            metadata_df.to_csv(writer, sep="|", index=False)

    metadata_df = metadata_df.drop_duplicates(subset=["hash"])

    return metadata_df
