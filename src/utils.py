import functools
import hashlib
import os
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


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


def get_hash_of_file(path_to_file: str) -> str:
    return hashlib.md5(open(path_to_file, "rb").read()).hexdigest()


def read_metadata_and_calculate_hash(path_to_metadata: str, path_to_dataset: str, n_jobs=-1) -> pd.DataFrame:
    metadata_df = pd.read_csv(path_to_metadata, sep="|").drop_duplicates()

    assert "path_to_wav" in metadata_df.columns
    assert "speaker_id" in metadata_df.columns

    if "hash" not in metadata_df.columns:
        tqdm_bar = tqdm(total=len(metadata_df), desc="Getting hashes from audio files")
        metadata_df["hash"] = Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(update_bar_on_ending(tqdm_bar)(get_hash_of_file))(os.path.join(path_to_dataset, sample.path_to_wav))
            for sample in metadata_df.itertuples()
        )
        metadata_df.to_csv(path_to_metadata, sep="|", index=False)

    return metadata_df
