import pandas as pd


def read_sample(path, n=1000):
    """
    Read the first n rows from the CSV.
    Use low_memory=False to avoid mixed-type warnings.
    """
    return pd.read_csv(path, nrows=n, low_memory=False)


def stream_csv(path, chunksize=200_000):
    """
    Generator to read the CSV in chunks.
    Useful for huge datasets.
    """
    return pd.read_csv(path, chunksize=chunksize, low_memory=False)
