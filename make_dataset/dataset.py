from pathlib import Path

import click
import dask.dataframe as dd
import pandas as pd
from distributed import Client
from sklearn.model_selection import train_test_split


def _save_datasets(train, test, outdir: Path, use_dask: bool = False):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    flag = outdir / '.SUCCESS'

    if use_dask:
        out_train = outdir / 'train.parquet/'
        out_test = outdir / 'test.parquet/'

        train.to_parquet(str(out_train))
        test.to_parquet(str(out_test))
    else:
        out_train = outdir / 'train.csv'
        out_test = outdir / 'test.csv'

        train.to_csv(str(out_train), index=False)
        test.to_csv(str(out_test), index=False)

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--test-perc', type=int)
@click.option('--out-dir')
def make_datasets(in_csv, test_perc, out_dir):
    use_dask = False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_dask:
        # Connect to the dask cluster
        _ = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    if use_dask:
        df = dd.read_csv(in_csv, blocksize=1e6)
    else:
        df = pd.read_csv(in_csv)

    # we set the index so we can properly execute loc below
    df = df.set_index('Unnamed: 0')

    # Get y so that train and test can be split equally
    y = df[['points']]

    test_size = test_perc / 100

    # split train and test stratified,
    # with fixed random seed for reproducibility.
    train, test, _, _ = train_test_split(df, y, test_size=test_size,
                                         random_state=42,
                                         stratify=y)

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
