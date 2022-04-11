import sys

import pandas as pd
from pandas_profiling import ProfileReport
import click


@click.command()
@click.option('--input-path', help='Path to input data')
@click.option("--summary-path", help="Path to the summary file")
@click.option('--train-path', help='Output path of training data')
@click.option('--test-path', help='Output path of test data')
def split_step(input_path, summary_path, train_path, test_path):
    df = pd.read_parquet(input_path)

    profile = ProfileReport(df, title="Summary of Input Dataset", minimal=True)
    profile.to_file(output_file=summary_path)

    # Drop null values.
    # TODO: load from conf
    df = df.dropna(subset=["price"])

    hash_buckets = df.apply(lambda x: abs(hash(tuple(x))) % 100, axis=1)
    is_train = hash_buckets < 80
    train = df[is_train]
    test = df[~is_train]

    train.to_parquet(train_path)
    test.to_parquet(test_path)

if __name__ == "__main__":
    split_step()
