import sys

import pandas as pd
from pandas_profiling import ProfileReport
import click


def run_split_step(input_path, summary_path, train_output_path, test_output_path):
    """
    :param input_path: Path to input data
    :param summary_path: Path to summary file
    :param train_output_path: Output path of training data
    :param test_output_path: Output path of test data
    """
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

    train.to_parquet(train_output_path)
    test.to_parquet(test_output_path)
