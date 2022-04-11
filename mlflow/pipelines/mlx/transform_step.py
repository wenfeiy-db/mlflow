import sys
import cloudpickle
import click
import pandas as pd
import importlib
import os
import yaml

@click.command()
@click.option('--input-path', help='Path to input data')
@click.option("--transformer-config", help="Path to the transformer config")
@click.option('--transformer-path', help='Output path of transformer')
@click.option('--transformed-path', help='Output path to transformed data')
def transform_step(input_path, transformer_config, transformer_path, transformed_path):
    """
    Transform data using a transformer method.
    """
    sys.path.append(os.curdir)
    module_name, method_name = yaml.safe_load(open(transformer_config, "r")).get("transformer_method").rsplit('.', 1)
    transformer_fn = getattr(importlib.import_module(module_name), method_name)
    transformer = transformer_fn()

    df = pd.read_parquet(input_path)

    # TODO: load from conf
    X = df.drop(columns=['price'])
    y = df['price']

    features = transformer.fit_transform(X)

    with open(transformer_path, 'wb') as f:
        cloudpickle.dump(transformer, f)

    transformed = pd.DataFrame(data={"features": list(features), "target": y})
    transformed.to_parquet(transformed_path)

if __name__ == "__main__":
    transform_step()
