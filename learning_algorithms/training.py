import click

from typing import List

from learning_algorithms.model import Autoencoder
from learning_algorithms.registry import ModelRegistry
from learning_algorithms.engineering import FeatureStore
from learning_algorithms.utils import resolution_to_shape

from learning_algorithms.dataset import load_preprocessed_minist_dataset
from learning_algorithms.dataset import load_preprocessed_unsplash_dataset


@click.command()
@click.argument("datasource", type=click.Choice(["mnist", "unsplash"]))
@click.argument("resolution", type=click.Choice(["28x28", "64x64", "128x128"]))
@click.option("-e", "--epochs", type=click.INT, default=50)
@click.option("-b", "--batch-size", type=click.INT, default=256)
@click.option("-d", "--dimensionality", default=["32"], multiple=True, type=click.Choice(["32", "64", "128", "256", "512", "1024"]))
def train(datasource: str, resolution: str, epochs: int, batch_size: int, dimensionality: List[str]) -> None:
    for dim in dimensionality:
        shape = resolution_to_shape(resolution)
        autoencoder = Autoencoder.empty(shape, int(dim))

        store = FeatureStore("./data/features")
        registry = ModelRegistry("./data/registry")

        if datasource == "mnist":
            dataset = load_preprocessed_minist_dataset(batch_size)
        elif datasource == "unsplash":
            dataset = load_preprocessed_unsplash_dataset(store, shape, batch_size)

        autoencoder.train(*dataset, epochs=epochs, batch_size=batch_size)
        autoencoder.save(registry.resolve("h5", {
            "datasource": datasource,
            "resolution": resolution,
            "epochs": epochs,
            "batch-size": batch_size,
            "dimensionality": int(dim)
        }))
