import click

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from typing import List
from typing import Tuple

from learning_algorithms.model import Autoencoder
from learning_algorithms.registry import ModelRegistry
from learning_algorithms.engineering import FeatureStore
from learning_algorithms.utils import resolution_to_shape

from learning_algorithms.dataset import load_preprocessed_minist_dataset
from learning_algorithms.dataset import load_preprocessed_unsplash_dataset


def strip_placeholders(data: Dict[str, str]) -> Dict[str, str]:
    return {key: value for key, value in data.items() if value != "all"}


def pick_placeholders(data: Dict[str, str]) -> List[str]:
    return [key for key, value in data.items() if value == "all"]


def sort_by_placeholders(models: List[Tuple[str, Dict[str, str]]], placeholders: List[str]) -> List[Tuple[str, Dict[str, str]]]:
    return list(sorted(models, key=lambda model: tuple(int(model[1][placeholder]) for placeholder in placeholders)))


def vizualize(shape: Tuple[int, int], models: List[Tuple[str, Dict[str, str]]], images: np.ndarray, path: str) -> None:
    fig, axes = plt.subplots(1 + len(models), len(images), figsize=(20, 2 + 2 * len(models)))

    for i in range(len(images)):
        axes[0, i].imshow(images[i].reshape(shape), cmap="gray")
        axes[0, i].axis('off')
        axes[0, len(images) // 2].set_title("original", fontsize=12)

    for (model, attributes), offset in zip(models, range(len(models))):
        autoencoder = Autoencoder.load(model)
        predicted = autoencoder.predict(images)

        for i in range(len(images)):
            axes[1 + offset, i].imshow(predicted[i].reshape(shape), cmap="gray")
            axes[1 + offset, i].axis('off')
            axes[1 + offset, len(images) // 2].set_title(str(attributes), fontsize=12)

    fig.savefig(path)


def pick_samples(datasource: str, shape: Tuple[int, int], size: int) -> np.ndarray:
    if datasource == "mnist":
        _, images = load_preprocessed_minist_dataset(size)
    elif datasource == "unsplash":
        _, images = load_preprocessed_unsplash_dataset(FeatureStore("./data/features"), shape, size)

    return np.array(list(images.map(lambda x, _: x).shuffle(buffer_size=1024).take(1).as_numpy_iterator())).reshape((size, np.product(shape)))


@click.command()
@click.argument("datasource", type=click.Choice(["mnist", "unsplash"]))
@click.argument("resolution", type=click.Choice(["28x28", "64x64", "128x128"]))
@click.option("-e", "--epochs", default="all", type=click.Choice(["all", "50"]))
@click.option("-b", "--batch-size", default="all", type=click.Choice(["all", "256"]))
@click.option("-d", "--dimensionality", default="all", type=click.Choice(["all", "32", "64", "128", "256", "512", "1024"]))
def validate(datasource: str, resolution: str, epochs: str, batch_size: str, dimensionality: str) -> None:
    shape = resolution_to_shape(resolution)
    registry = ModelRegistry("./data/registry")

    samples = pick_samples(datasource, shape, 11)
    attributes = {
        "datasource": datasource,
        "resolution": resolution,
        "epochs": epochs,
        "batch-size": batch_size,
        "dimensionality": dimensionality
    }

    placeholders = pick_placeholders(attributes)
    attributes = strip_placeholders(attributes)

    models = list(registry.query("h5", attributes))
    models = sort_by_placeholders(models, placeholders)

    vizualize(shape, models, samples, registry.resolve("png", attributes))
