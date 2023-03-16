import os
import io
import click

import numpy as np

from typing import Tuple
from typing import Iterator

from PIL import Image
from PIL import ImageOps

from learning_algorithms.unsplash import ImageRepository
from learning_algorithms.utils import resolution_to_shape
from learning_algorithms.utils import shape_to_resolution


class FeatureStore:
    def __init__(self, path: str):
        self._path = path

    @staticmethod
    def open(path: str) -> "FeatureStore":
        if not os.path.exists(path):
            os.mkdir(path)

        return FeatureStore(path)

    @staticmethod
    def _find_square(width: int, height: int) -> Tuple[int, int, int, int]:
        if width > height:
            left = (width - height) // 2
            top = 0
            right = left + height
            bottom = height
        else:
            left = 0
            top = (height - width) // 2
            right = width
            bottom = top + width

        return left, top, right, bottom

    @staticmethod
    def _image_to_features(data: bytes, shape: Tuple[int, int]) -> np.ndarray:
        image = Image.open(io.BytesIO(data))
        size = FeatureStore._find_square(*image.size)

        transformed = image.crop(size).resize(shape)
        grayscaled = ImageOps.grayscale(transformed)

        return np.asarray(grayscaled, dtype=np.uint8).reshape((1, np.product(shape)))

    def write(self, topic: str, images: Iterator[Tuple[str, bytes]], shape: Tuple[int, int]) -> int:
        vector = [
            FeatureStore._image_to_features(data, shape)
            for _, data in images
        ]

        features = np.concatenate(vector, axis=0)
        resolution = shape_to_resolution(shape)

        if not os.path.exists(os.path.join(self._path, resolution)):
            os.mkdir(os.path.join(self._path, resolution))

        np.save(os.path.join(self._path, resolution, topic + ".npy"), features)
        return len(features)

    def load(self, shape: Tuple[int, int]) -> Iterator[np.ndarray]:
        resolution = shape_to_resolution(shape)

        features: Iterator[np.ndarray] = (
            np.load(os.path.join(self._path, resolution, file))
            for file in os.listdir(os.path.join(self._path, resolution))
        )

        return features


@click.command()
@click.argument("resolution", type=click.Choice(["28x28", "64x64", "128x128"]))
def preprocess(resolution: str) -> None:
    repository = ImageRepository.open("./data/unsplash")
    features = FeatureStore.open("./data/features")

    for topic in repository.topics():
        data = repository.stream(topic)
        shape = resolution_to_shape(resolution)

        samples = features.write(topic, data, shape)
        print(f"{topic}: {samples}")
