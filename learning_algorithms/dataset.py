import numpy as np

from typing import Tuple
from keras.datasets import mnist

from learning_algorithms.engineering import FeatureStore


def load_preprocessed_minist_dataset() -> Tuple[np.ndarray, np.ndarray]:
    def reshape_images(images: np.ndarray) -> np.ndarray:
        images = images.astype("float32") / 255.0
        images = images.reshape((len(images), np.prod(images.shape[1:])))

        return images

    (train_images, _), (test_images, _) = mnist.load_data()
    return reshape_images(train_images), reshape_images(test_images)


def load_preprocessed_unsplash_dataset(store: FeatureStore, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    images = store.load(image_shape).astype("float32") / 255.0
    rng = np.random.RandomState(42)

    indices = rng.permutation(images.shape[0])
    split = int(images.shape[0] * 0.8)

    return images[indices[:split]], images[indices[split:]]
