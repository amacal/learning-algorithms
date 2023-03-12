import numpy as np

from typing import Tuple

from keras.datasets import mnist
from learning_algorithms.model import Autoencoder


def reshape_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    images = images.reshape((len(images), np.prod(images.shape[1:])))

    return images


def load_preprocessed_minist_dataset() -> Tuple[np.ndarray, np.ndarray]:
    (train_images, _), (test_images, _) = mnist.load_data()
    return reshape_images(train_images), reshape_images(test_images)


def main() -> None:
    autoencoder = Autoencoder.empty((28, 28), 32)
    dataset = load_preprocessed_minist_dataset()

    autoencoder.train(*dataset, epochs=50, batch_size=256)
    autoencoder.save("./data/mnist.h5")
