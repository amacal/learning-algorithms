import numpy as np
import matplotlib.pyplot as plt

from learning_algorithms.model import Autoencoder
from learning_algorithms.train import load_preprocessed_minist_dataset


def validate(autoencoder: Autoencoder, images: np.ndarray, path: str) -> None:
    fig, axes = plt.subplots(2, len(images), figsize=(20, 4))
    predicted = autoencoder.predict(images)

    for i in range(len(images)):
        axes[0, i].imshow(images[i].reshape((28, 28)), cmap="gray")
        axes[0, i].axis('off')

        axes[1, i].imshow(predicted[i].reshape((28, 28)), cmap="gray")
        axes[1, i].axis('off')

    fig.savefig(path)


def main() -> None:
    _, images = load_preprocessed_minist_dataset()
    samples = np.random.choice(images.shape[0], size=10, replace=False)

    autoencoder = Autoencoder.load("./data/mnist.h5")
    validate(autoencoder, images[samples], "./data/mnist.png")
