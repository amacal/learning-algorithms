import keras
import keras.layers
import keras.models

import numpy as np

from typing import Tuple


class Autoencoder:
    def __init__(self, model: keras.Model) -> None:
        self._model = model

    @staticmethod
    def load(path: str) -> "Autoencoder":
        return Autoencoder(keras.models.load_model(path))

    @staticmethod
    def empty(image_shape: Tuple[int, int], encoding_dimension: int) -> "Autoencoder":
        input_layer = keras.Input(shape=(image_shape[0] * image_shape[1], ), name="image")
        encoded_layer = keras.layers.Dense(encoding_dimension, activation="relu", name="encoded")(input_layer)
        decoded_layer = keras.layers.Dense(image_shape[0] * image_shape[1], activation="sigmoid", name="decoded")(encoded_layer)

        model = keras.Model(input_layer, decoded_layer, name="autoencoder")
        model.compile(optimizer="adam", loss="binary_crossentropy")

        return Autoencoder(model)

    def summary(self) -> None:
        self._model.summary()

    def train(self, train_images: np.ndarray, test_images: np.ndarray, *, epochs: int = 50, batch_size: int = 256) -> None:
        self._model.fit(
            train_images, train_images,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_images, test_images)
        )

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self._model.predict(images)

    def save(self, path: str) -> None:
        self._model.save(path)


def main() -> None:
    Autoencoder.empty((28, 28), 32).summary()
