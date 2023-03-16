import keras
import keras.layers
import keras.models
import keras.callbacks

import numpy as np
import tensorflow as tf

from typing import Tuple


class Autoencoder:
    def __init__(self, model: keras.Model):
        self._model = model

    @staticmethod
    def load(path: str) -> "Autoencoder":
        return Autoencoder(keras.models.load_model(path))

    @staticmethod
    def empty(image_shape: Tuple[int, int], dimensionality: int) -> "Autoencoder":
        inputs: tf.Tensor = keras.Input(shape=(np.product(image_shape), ), name="image")
        embeddings: tf.Tensor = keras.layers.Dense(dimensionality, activation="relu", name="encoded")(inputs)
        outputs: tf.Tensor = keras.layers.Dense(np.product(image_shape), activation="sigmoid", name="decoded")(embeddings)

        model = keras.Model(inputs, outputs, name="autoencoder")
        model.compile(optimizer="adam", loss="binary_crossentropy")

        return Autoencoder(model)

    def summary(self) -> None:
        self._model.summary()

    def train(self, train: tf.data.Dataset, test: tf.data.Dataset, *, epochs: int = 50, batch_size: int = 256) -> None:
        self._model.fit(
            train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=test
        )

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self._model.predict(images)

    def save(self, path: str) -> None:
        self._model.save(path)


def main() -> None:
    Autoencoder.empty((28, 28), 32).summary()
