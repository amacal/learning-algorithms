import numpy as np
import tensorflow as tf

from typing import Tuple
from typing import Callable
from typing import Iterator

from keras.datasets import mnist
from learning_algorithms.engineering import FeatureStore


def load_preprocessed_minist_dataset(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def reshape_images(images: np.ndarray) -> tf.data.Dataset:
        images = images.astype("float32") / 255.0
        images = images.reshape((len(images), np.prod(images.shape[1:])))

        return tf.data.Dataset.from_tensor_slices((images, images)).batch(batch_size)

    (train_images, _), (test_images, _) = mnist.load_data()
    return reshape_images(train_images), reshape_images(test_images)


def load_preprocessed_unsplash_dataset(store: FeatureStore, shape: Tuple[int, int], batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def generate_dataset(training: bool) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for images in store.load(shape):
            rng = np.random.RandomState(42)
            indices = rng.permutation(images.shape[0])
            split = int(images.shape[0] * 0.8)

            for image in images[indices[:split] if training else indices[:split]]:
                yield image.astype("float32") / 255.0, image.astype("float32") / 255.0

    def generate_train_dataset() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return generate_dataset(True)

    def generate_test_dataset() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return generate_dataset(False)

    def build_dataset(generator: Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]]) -> tf.data.Dataset:
        output_types = (tf.float32, tf.float32)
        output_shapes = (np.product(shape), np.product(shape))

        result = tf.data.Dataset.from_generator(generator, output_types, output_shapes=output_shapes)
        result = result.batch(batch_size).prefetch(2 * batch_size)

        return result

    return build_dataset(generate_train_dataset), build_dataset(generate_test_dataset)
