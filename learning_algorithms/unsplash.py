import os
import random
import requests

from typing import Set
from typing import List

from typing import Tuple
from typing import Iterator


class ImageRepository:
    def __init__(self, path: str, ids: Set[str]):
        self._path = path
        self._ids = ids

    @staticmethod
    def open(path: str) -> "ImageRepository":
        if not os.path.exists(path):
            os.mkdir(path)

        ids = {
            os.path.splitext(file)[0]
            for _, _, files in os.walk(path)
            for file in files
        }

        return ImageRepository(path, ids)

    def length(self) -> int:
        return len(self._ids)

    def contains(self, id: str) -> bool:
        return id in self._ids

    def topics(self) -> List[str]:
        return [
            os.path.basename(name)
            for name in os.listdir(self._path)
        ]

    def stream(self, topic: str) -> Iterator[Tuple[str, bytes]]:
        for name in os.listdir(os.path.join(self._path, topic)):
            with open(os.path.join(self._path, topic, name), "rb") as file:
                yield os.path.splitext(name)[0], file.read()

    def append(self, topic: str, id: str, data: bytes) -> None:
        if not os.path.exists(os.path.join(self._path, topic)):
            os.mkdir(os.path.join(self._path, topic))

        with open(os.path.join(self._path, topic, id + ".jpg"), "wb") as file:
            file.write(data)

        self._ids.add(id)


class UnsplashService:
    def __init__(self):
        pass

    def topics(self) -> List[str]:
        return ["wallpapers", "3d-renders", "travel", "nature", "street-photography", "experimental", "textures-patterns", "animals", "architecture-interior", "fashion-beauty", "film", "food-drink", "people", "spirituality", "business-work", "athletics", "health", "current-events", "arts-culture"]

    def stream(self, topics: List[str], pages: Tuple[int, int], per_page: int = 20) -> Iterator[Tuple[str, str, str]]:
        visited: Set[str] = set()

        while True:
            topic = random.choice(topics)
            page = random.randint(*pages)

            if f"{topic}/{page}" not in visited:
                response = requests.get(
                    f"https://unsplash.com/napi/topics/{topic}/photos?page={page}&per_page={per_page}")

                response.raise_for_status()
                for item in response.json():
                    yield (topic, item["id"], item["urls"]["thumb"])

    def download(self, url: str) -> bytes:
        response = requests.get(url)
        response.raise_for_status()

        return response.content


def download() -> None:
    repository = ImageRepository.open("./data/unsplash")
    unsplash = UnsplashService()

    for image in unsplash.stream(unsplash.topics(), (1, 250)):
        if not repository.contains(image[1]):
            repository.append(image[0], image[1], unsplash.download(image[2]))
            print(f"{repository.length()} {image[0]}/{image[1]}")

        if repository.length() >= 70000:
            break
