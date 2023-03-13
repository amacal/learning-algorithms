import os
import os.path

from typing import Dict
from typing import Tuple
from typing import Iterator


class ModelRegistry:
    def __init__(self, path: str):
        self._path = path

    @staticmethod
    def open(path: str) -> "ModelRegistry":
        if not os.path.exists(path):
            os.mkdir(path)

        return ModelRegistry(path)

    def resolve(self, extension: str, attributes: Dict[str, str]) -> str:
        path = "&".join(sorted([f"{key}={value}" for key, value in attributes.items()]))
        return os.path.join(self._path, f"{path}.{extension}")

    def query(self, extension: str, attributes: Dict[str, str | int]) -> Iterator[Tuple[str, Dict[str, str]]]:
        def subset(bigger: Dict[str, str], smaller: Dict[str, str]) -> bool:
            return all([item in bigger.items() and str(bigger[item[0]]) == str(item[1]) for item in smaller.items()])

        for file in os.listdir(self._path):
            if os.path.splitext(file)[1] == f".{extension}":
                found = {
                    item.split("=")[0]: item.split("=")[1]
                    for item in os.path.splitext(file)[0].split("&")
                }

                if subset(found, attributes):
                    yield os.path.join(self._path, file), found
