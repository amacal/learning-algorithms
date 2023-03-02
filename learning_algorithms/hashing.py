import hashlib

from typing import Tuple


class HashFunction:
    def __init__(self, index: int, width: int):
        self._index = index
        self._width = width

    def get_width(self) -> int:
        return self._width

    def compute_hash(self, key: str) -> Tuple[int, bool]:
        sha1 = hashlib.sha1()

        sha1.update(str(self._index).encode())
        sha1.update(key.encode())

        digest = sha1.digest()
        hash_value = int.from_bytes(digest, byteorder="big")

        return hash_value % self._width, (hash_value // self._width) % 2 == 0


class HashFunctionFamily:
    def __init__(self, depth: int, width: int):
        self._depth = depth
        self._data = [HashFunction(i, width) for i in range(depth)]

    def get_depth(self) -> int:
        return self._depth

    def get_width(self, index: int) -> int:
        return self._data[index].get_width()

    def compute_hash(self, index: int, key: str) -> Tuple[int, bool]:
        return self._data[index].compute_hash(key)
