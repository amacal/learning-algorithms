from typing import Any
from typing import Dict
from typing import List

from learning_algorithms.hashing import HashFunctionFamily


class CountMinSketch:
    def __init__(self, hash_famly: HashFunctionFamily):
        self._hash_family = hash_famly
        self._table = CountMinSketch._build_table(self._hash_family)

    @staticmethod
    def _build_table(hash_famly: HashFunctionFamily) -> List[List[int]]:
        return [([0] * hash_famly.get_width(i)) for i in range(hash_famly.get_depth())]

    def describe(self) -> Dict[str, Any]:
        return {
            "depth": self._hash_family.get_depth(),
            "width": self._hash_family.get_width(0),
        }

    def increment(self, key: str, count: int) -> None:
        for i in range(self._hash_family.get_depth()):
            hash_value, _ = self._hash_family.compute_hash(i, key)
            self._table[i][hash_value] += count

    def estimate(self, key: str) -> int:
        candidates: List[int] = list()

        for i in range(self._hash_family.get_depth()):
            hash_value, _ = self._hash_family.compute_hash(i, key)
            candidates.append(self._table[i][hash_value])

        return min(candidates)
