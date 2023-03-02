from typing import Any
from typing import Dict
from typing import Protocol


class CountEstimator(Protocol):
    def increment(self, key: str, count: int) -> None:
        ...

    def estimate(self, key: str) -> int:
        ...


class BenchmarkTarget(Protocol):
    def get_estimator(self) -> CountEstimator:
        ...

    def describe(self) -> Dict[str, Any]:
        ...


class Benchmark(Protocol):
    def execute(self, target: BenchmarkTarget) -> None:
        ...
