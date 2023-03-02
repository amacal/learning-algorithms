import click
import asyncio

from typing import Any
from typing import List
from typing import Dict

from concurrent.futures import ProcessPoolExecutor

from learning_algorithms.types import CountEstimator
from learning_algorithms.benchmarks import WebNetworkBenchmark
from learning_algorithms.count_min_sketch import CountMinSketch
from learning_algorithms.count_sketch import CountSketch
from learning_algorithms.hash_map import HashMap
from learning_algorithms.hashing import HashFunctionFamily
from learning_algorithms.types import Benchmark


class CountMinSketchTarget:
    def __init__(self, instance: CountMinSketch):
        self._instance = instance

    def get_estimator(self) -> CountEstimator:
        return self._instance

    def describe(self) -> Dict[str, Any]:
        return {
            "algorithm": "count-min-sketch",
            "parameters": self._instance.describe(),
        }


class CountSketchTarget:
    def __init__(self, instance: CountSketch):
        self._instance = instance

    def get_estimator(self) -> CountEstimator:
        return self._instance

    def describe(self) -> Dict[str, Any]:
        return {
            "algorithm": "count-sketch",
            "parameters": self._instance.describe(),
        }


class HashMapTarget:
    def __init__(self, instance: HashMap):
        self._instance = instance

    def get_estimator(self) -> CountEstimator:
        return self._instance

    def describe(self) -> Dict[str, Any]:
        return {
            "algorithm": "hash-map",
            "parameters": {},
        }


def execute_hash_map(benchmark: Benchmark) -> None:
    benchmark.execute(HashMapTarget(HashMap()))


def execute_count_sketch(benchmark: Benchmark, depth: int, width: int) -> None:
    benchmark.execute(
        CountSketchTarget(
            CountSketch(
                HashFunctionFamily(depth, width))))


def execute_count_min_sketch(benchmark: Benchmark, depth: int, width: int) -> None:
    benchmark.execute(
        CountMinSketchTarget(
            CountMinSketch(
                HashFunctionFamily(depth, width))))


@click.command(context_settings=dict(max_content_width=120))
@click.argument("algorithm", type=click.Choice(["hash-map", "count-sketch", "count-min-sketch"]))
@click.option("-d", "--depths", type=int, multiple=True)
@click.option("-w", "--widths", type=int, multiple=True)
def execute(algorithm: str, depths: List[int], widths: List[int]) -> None:
    web_benchmark = WebNetworkBenchmark(
        threshold=50_000_000, iterations=10_000_000)

    if algorithm == "count-min-sketch":
        executor = execute_count_min_sketch
    elif algorithm == "count-sketch":
        executor = execute_count_sketch
    else:
        executor = None

    if not depths:
        depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if not widths:
        widths = [
            100, 200, 500, 1000, 2000, 5000,
            10000, 20000, 50000, 100000, 200000, 500000
        ]

    if executor:
        items = [
            (executor, web_benchmark, depth, width)
            for depth in depths
            for width in widths
        ]
    else:
        items = [(execute_hash_map, web_benchmark)]

    async def trigger() -> None:
        with ProcessPoolExecutor(max_workers=6) as executor:
            loop = asyncio.get_running_loop()
            tasks = [loop.run_in_executor(executor, *item) for item in items]

            await asyncio.wait(tasks)

    asyncio.run(trigger())
