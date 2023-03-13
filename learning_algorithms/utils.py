from typing import Tuple


def resolution_to_shape(resolution: str) -> Tuple[int, int]:
    return tuple(map(int, resolution.split("x")))


def shape_to_resolution(shape: Tuple[int, int]) -> str:
    return "x".join(map(str, shape))
