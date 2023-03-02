class HashMap:
    def __init__(self):
        self._data = dict()

    def increment(self, key: str, count: int) -> None:
        self._data[key] = self.estimate(key) + count

    def estimate(self, key: str) -> int:
        return self._data.get(key, 0)
