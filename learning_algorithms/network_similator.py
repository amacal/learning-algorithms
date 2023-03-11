import random

from typing import Set
from typing import List
from typing import Iterator

from dataclasses import dataclass


@dataclass(kw_only=True)
class NetworkPacket:
    source: str
    destination: str
    size: int


@dataclass(kw_only=True)
class Network:
    size: int
    addresses: List[str]

    def split(self, weights: List[int]) -> List["Network"]:
        addresses = self.addresses
        output: List[List[str]] = [list()] * len(weights)

        for i, weight in enumerate(weights):
            output[i] = random.sample(addresses, weight)
            addresses = list(set(addresses) - set(output[i]))

        return [Network(size=len(data), addresses=data) for data in output]


class NetworkGenerator:
    @staticmethod
    def generate(size: int) -> Network:
        return Network(
            size=size,
            addresses=list(NetworkGenerator._generate_addresses(size)),
        )

    @staticmethod
    def _generate_addresses(count: int) -> Iterator[str]:
        generated: Set[str] = set()

        while count > len(generated):
            found = ".".join([str(random.randint(0, 255)) for _ in range(4)])
            if found not in generated:
                generated.add(found)
                yield found


class UniformedNetworkTrafficGenerator:
    def __init__(self, network: Network):
        self._network = network

    def stream(self, count: int) -> Iterator[NetworkPacket]:
        for _ in range(count):
            yield NetworkPacket(
                source=random.choice(self._network.addresses),
                destination=random.choice(self._network.addresses),
                size=random.randint(500, 1500),
            )


class WebNetworkTrafficGenerator:
    def __init__(self, servers: Network, clients: Network):
        self._servers = servers
        self._clients = clients

    def stream(self, count: int) -> Iterator[NetworkPacket]:
        for _ in range(count):
            download_factor = random.randint(1, 10)
            is_uploading = download_factor == 1

            source = random.choice(self._clients.addresses)
            destination = random.choice(self._servers.addresses)

            yield NetworkPacket(
                source=source if is_uploading else destination,
                destination=destination if is_uploading else source,
                size=download_factor * random.randint(50, 150),
            )


class MixedNetworkTrafficGenerator:
    def __init__(self, servers: Network, clients: Network):
        self._web = WebNetworkTrafficGenerator(servers, clients)
        self._uniformed = UniformedNetworkTrafficGenerator(clients)

    def stream(self, count: int) -> Iterator[NetworkPacket]:
        web = self._web.stream(count)
        uniformed = self._uniformed.stream(count)

        for _ in range(count):
            if random.randint(1, 2) == 1:
                yield web.__next__()
            else:
                yield uniformed.__next__()
