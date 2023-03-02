import sys
import json
import random

from learning_algorithms.types import BenchmarkTarget
from learning_algorithms.network_similator import NetworkGenerator
from learning_algorithms.network_similator import WebNetworkTrafficGenerator


class WebNetworkBenchmark:
    def __init__(self, *, threshold: int, iterations: int):
        self._threshold = threshold
        self._iterations = iterations

    def execute(self, target: BenchmarkTarget) -> None:
        random.seed(0)

        network = NetworkGenerator.generate(1000000)
        [servers, clients, others] = network.split([5, 999895, 100])

        threshold, total = self._threshold, 0
        servers_found, clients_found, others_found = 0, 0, 0

        servers_min, servers_max = sys.maxsize, -sys.maxsize
        clients_min, clients_max = sys.maxsize, -sys.maxsize
        others_min, others_max = sys.maxsize, -sys.maxsize

        estimator = target.get_estimator()
        generator = WebNetworkTrafficGenerator(servers=servers, clients=clients)

        for packet in generator.stream(self._iterations):
            total += packet.size
            estimator.increment(packet.source, packet.size)
            estimator.increment(packet.destination, packet.size)

        for server in servers.addresses:
            estimate = estimator.estimate(server)
            servers_min = min(servers_min, estimate)
            servers_max = max(servers_max, estimate)
            if estimate > threshold:
                servers_found += 1

        for client in clients.addresses:
            estimate = estimator.estimate(client)
            clients_min = min(clients_min, estimate)
            clients_max = max(clients_max, estimate)
            if estimate > threshold:
                clients_found += 1

        for other in others.addresses:
            estimate = estimator.estimate(other)
            others_min = min(others_min, estimate)
            others_max = max(others_max, estimate)
            if estimate > threshold:
                others_found += 1

        result = {
            "benchmark": {
                "name": "web-network",
                "target": target.describe(),
                "threshold": threshold,
                "total": total,
            },
            "servers": {
                "total": servers.size,
                "found": servers_found,
                "estimation": {
                    "min": servers_min,
                    "max": servers_max,
                }
            },
            "clients": {
                "total": clients.size,
                "found": clients_found,
                "estimation": {
                    "min": clients_min,
                    "max": clients_max,
                }
            },
            "others": {
                "total": others.size,
                "found": others_found,
                "estimation": {
                    "min": others_min,
                    "max": others_max,
                }
            }
        }

        print(json.dumps(result, indent=None))
