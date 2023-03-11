import os
import os.path

import click
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from typing import List


def load_benchmark(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = pd.json_normalize(df.to_dict("records"))

    return df


def combine_benchmarks(benchmarks: List[pd.DataFrame]) -> pd.DataFrame:
    columns = list(set([
        column for benchmark in benchmarks
        for column in benchmark.columns
    ]))

    for column in columns:
        for benchmark in benchmarks:
            if column not in benchmark.columns:
                benchmark[column] = None

    return pd.concat(benchmarks, ignore_index=True)


@click.command(context_settings=dict(max_content_width=120))
@click.argument("path", type=click.STRING)
@click.argument("benchmark", type=click.Choice(["web-network", "mixed-network"]))
@click.argument("algorithm", type=click.Choice(["hash-map", "count-sketch", "count-min-sketch"]))
@click.argument("target", type=click.Choice(["servers", "clients", "others"]))
def analyze(path: str, benchmark: str, algorithm: str, target: str) -> None:
    pd.set_option("display.max_rows", None)

    df = combine_benchmarks([
        load_benchmark(os.path.join(path, item))
        for item in os.listdir(path) if item.endswith(".json")
    ])

    df = df[df["benchmark.name"] == benchmark]
    df = df.rename(columns={
        "benchmark.name": "name",
        "benchmark.target.algorithm": "algorithm",
        "benchmark.target.parameters.depth": "depth",
        "benchmark.target.parameters.width": "width",
        "servers.estimation.min": "servers",
        "clients.estimation.max": "clients",
        "others.estimation.max": "others",
    })

    ideal_servers = df[df["algorithm"] == "hash-map"]["servers"].max()
    ideal_clients = df[df["algorithm"] == "hash-map"]["clients"].max()

    df["servers"] = df["servers"] / ideal_servers
    df["clients"] = df["clients"] / ideal_clients

    df = df[["name", "algorithm", "depth", "width", target]]
    df = df[df["algorithm"] == algorithm].pivot(index="depth", columns="width", values=target)

    row_labels = df.index.astype(str)
    col_labels = df.columns.astype(str)

    plt.figure(figsize=(19.20, 10.80))
    plt.axis("off")

    _, ax = plt.subplots(figsize=(40, 10.80))
    ax.matshow(df, cmap=plt.cm.Reds, norm=LogNorm())

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Width")
    ax.set_ylabel("Depth")

    if target == "clients":
        convert = lambda i, j: f"{float(df.iloc[i, j]):.1f}"
    elif target == "servers":
        convert = lambda i, j: f"{float(df.iloc[i, j]):.4f}"
    else:
        convert = lambda i, j: str(int(df.iloc[i, j]))

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, convert(i, j), va='center', ha='center')

    plt.savefig(os.path.join(path, f"{benchmark}-{algorithm}-{target}.png"))

