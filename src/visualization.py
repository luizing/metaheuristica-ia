from pathlib import Path
import os
import tempfile

import numpy as np

from src.core.experiment import ALGORITHMS, mean_convergence
from src.core.problem import Problem
from src.core.result import SearchResult


def create_plots(
    output_directory: Path,
    problem: Problem,
    all_results: dict[str, list[SearchResult]],
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "av3_matplotlib_cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    slug = problem.name.split(" - ", 1)[0].lower()

    fig, axis = plt.subplots(figsize=(8, 5))
    for algorithm in ALGORITHMS:
        axis.plot(mean_convergence(all_results[algorithm]), label=algorithm)
    axis.set(
        title=f"Convergência média - {problem.name}",
        xlabel="Iteração",
        ylabel="Melhor valor acumulado",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_directory / f"{slug}_convergencia.png", dpi=160)
    plt.close(fig)

    values = [
        [result.best_value for result in all_results[algorithm]]
        for algorithm in ALGORITHMS
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=list(ALGORITHMS), showmeans=True)
    axis.set(
        title=f"Distribuição das execuções - {problem.name}",
        xlabel="Algoritmo",
        ylabel="Melhor valor final",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / f"{slug}_boxplot.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, algorithm, algorithm_values in zip(axes, ALGORITHMS, values):
        axis.hist(algorithm_values, bins="auto", edgecolor="black", alpha=0.8)
        axis.set_title(algorithm)
        axis.set_xlabel("Melhor valor final")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Frequência")
    fig.suptitle(f"Histogramas - {problem.name}")
    fig.tight_layout()
    fig.savefig(output_directory / f"{slug}_histograma.png", dpi=160)
    plt.close(fig)
