import os
import tempfile
from pathlib import Path

import numpy as np

from src3.experiments.tsp_experiment import ExperimentResult
from src3.tsp.point import PointDataset


def _matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "av3_matplotlib_cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mean_history(
    experiment: ExperimentResult,
    attribute: str,
) -> np.ndarray:
    maximum_length = max(len(result.history) for result in experiment.results)
    histories = np.empty((len(experiment.results), maximum_length))
    for row, result in enumerate(experiment.results):
        values = np.asarray(
            [getattr(statistics, attribute) for statistics in result.history]
        )
        histories[row, : len(values)] = values
        histories[row, len(values) :] = values[-1]
    return np.mean(histories, axis=0)


def create_experiment_plots(
    output_directory: Path,
    experiment: ExperimentResult,
    dataset: PointDataset,
) -> None:
    plt = _matplotlib()
    output_directory.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(_mean_history(experiment, "best_fitness"), color="teal")
    axis.set(
        title="Convergência média do melhor fitness",
        xlabel="Geração",
        ylabel="Melhor fitness",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "convergencia.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(_mean_history(experiment, "average_fitness"), color="darkorange")
    axis.set(
        title="Fitness médio da população",
        xlabel="Geração",
        ylabel="Fitness médio",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "fitness_medio.png", dpi=160)
    plt.close(fig)

    generations = [
        result.generations_to_acceptable
        if result.generations_to_acceptable is not None
        else result.generations
        for result in experiment.results
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(generations, bins="auto", color="slateblue", edgecolor="black")
    axis.set(
        title="Distribuição das gerações",
        xlabel="Gerações",
        ylabel="Frequência",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "histograma_geracoes.png", dpi=160)
    plt.close(fig)

    best_result = min(
        experiment.results,
        key=lambda result: result.best.distance,
    )
    route_points = dataset.points[best_result.best.chromosome]
    closed_route = np.vstack((dataset.origin, route_points, dataset.origin))
    fig = plt.figure(figsize=(9, 7))
    axis = fig.add_subplot(projection="3d")
    axis.plot(
        closed_route[:, 0],
        closed_route[:, 1],
        closed_route[:, 2],
        color="teal",
        linewidth=1.2,
    )
    axis.scatter(
        dataset.points[:, 0],
        dataset.points[:, 1],
        dataset.points[:, 2],
        color="navy",
        s=22,
        label="Pontos",
    )
    axis.scatter(
        *dataset.origin,
        color="red",
        marker="*",
        s=160,
        label="Origem",
    )
    axis.set_title(f"Melhor rota - distância {best_result.best.distance:.3f}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "melhor_rota_3d.png", dpi=160)
    plt.close(fig)


def create_comparison_plots(
    output_directory: Path,
    comparisons: dict[int, ExperimentResult],
) -> None:
    plt = _matplotlib()
    output_directory.mkdir(parents=True, exist_ok=True)
    elite_sizes = sorted(comparisons)

    fig, axis = plt.subplots(figsize=(8, 5))
    for elite_size in elite_sizes:
        axis.plot(
            _mean_history(comparisons[elite_size], "best_distance"),
            label=f"Elite {elite_size}",
        )
    axis.set(
        title="Convergência com e sem elitismo",
        xlabel="Geração",
        ylabel="Melhor distância",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "comparacao_convergencia.png", dpi=160)
    plt.close(fig)

    values = [
        [result.best.distance for result in comparisons[elite].results]
        for elite in elite_sizes
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        values,
        tick_labels=[f"Elite {elite}" for elite in elite_sizes],
        showmeans=True,
    )
    axis.set(
        title="Qualidade final com e sem elitismo",
        ylabel="Distância final",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "comparacao_distancias.png", dpi=160)
    plt.close(fig)
