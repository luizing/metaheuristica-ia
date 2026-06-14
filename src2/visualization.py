import os
import tempfile
from pathlib import Path

import numpy as np

from src2.annealing.simulated_annealing import AnnealingResult


def create_experiment_plots(
    output_directory: Path,
    results: list[AnnealingResult],
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "av3_matplotlib_cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    representative = min(
        results,
        key=lambda result: (
            not result.success,
            result.iterations if result.success else -result.fitness,
        ),
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(representative.fitness_history, color="teal")
    axis.set(
        title="Convergência do Simulated Annealing",
        xlabel="Iteração",
        ylabel="Melhor fitness",
    )
    axis.set_ylim(0, 29)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "convergencia.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(representative.temperature_history, color="darkorange")
    axis.set(
        title="Resfriamento geométrico",
        xlabel="Iteração",
        ylabel="Temperatura",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "temperatura.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        np.asarray([result.iterations for result in results]),
        bins="auto",
        color="slateblue",
        edgecolor="black",
    )
    axis.set(
        title="Distribuição das iterações",
        xlabel="Iterações",
        ylabel="Frequência",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_directory / "histograma_iteracoes.png", dpi=160)
    plt.close(fig)
