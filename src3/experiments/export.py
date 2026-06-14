from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src3.experiments.tsp_experiment import ExperimentResult
from src3.tsp.point import PointDataset


def export_experiment(
    output_directory: Path,
    experiment: ExperimentResult,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "run",
                "generations",
                "generation_to_acceptable",
                "final_distance",
                "elapsed_seconds",
                "valid_route",
                "route",
            ]
        )
        for run, result in enumerate(experiment.results, start=1):
            writer.writerow(
                [
                    run,
                    result.generations,
                    result.generations_to_acceptable,
                    result.best.distance,
                    result.elapsed_seconds,
                    True,
                    " ".join(
                        str(int(gene) + 1)
                        for gene in result.best.chromosome
                    ),
                ]
            )

    payload = {
        "summary": asdict(experiment.summary),
        "random_baseline": asdict(experiment.baseline),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def export_generation_history(
    path: Path,
    experiment: ExperimentResult,
) -> None:
    best_result = min(
        experiment.results,
        key=lambda result: result.best.distance,
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "generation",
                "best_distance",
                "average_distance",
                "worst_distance",
                "best_fitness",
                "average_fitness",
                "worst_fitness",
            ]
        )
        for statistics in best_result.history:
            writer.writerow(asdict(statistics).values())


def export_comparison(
    path: Path,
    comparisons: dict[int, ExperimentResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "elite_size",
                "best_distance",
                "worst_distance",
                "mean_distance",
                "std_distance",
                "mode_generations",
                "mean_generations",
                "mean_seconds",
                "acceptable_success_rate",
                "improvement_over_random_percent",
            ]
        )
        for elite_size, experiment in sorted(comparisons.items()):
            summary = experiment.summary
            writer.writerow(
                [
                    elite_size,
                    summary.best_distance,
                    summary.worst_distance,
                    summary.mean_distance,
                    summary.standard_deviation_distance,
                    summary.mode_generations,
                    summary.mean_generations,
                    summary.mean_seconds,
                    summary.acceptable_success_rate,
                    summary.improvement_over_random_percent,
                ]
            )


def export_markdown(
    path: Path,
    dataset: PointDataset,
    comparisons: dict[int, ExperimentResult],
) -> None:
    lines = [
        "# Resultados da Parte 3 - Caixeiro Viajante 3D",
        "",
        f"- Grupo utilizado: {dataset.group}",
        f"- Quantidade de pontos: {dataset.size}",
        "- A origem não faz parte do cromossomo e é incluída no início e no fim.",
        "",
        "## Comparação de elitismo",
        "",
        "| Elite | Melhor distância | Pior distância | Média | Desvio padrão | "
        "Moda gerações | Tempo médio (s) | Melhoria sobre aleatório |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for elite_size, experiment in sorted(comparisons.items()):
        summary = experiment.summary
        lines.append(
            f"| {elite_size} | {summary.best_distance:.6f} | "
            f"{summary.worst_distance:.6f} | {summary.mean_distance:.6f} | "
            f"{summary.standard_deviation_distance:.6f} | "
            f"{summary.mode_generations} | {summary.mean_seconds:.6f} | "
            f"{summary.improvement_over_random_percent:.2f}% |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
