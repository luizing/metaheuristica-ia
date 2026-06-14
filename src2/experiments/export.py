from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from src2.annealing.simulated_annealing import AnnealingResult
from src2.experiments.queens_experiment import (
    CollectionResult,
    ExperimentSummary,
    ParameterSummary,
)
from src2.queens.board import MAX_PAIRS


def export_independent_results(
    output_directory: Path,
    results: list[AnnealingResult],
    summary: ExperimentSummary,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "run",
                "solution",
                "fitness",
                "success",
                "iterations",
                "final_temperature",
                "elapsed_seconds",
            ]
        )
        for run, result in enumerate(results, start=1):
            writer.writerow(
                [
                    run,
                    " ".join(str(int(row)) for row in result.state),
                    result.fitness,
                    result.success,
                    result.iterations,
                    result.final_temperature,
                    result.elapsed_seconds,
                ]
            )
    (output_directory / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )


def export_parameter_grid(
    path: Path,
    summaries: list[ParameterSummary],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "initial_temperature",
                "alpha",
                "mean_iterations",
                "std_iterations",
                "mean_seconds",
                "success_rate",
                "best_fitness",
                "worst_fitness",
            ]
        )
        for parameter in summaries:
            summary = parameter.summary
            writer.writerow(
                [
                    parameter.initial_temperature,
                    parameter.alpha,
                    summary.mean_iterations,
                    summary.standard_deviation_iterations,
                    summary.mean_seconds,
                    summary.success_rate,
                    summary.best_fitness,
                    summary.worst_fitness,
                ]
            )


def export_collection(
    output_directory: Path,
    collection: CollectionResult,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "solutions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "number",
                "solution",
                "fitness",
                "restart",
                "iterations",
                "elapsed_seconds",
                "final_temperature",
            ]
        )
        for discovery in collection.discoveries:
            writer.writerow(
                [
                    discovery.solution_number,
                    " ".join(str(row) for row in discovery.solution),
                    MAX_PAIRS,
                    discovery.restart,
                    discovery.iterations,
                    discovery.elapsed_seconds,
                    discovery.final_temperature,
                ]
            )

    metrics = {
        "solutions_found": len(collection.repository),
        "restarts": collection.restarts,
        "successful_restarts": collection.successful_restarts,
        "duplicate_solutions": collection.duplicate_solutions,
        "failed_restarts": collection.failed_restarts,
        "total_iterations": collection.total_iterations,
        "elapsed_seconds": collection.elapsed_seconds,
        "peak_memory_bytes": collection.peak_memory_bytes,
        "peak_memory_megabytes": collection.peak_memory_bytes / (1024**2),
    }
    (output_directory / "collection_summary.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def export_markdown_report(
    path: Path,
    summary: ExperimentSummary | None,
    parameters: list[ParameterSummary] | None,
    collection: CollectionResult | None,
) -> None:
    lines = ["# Resultados da Parte 2 - 8 Rainhas", ""]

    if summary is not None:
        lines.extend(
            [
                "## Experimento principal",
                "",
                "| Execuções | Sucessos | Taxa de sucesso | Média iterações | "
                "Desvio padrão | Média de tempo (s) | Melhor fitness | Pior fitness |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {summary.runs} | {summary.successes} | "
                f"{summary.success_rate:.2%} | {summary.mean_iterations:.3f} | "
                f"{summary.standard_deviation_iterations:.3f} | "
                f"{summary.mean_seconds:.6f} | {summary.best_fitness} | "
                f"{summary.worst_fitness} |",
                "",
            ]
        )

    if parameters is not None:
        lines.extend(
            [
                "## Estudo de temperatura e alpha",
                "",
                "| Temperatura | Alpha | Média iterações | Taxa de sucesso | "
                "Média de tempo (s) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for parameter in parameters:
            lines.append(
                f"| {parameter.initial_temperature:g} | {parameter.alpha:g} | "
                f"{parameter.summary.mean_iterations:.3f} | "
                f"{parameter.summary.success_rate:.2%} | "
                f"{parameter.summary.mean_seconds:.6f} |"
            )
        lines.append("")

    if collection is not None:
        lines.extend(
            [
                "## Coleta de soluções distintas",
                "",
                f"- Soluções encontradas: {len(collection.repository)}",
                f"- Reinicializações: {collection.restarts}",
                f"- Iterações totais: {collection.total_iterations}",
                f"- Tempo total: {collection.elapsed_seconds:.6f} s",
                f"- Pico de memória: {collection.peak_memory_bytes / (1024**2):.3f} MB",
                f"- Soluções duplicadas: {collection.duplicate_solutions}",
                f"- Reinicializações sem solução: {collection.failed_restarts}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
