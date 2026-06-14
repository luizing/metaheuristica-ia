import numpy as np

from src3.tsp.route import validate_route


def order_crossover(
    first_parent: np.ndarray,
    second_parent: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first_parent = np.asarray(first_parent, dtype=np.int32)
    second_parent = np.asarray(second_parent, dtype=np.int32)
    if first_parent.shape != second_parent.shape:
        raise ValueError("Parents must have equal chromosome lengths.")

    size = len(first_parent)
    validate_route(first_parent, size)
    validate_route(second_parent, size)
    first_cut, second_cut = sorted(
        rng.choice(size, size=2, replace=False).tolist()
    )
    second_cut += 1

    return (
        _create_child(first_parent, second_parent, first_cut, second_cut),
        _create_child(second_parent, first_parent, first_cut, second_cut),
    )


def _create_child(
    segment_parent: np.ndarray,
    fill_parent: np.ndarray,
    first_cut: int,
    second_cut: int,
) -> np.ndarray:
    size = len(segment_parent)
    child = np.full(size, -1, dtype=np.int32)
    child[first_cut:second_cut] = segment_parent[first_cut:second_cut]
    used = set(int(gene) for gene in child[first_cut:second_cut])

    fill_positions = list(range(second_cut, size)) + list(range(0, first_cut))
    fill_genes = [
        int(gene)
        for gene in np.concatenate(
            (fill_parent[second_cut:], fill_parent[:second_cut])
        )
        if int(gene) not in used
    ]
    child[fill_positions] = fill_genes
    return child
