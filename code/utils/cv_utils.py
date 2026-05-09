import numpy as np


def make_kfold_splits(
    n_cells: int, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Permute cell indices once with a fixed seed and return
    [(train_idx, test_idx), ...] for each of k folds.
    All ensemble seeds share the same fold splits because cv_seed is fixed.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cells)

    fold_sizes = np.full(k, n_cells // k)
    fold_sizes[: n_cells % k] += 1  # distribute remainder to first folds

    splits = []
    current = 0
    for size in fold_sizes:
        test_idx = perm[current : current + size]
        train_idx = np.concatenate([perm[:current], perm[current + size :]])
        splits.append((train_idx, test_idx))
        current += size

    return splits
