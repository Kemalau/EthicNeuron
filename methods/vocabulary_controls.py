"""Layer-matched random writes and centred FastICA for vocabulary readout."""

from collections import defaultdict
import math
import warnings

import numpy as np
import torch


@torch.no_grad()
def dose_weighted_writes(columns, gaps, alpha):
    """Return FP32 [channel, hidden] writes on the device of the weight columns."""
    if columns.ndim != 2 or not columns.is_floating_point() or not columns.numel():
        raise ValueError("Expected a nonempty floating-point [hidden, channel] tensor")
    gaps = np.asarray(gaps, dtype=np.float64)
    if gaps.shape != (columns.shape[1],) or not math.isfinite(alpha):
        raise ValueError("Expected one signed gap per column and a finite dose")
    doses = torch.as_tensor(alpha * gaps, dtype=torch.float32, device=columns.device)
    writes = (columns.T.float() * doses[:, None]).contiguous()
    if not torch.isfinite(writes).all():
        raise ValueError("Nonfinite dose-weighted write")
    return writes


def matched_random_writes(selected, score_layers, mean_gaps, get_columns, *, alpha=2., seeds=range(10)):
    """Yield (audit, scaled raw writes) per layer and seed, before lens transport.

    Pass the Pub97 discovery score archive's layers and mean_gap arrays.
    get_columns(layer, sorted_neuron_ids) supplies [hidden, count] weight columns
    in the same recipient for selected and sampled neurons. Each draw uses its
    own signed gaps and matches the norm of the sum of the selected raw writes.
    """
    layers = np.asarray(score_layers)
    mean_gaps = np.asarray(mean_gaps)
    if (layers.ndim != 1 or layers.dtype.kind not in "iu" or np.any(layers < 0)
            or len(np.unique(layers)) != len(layers) or mean_gaps.ndim != 2
            or mean_gaps.shape[0] != len(layers) or not np.isfinite(mean_gaps).all()):
        raise ValueError("Expected unique layer IDs and a finite [layer, channel] gap table")
    seeds = tuple(seeds)
    if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("Seeds must be distinct nonnegative integers")
    gaps_by_layer = dict(zip(layers.tolist(), mean_gaps))
    targets = defaultdict(set)
    for row in selected:
        layer, neuron = row["layer"], row["neuron"]
        if (type(layer) is not int or layer not in gaps_by_layer or type(neuron) is not int
                or not 0 <= neuron < mean_gaps.shape[1] or neuron in targets[layer]):
            raise ValueError("Expected unique coordinates belonging to the discovery gap table")
        if float(row["mean_activation_gap"]) != float(gaps_by_layer[layer][neuron]):
            raise ValueError("Selected gap differs from the discovery gap table")
        targets[layer].add(neuron)
    if not targets:
        raise ValueError("Expected a nonempty selection")

    for layer, neurons in sorted(targets.items()):
        ids = sorted(neurons)
        pool = np.setdiff1d(np.arange(mean_gaps.shape[1]), ids, assume_unique=True)
        if len(pool) < len(ids):
            raise ValueError(f"Too few unselected channels at layer {layer}")
        gaps = gaps_by_layer[layer]
        selected_writes = dose_weighted_writes(get_columns(layer, ids), gaps[ids], alpha)
        target_norm = selected_writes.sum(0).double().norm().item()
        if not math.isfinite(target_norm) or target_norm <= 0:
            raise ValueError(f"Undefined selected write direction at layer {layer}")
        for seed in seeds:
            members = sorted(np.random.default_rng(seed).choice(pool, len(ids), replace=False).tolist())
            raw = dose_weighted_writes(get_columns(layer, members), gaps[members], alpha)
            raw_norm = raw.sum(0).double().norm().item()
            if not math.isfinite(raw_norm) or raw_norm <= 0:
                raise ValueError(f"Undefined random write direction at layer {layer}, seed {seed}")
            scale = target_norm / raw_norm
            matched = raw * scale
            matched_norm = matched.sum(0).double().norm().item()
            if not math.isclose(matched_norm, target_norm, rel_tol=1e-6):
                raise ValueError(f"Write-norm matching failed at layer {layer}, seed {seed}")
            yield dict(layer=layer, seed=seed, neurons=members, mean_gaps=gaps[members].tolist(),
                       alpha=alpha, scale=scale, target_write_norm=target_norm, raw_write_norm=raw_norm,
                       matched_write_norm=matched_norm), matched


def centered_fastica(writes, n_components, *, seed=0):
    """Fit one selected population's transported write rows; return directions and fit metadata.

    FP64 FastICA centres each hidden dimension over neurons. The outputs for the
    vocabulary head are rows of shape [n_components, hidden] (mixing columns),
    oriented toward the population's total signed write, returned as CPU FP32.
    """
    try:
        from sklearn import __version__ as sklearn_version
        from sklearn.decomposition import FastICA
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:
        raise ImportError("Centred FastICA requires scikit-learn >= 1.2") from exc
    if (writes.ndim != 2 or not writes.is_floating_point() or writes.shape[0] < 2
            or writes.shape[1] < 1 or type(n_components) is not int or n_components < 1
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Expected multiple write rows, positive component count and a uint32 seed")
    matrix = writes.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("FastICA requires finite write vectors")
    centered = matrix - matrix.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    # The transported observations originate in FP32 even though fitting uses FP64.
    tolerance = singular[0] * max(matrix.shape) * np.finfo(np.float32).eps
    rank = int(np.count_nonzero(singular > tolerance))
    if n_components > rank:
        raise ValueError(f"Component count {n_components} exceeds centred numerical rank {rank}")
    settings = dict(n_components=n_components, algorithm="parallel", whiten="unit-variance",
                    whiten_solver="svd", fun="logcosh", max_iter=4000, tol=1e-4, random_state=seed)
    fit = FastICA(**settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        loadings = fit.fit_transform(matrix)
    directions = fit.mixing_.T.copy()
    if not np.isfinite(directions).all() or not np.isfinite(loadings).all():
        raise ValueError("FastICA produced a nonfinite decomposition")
    total = matrix.sum(axis=0)
    for index, vector in enumerate(directions):
        anchor = vector @ total
        if abs(anchor) <= 1e-12 * np.linalg.norm(vector) * np.linalg.norm(total):
            anchor = vector[np.argmax(np.abs(vector))]
        if anchor < 0:
            directions[index] *= -1
            loadings[:, index] *= -1
    residual = centered - loadings @ directions
    explained = 1 - (np.linalg.norm(residual) / np.linalg.norm(centered))**2
    audit = dict(**settings, sklearn_version=sklearn_version, centered_rank_fp32=rank,
                 iterations=int(fit.n_iter_), explained_centered_fraction=float(explained),
                 converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return torch.from_numpy(directions.astype(np.float32)), audit
