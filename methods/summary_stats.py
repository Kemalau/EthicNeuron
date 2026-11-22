"""Response-level Wilson intervals and summaries across independent training seeds."""

from collections.abc import Mapping
import math
from numbers import Integral, Real
from statistics import NormalDist, mean, stdev


def wilson95(confirmed, planned):
    """Two-sided 95% Wilson limits for a response proportion, on the 0–1 scale."""
    if (not isinstance(confirmed, Integral) or isinstance(confirmed, bool)
            or not isinstance(planned, Integral) or isinstance(planned, bool)
            or not 0 <= confirmed <= planned or planned == 0):
        raise ValueError("Expected integer counts with 0 <= confirmed <= planned and planned > 0")
    k, n = int(confirmed), int(planned)
    p = k / n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    radius = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (0. if k == 0 else (center - radius) / denominator,
            1. if k == n else (center + radius) / denominator)


def training_seed_summary(outcomes):
    """Combine one scalar per independent training seed for one common endpoint.

    Pass {training_seed: per_seed_metric} for a single model, method, panel,
    domain and checkpoint step. For rates use each seed's own confirmed/planned
    fraction (or percentage); for NLL use its total nats/target-token count.
    All output values retain the input unit. SD uses n-1 across training seeds.
    """
    if not isinstance(outcomes, Mapping) or len(outcomes) < 2:
        raise ValueError("Supply one outcome for each of at least two independent training seeds")
    if any(not isinstance(s, Integral) or isinstance(s, bool) or s < 0 for s in outcomes):
        raise ValueError("Training seed IDs must be nonnegative integers")
    if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) for v in outcomes.values()):
        raise ValueError("Expected a finite scalar outcome for each training seed")
    values = [float(outcomes[s]) for s in sorted(outcomes)]
    return dict(n_seeds=len(values), training_seeds=sorted(int(s) for s in outcomes),
                mean=mean(values), sample_sd=stdev(values), minimum=min(values), maximum=max(values))
