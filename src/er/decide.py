"""Stage 4: candidate scores to accepted sets, including empty sets.

Implementation TBD: new decision rules, thresholds/search, calibration, ties.
legacy_v1 reproduces the existing validation threshold selection exactly.
"""
import numpy as np

from .data import validate_pairs
from .evaluate import macro_f05


def apply(scores, query_ids, policy):
    if policy["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: decision policy")
    query_ids = list(query_ids)
    validate_pairs(scores, query_ids)
    if not np.isfinite(scores.score.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite matcher score")
    result = {q: set() for q in query_ids}
    for row in scores[scores.score.ge(policy["threshold"])].itertuples(index=False):
        result[row.source1_entity_id].add(row.candidate_entity_id)
    return result


def tune(validation_scores, validation_truth, query_ids, config):
    if config["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: decision tuning")
    query_ids = list(query_ids)
    if not query_ids or set(query_ids) != set(validation_truth):
        raise ValueError("Tuning requires complete validation truth and nonempty queries")
    quantiles = validation_scores.score.quantile(np.linspace(0, 1, 101)).dropna().to_numpy()
    thresholds = sorted(set(np.r_[np.linspace(0.01, 0.99, 99), quantiles, 1.000001]))
    trials = []
    for threshold in thresholds:
        policy = {"method": "legacy_v1", "threshold": float(threshold)}
        predictions = apply(validation_scores, query_ids, policy)
        trials.append({"threshold": float(threshold), "macro_f0_5": macro_f05(query_ids, validation_truth, predictions)})
    best = max(trials, key=lambda r: (r["macro_f0_5"], r["threshold"]))
    return {"method": "legacy_v1", "threshold": best["threshold"]}, trials
