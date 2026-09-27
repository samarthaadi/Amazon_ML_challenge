"""Stage 3: pair scores on the exact retrieved input.

Implementation TBD: architectures, features, losses, and negative sampling.
Only the unchanged historical baseline is adapted here for comparison.
"""
import time

import pandas as pd

from .data import PAIR_KEYS, legacy, validate_pairs


def features(candidates, records, tokens, state):
    if state["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: matching features")
    baseline = legacy()
    references = records.set_index("entity_id").to_dict("index")
    token_map = baseline.token_maps(tokens)
    result = []
    for row in candidates.itertuples(index=False):
        candidate = {"business_name": row.candidate_name, "business_address": row.candidate_address}
        selected = token_map.get(row.source1_entity_id, {})
        result.append(baseline.feature_row(references[row.source1_entity_id], candidate,
                                           selected.get("name", ()), selected.get("address", ())))
    return pd.DataFrame(result, columns=baseline.FEATURES)


def fit(training_candidates, records, training_labels, config, tokens):
    if config["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: matcher")
    validate_pairs(training_candidates, training_labels)
    matrix = features(training_candidates, records, tokens, config)
    labels = [int(r.candidate_entity_id in training_labels[r.source1_entity_id])
              for r in training_candidates.itertuples(index=False)]
    if len(set(labels)) != 2:
        raise ValueError("Legacy baseline needs both retrieved positives and negatives")
    # Exact existing parameters, not a new model architecture selection.
    model = legacy().HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
        l2_regularization=1.0, class_weight="balanced", early_stopping=False,
        random_state=config["seed"],
    )
    model.fit(matrix[legacy().FEATURES], labels)
    return {"method": "legacy_v1", "model": model}


def score(candidates, records, state, tokens):
    started = time.perf_counter()
    validate_pairs(candidates, records.entity_id)
    result = candidates[PAIR_KEYS].copy()
    if len(candidates):
        matrix = features(candidates, records, tokens, state)
        result["score"] = state["model"].predict_proba(matrix[legacy().FEATURES])[:, 1]
    else:
        result["score"] = pd.Series(dtype=float)
    return result, {"seconds": time.perf_counter() - started, "pairs": len(result)}
