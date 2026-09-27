"""Stage 5: official scoring, stage diagnostics, reports, and error attribution.

Definition: student_resource/README.md, Evaluation Criteria. No extra model stage.
Undefined diagnostic ratios are null; singleton official scores are defined.
"""
from __future__ import annotations

import json

import pandas as pd

from .data import validate_pairs


def f05(truth, prediction):
    truth, prediction = set(truth), set(prediction)
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    return 1.25 * tp / (0.25 * len(truth) + len(prediction))


def macro_f05(query_ids, truth, predictions):
    ids = list(query_ids)
    if not ids or set(ids) != set(truth) or set(ids) != set(predictions):
        raise ValueError("Scoring requires complete truth/predictions for a nonempty query list")
    return sum(f05(truth[q], predictions[q]) for q in ids) / len(ids)


def ratio(n, d):
    return n / d if d else None


def evaluate(queries, truth, candidates, predictions, ks=(10, 25, 50, 100)):
    """Queries map IDs to countries. Returns per-business rows and pair errors.

    Batch only on query boundaries. summarize() can combine these rows without
    retaining candidate-level predictions in memory for the entire corpus.
    """
    if set(queries) != set(truth) or set(queries) != set(predictions):
        raise ValueError("Incomplete truth/prediction query coverage")
    validate_pairs(candidates, queries)
    by_query = {q: part for q, part in candidates.groupby("source1_entity_id", sort=False)}
    empty = candidates.iloc[:0]
    results, errors = [], []
    for q, country in queries.items():
        part = by_query.get(q, empty)
        retrieved = set(part.candidate_entity_id)
        accepted, actual = set(predictions[q]), set(truth[q])
        if not accepted <= retrieved:
            raise ValueError("Predicted match absent from retrieved candidates")
        tp = len(accepted & actual)
        row = {
            "source1_entity_id": q, "country": country, "has_matches": bool(actual),
            "true_matches": len(actual), "candidate_count": len(retrieved),
            "retrieved_true_matches": len(actual & retrieved), "accepted_matches": len(accepted),
            "true_acceptances": tp, "false_acceptances": len(accepted - actual),
            "all_true_matches_retrieved": bool(actual) and actual <= retrieved,
            "singleton_false_acceptance": not actual and bool(accepted),
            "macro_f0_5": f05(actual, accepted), "retrieval_ceiling": f05(actual, actual & retrieved),
        }
        for k in ks:
            selected = set(part.loc[part["rank"].le(k), "candidate_entity_id"])
            row[f"retrieved_true_at_{k}"] = len(actual & selected)
            if "name_rank" in part and "address_rank" in part:
                union = set(part.loc[part.name_rank.le(k) | part.address_rank.le(k), "candidate_entity_id"])
                row[f"legacy_union_true_at_per_route_{k}"] = len(actual & union)
        results.append(row)
        for kind, ids in (("retrieval_miss", actual - retrieved),
                          ("matcher_false_rejection", (actual & retrieved) - accepted),
                          ("false_acceptance", accepted - actual)):
            errors.extend({"source1_entity_id": q, "candidate_entity_id": target,
                           "country": country, "error_type": kind} for target in sorted(ids))
    return pd.DataFrame(results), pd.DataFrame(errors, columns=["source1_entity_id", "candidate_entity_id", "country", "error_type"])


def _summary(frame):
    n = len(frame)
    matched = int(frame.has_matches.sum()) if n else 0
    truth = int(frame.true_matches.sum()) if n else 0
    retrieved = int(frame.retrieved_true_matches.sum()) if n else 0
    accepted = int(frame.accepted_matches.sum()) if n else 0
    tp = int(frame.true_acceptances.sum()) if n else 0
    single_fp = int(frame.singleton_false_acceptance.sum()) if n else 0
    all_retrieved = int(frame.all_true_matches_retrieved.sum()) if n else 0
    result = {
        "queries": n, "matched_queries": matched, "singleton_queries": n - matched,
        "true_matches": truth, "retrieved_true_matches": retrieved,
        "accepted_matches": accepted, "true_acceptances": tp,
        "retrieval": {
            "recall": ratio(retrieved, truth), "all_matches_retrieved": all_retrieved,
            "all_matches_retrieved_rate": ratio(all_retrieved, matched),
            "oracle_macro_f0_5": float(frame.retrieval_ceiling.mean()) if n else None,
            "candidate_total": int(frame.candidate_count.sum()) if n else 0,
            "candidate_mean": float(frame.candidate_count.mean()) if n else None,
            "candidate_median": float(frame.candidate_count.median()) if n else None,
            "candidate_p95": float(frame.candidate_count.quantile(.95)) if n else None,
            "candidate_max": int(frame.candidate_count.max()) if n else None,
            "zero_candidate_queries": int(frame.candidate_count.eq(0).sum()) if n else 0,
        },
        "matching_decisions": {
            "precision": ratio(tp, accepted), "recall_on_retrieved": ratio(tp, retrieved),
            "singleton_false_acceptances": single_fp,
            "singleton_false_acceptance_rate": ratio(single_fp, n - matched),
        },
        "end_to_end": {"macro_f0_5": float(frame.macro_f0_5.mean()) if n else None,
                       "pair_recall_including_retrieval_misses": ratio(tp, truth)},
        "errors": {"retrieval_miss": truth - retrieved, "matcher_false_rejection": retrieved - tp,
                   "false_acceptance": accepted - tp},
    }
    for prefix, output in (("retrieved_true_at_", "recall_at_k"),
                           ("legacy_union_true_at_per_route_", "legacy_union_recall_at_per_route_k")):
        result["retrieval"][output] = {
            column.removeprefix(prefix): {"retrieved_true_matches": int(frame[column].sum()),
                                         "recall": ratio(int(frame[column].sum()), truth)}
            for column in frame if column.startswith(prefix)
        }
    return result


def summarize(per_business):
    result = {"all": _summary(per_business)}
    for country, part in per_business.groupby("country", sort=True):
        result[f"country={country}"] = _summary(part)
        for label, flag in (("matched", True), ("no_matches", False)):
            result[f"country={country}/{label}"] = _summary(part[part.has_matches.eq(flag)])
    for label, flag in (("matched", True), ("no_matches", False)):
        result[label] = _summary(per_business[per_business.has_matches.eq(flag)])
    return result


def report(path, manifest, metrics):
    lines = ["# Entity-resolution experiment", "", f"Status: **{manifest['status']}**.", "",
             f"Setting: `{manifest['setting']}`. Query scope: `{manifest['query_scope']}`.", "",
             "| Group | Businesses | Macro F0.5 | Retrieval recall | Candidate precision | Retrieved recall |",
             "|---|---:|---:|---:|---:|---:|"]
    def number(value):
        return "—" if value is None else f"{value:.6f}"
    for label, row in metrics.items():
        lines.append(f"| {label} | {row['queries']} | {number(row['end_to_end']['macro_f0_5'])} | "
                     f"{number(row['retrieval']['recall'])} | {number(row['matching_decisions']['precision'])} | "
                     f"{number(row['matching_decisions']['recall_on_retrieved'])} |")
    lines.extend(["", "Runtime (seconds; imported retrieval time is not remeasured):", "",
                  "```json", json.dumps(manifest["runtime"], indent=2), "```", "",
                  "Historical baseline scores use different development groups and are not paired comparisons.",
                  "Country-transfer results are stress tests, not estimates of France accuracy.",
                  "Destination validation results are development feedback once inspected.", ""])
    if "historical_baseline" in manifest:
        lines.extend(["Historical reference (different query groups):", ""])
        for split, row in manifest["historical_baseline"].items():
            lines.append(f"- {split}: {row['references']} businesses; macro F0.5 {row['macro_f0_5']:.6f}.")
    path.write_text("\n".join(lines))


# Frozen independently of retrieval. Corrections require a new version and reruns.
BLOCKING_VERSION = "blocking-v2"


def blocking(queries, truth, candidates, *, completed_queries, known_ids, budget,
             route_budgets=None, missing_fields=None, ks=(10, 25, 50, 100)):
    """Evaluate an immutable candidate artifact; never generate/repair candidates.

    known_ids validates IDs against the authoritative pool (a callable or set).
    completed_queries comes from a verified completion manifest, including zeros.
    Ranks are contiguous, one-based within each query/route. No label columns.
    """
    import numpy as np
    if not queries or set(queries) != set(truth):
        raise ValueError("Blocking requires nonempty, complete queries and truth")
    missing_keys = {"missing_name", "missing_address"}
    if missing_fields is not None:
        if set(missing_fields) != set(queries):
            raise ValueError("Missing-field metadata must cover exactly the evaluation queries")
        for fields in missing_fields.values():
            if not isinstance(fields, dict) or set(fields) != missing_keys:
                raise ValueError("Only missing_name and missing_address metadata are allowed")
            if any(not isinstance(value, (bool, np.bool_)) for value in fields.values()):
                raise ValueError("Missing-field flags must be booleans")
    if len(completed_queries) != len(queries) or set(completed_queries) != set(queries):
        raise ValueError("Incomplete/duplicate query completion; retrieval may have failed")
    if budget < 1 or any(k < 1 for k in ks):
        raise ValueError("Budgets and prefixes must be positive")
    if {"label", "is_match", "truth", "matched_entity_ids"} & set(candidates.columns):
        raise ValueError("Candidate artifacts must be label-free")
    validate_pairs(candidates, queries)
    referenced = set(candidates.candidate_entity_id).union(*(set(t) for t in truth.values()))
    valid = known_ids(referenced) if callable(known_ids) else referenced <= set(known_ids)
    if not valid:
        raise ValueError("Unknown candidate/truth record ID")
    for col in ["rank", *(f"{r}_rank" for r in (route_budgets or {}))]:
        if col not in candidates:
            raise ValueError(f"Missing rank column: {col}")
        numeric = pd.to_numeric(candidates[col], errors="raise")
        if col == "rank" and numeric.isna().any():
            raise ValueError("Missing overall rank")
        present = numeric.dropna()
        if not np.isfinite(present).all() or (present < 1).any() or (present % 1 != 0).any():
            raise ValueError("Ranks must be finite positive integers")
        for _, part in candidates.assign(_rank=numeric).groupby("source1_entity_id"):
            ranks = sorted(part._rank.dropna().astype(int))
            if ranks != list(range(1, len(ranks) + 1)):
                raise ValueError("Ranks must be unique and contiguous")
            limit = budget if col == "rank" else route_budgets[col.removesuffix("_rank")]
            if len(ranks) > limit:
                raise ValueError("Candidate budget exceeded")
    if route_budgets is not None:
        if sum(route_budgets.values()) > budget or any(v < 0 for v in route_budgets.values()):
            raise ValueError("Invalid route allocation")
        memberships = []
        for route in route_budgets:
            present = candidates[f"{route}_rank"].notna()
            memberships.append(present)
            if f"{route}_score" in candidates:
                score = pd.to_numeric(candidates[f"{route}_score"], errors="raise")
                if not np.isfinite(score[present]).all() or score[~present].notna().any():
                    raise ValueError("Invalid route scores")
            if f"in_{route}" in candidates and not candidates[f"in_{route}"].eq(present).all():
                raise ValueError("Route membership disagrees with ranks")
        if len(candidates) and not pd.concat(memberships, axis=1).any(axis=1).all():
            raise ValueError("Candidate absent from every route")
    if "candidate_source" in candidates and not candidates.candidate_source.eq(
            candidates.candidate_entity_id.str.split("-").str[0]).all():
        raise ValueError("Candidate source disagrees with source-qualified ID")
    grouped = dict(tuple(candidates.groupby("source1_entity_id", sort=False)))
    per_query, misses = [], []
    for q, country in queries.items():
        part = grouped.get(q, candidates.iloc[:0])
        actual, retrieved = set(truth[q]), set(part.candidate_entity_id)
        hits = actual & retrieved
        lexical = set(part.loc[part[[c for c in ("name_rank", "address_rank") if c in part]].notna().any(axis=1),
                               "candidate_entity_id"]) if any(c in part for c in ("name_rank", "address_rank")) else set()
        dense = set(part.loc[part.dense_rank.notna(), "candidate_entity_id"]) if "dense_rank" in part else set()
        row = {"source1_entity_id": q, "country": country, "true_links": len(actual), "hits": len(hits),
               "candidates": len(retrieved), "ceiling": f05(actual, hits),
               "dense_unique_hits": len(actual & (dense - lexical)),
               "lexical_unique_hits": len(actual & (lexical - dense))}
        if missing_fields is not None:
            row["missing_name"] = bool(missing_fields[q]["missing_name"])
            row["missing_address"] = bool(missing_fields[q]["missing_address"])
        for source in ("S2", "S3"):
            row[f"{source}_truth"] = sum(i.startswith(source + "-") for i in actual)
            row[f"{source}_hits"] = sum(i.startswith(source + "-") for i in hits)
        for k in ks:
            row[f"hits_at_{k}"] = len(actual & set(part.loc[part["rank"].le(k), "candidate_entity_id"]))
        per_query.append(row)
        misses.extend({"source1_entity_id": q, "candidate_entity_id": i, "country": country,
                       "error_type": "retrieval_miss"} for i in sorted(actual - retrieved))
    frame = pd.DataFrame(per_query)
    groups = {"all": frame, "matched": frame[frame.true_links.gt(0)], "no_matches": frame[frame.true_links.eq(0)]}
    for country in ("India", "US"):
        groups[f"country={country}"] = frame[frame.country.eq(country)]
    for field in ("missing_name", "missing_address"):
        if field in frame:
            for value in (True, False):
                groups[f"{field}={value}"] = frame[frame[field].eq(value)]
    return ({"version": BLOCKING_VERSION, "budget": budget,
             "groups": {name: _blocking_summary(part, ks) for name, part in groups.items()}},
            frame, pd.DataFrame(misses, columns=["source1_entity_id", "candidate_entity_id", "country", "error_type"]))


def _blocking_summary(frame, ks):
    n, matched = len(frame), frame[frame.true_links.gt(0)]
    m, links, hits = len(matched), int(frame.true_links.sum()), int(frame.hits.sum())
    def recall(column):
        count = int(frame[column].sum())
        complete = int(matched[column].eq(matched.true_links).sum())
        return {"hits": count, "true_links": links, "matched_queries": m,
                "micro_recall": ratio(count, links),
                "macro_recall": float((matched[column] / matched.true_links).mean()) if m else None,
                "complete_queries": complete, "complete_recall": ratio(complete, m)}
    return {"queries": n, "matched_queries": m, "no_match_queries": n - m,
            "no_match_prevalence": ratio(n - m, n), "ceiling": float(frame.ceiling.mean()) if n else None,
            "matched_ceiling": float(matched.ceiling.mean()) if m else None,
            **recall("hits"), "any_match_queries": int(matched.hits.gt(0).sum()),
            "any_match_coverage": ratio(int(matched.hits.gt(0).sum()), m),
            "prefixes": {str(k): recall(f"hits_at_{k}") for k in ks},
            "sources": {s: {"hits": int(frame[f"{s}_hits"].sum()), "true_links": int(frame[f"{s}_truth"].sum()),
                              "micro_recall": ratio(int(frame[f"{s}_hits"].sum()), int(frame[f"{s}_truth"].sum()))}
                        for s in ("S2", "S3")},
            "marginal": {r: {"unique_hits": int(frame[f"{r}_unique_hits"].sum()), "true_links": links,
                              "recall_percentage_points": 100 * frame[f"{r}_unique_hits"].sum() / links if links else None}
                         for r in ("dense", "lexical")},
            "candidate_cost": {"total": int(frame.candidates.sum()),
                               "mean": float(frame.candidates.mean()) if n else None,
                               "median": float(frame.candidates.median()) if n else None,
                               "p95": float(frame.candidates.quantile(.95)) if n else None,
                               "max": int(frame.candidates.max()) if n else None,
                               "zero_queries": int(frame.candidates.eq(0).sum()),
                               "zero_rate": ratio(int(frame.candidates.eq(0).sum()), n)}}


def blocking_report(path, manifest, metrics, comparisons):
    """Quality is oracle coverage, not achieved matching accuracy."""
    lines = ["# Blocking experiment", "", f"Scope: {manifest['scope']}. Role: {manifest['purpose']}.",
             "", "Ceiling is official macro F0.5 for a perfect matcher on the saved union.",
             "No-match queries receive oracle 1 regardless of candidate count.", "",
             "| Group | Queries | Ceiling | Matched ceiling | Micro recall | Macro recall | Complete recall |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    def number(v):
        return "undefined" if v is None else f"{v:.6f}"
    for name, group in metrics["groups"].items():
        lines.append(f"| {name} | {group['queries']} | " + " | ".join(number(group[k]) for k in
                     ("ceiling", "matched_ceiling", "micro_recall", "macro_recall", "complete_recall")) + " |")
    lines += ["", "Prefixes use the saved fused ranking; they are not new route allocations.",
              "Destination-country validation is development feedback, not France performance.",
              "", "Paired comparisons (same queries/truth; budgets disclosed):", "```json",
              json.dumps(comparisons, indent=2), "```", "", "Execution statistics:", "```json",
              json.dumps(manifest["runtime"], indent=2), "```", ""]
    path.write_text("\n".join(lines))
