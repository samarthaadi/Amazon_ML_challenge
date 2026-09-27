"""CLI orchestration. Run from the project root: PYTHONPATH=src python -m er.run.

Blocking commands implement the approved TF-IDF/BGE-M3 plan. Matching and
decision methods remain TBD; baseline adapts the existing legacy method.
All development label access uses the frozen country/group permissions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import pandas as pd

from . import data, decide, evaluate, match, preprocess, retrieve


def code_identity():
    paths = [*sorted((data.ROOT / "src/er").glob("*.py")), data.ROOT / "pyproject.toml",
             data.ROOT / "uv.lock", data.ROOT / "eda/analysis/eda_first_pass.py",
             data.ROOT / "eda/analysis/full_pool_baseline.py"]
    return data.inventory(paths)


def new_run(root, name):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + name
    destination = root / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def prepare(cfg):
    if not np.isclose(sum(cfg[f"{x}_fraction"] for x in ("train", "validation", "holdout")), 1):
        raise ValueError("Split fractions must sum to one")
    root = data.output_root(cfg)
    destination = root / "splits" / cfg["split_version"]
    # Hash preservation inputs before any writes. Audit does not score holdout queries.
    protected = [*sorted((data.ROOT / "dataset").rglob("*.tsv")),
                 *sorted((data.ROOT / "eda").rglob("*"))]
    protected = [p for p in protected if p.is_file() and "__pycache__" not in p.parts]
    before = data.inventory(protected)
    historical = data.legacy_membership().to_dict("records")
    result = data.prepare_splits(destination, data.ROOT / "dataset/train/train_source1.tsv",
                                 data.ROOT / "dataset/train/train_ground_truth.tsv", cfg,
                                 historical, data.exposure_ids())
    after = data.inventory(protected)
    if before != after:
        raise ValueError("Raw or historical artifact changed during preparation")
    result["preserved_inputs"] = before
    result["historical_query_ids"] = sorted(r["entity_id"] for r in historical)
    data.write_json(destination / "manifest.json", result)
    print(f"Frozen splits: {destination}", flush=True)
    print(json.dumps(result["counts"], indent=2))


def provenance(split_manifest, index, *, legacy_cache):
    protected = split_manifest["preserved_inputs"]
    sources = {p: fp for p, fp in protected.items() if p.startswith("dataset/train/") and "ground_truth" not in p}
    result = {"inputs": sources, "pool": "all training Source 2/3 records",
              "preprocessing": "legacy_v1", "retrieval": "legacy_v1", "seed": 42,
              "retrieval_config": {"per_route_k": 100}, "index": index["signature"],
              "code": {path: identity for path, identity in code_identity().items()
                       if path in {"src/er/data.py", "src/er/preprocess.py", "src/er/retrieve.py",
                                   "eda/analysis/eda_first_pass.py", "eda/analysis/full_pool_baseline.py",
                                   "pyproject.toml", "uv.lock"}},
              "fitted_retrieval_state": None}
    if legacy_cache:
        result["imported_candidates"] = protected["eda/reports/full_pool/candidate_pairs.tsv.gz"]
        result["imported_tokens"] = protected["eda/reports/full_pool/query_tokens.tsv"]
    return result


def verify_inputs(split_manifest):
    # Full content fingerprints, not only mtimes. Cached identity must not silently
    # adopt changed raw records, legacy scores, or index contents.
    for relative, expected in split_manifest["preserved_inputs"].items():
        if relative.startswith("dataset/") or relative.startswith("eda/reports/full_pool/"):
            if data.fingerprint(data.ROOT / relative) != expected:
                raise ValueError(f"Input changed since split preparation: {relative}")


def load_split(cfg):
    directory = data.output_root(cfg) / "splits" / cfg["split_version"]
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["seed"] != cfg["seed"] or manifest["target_fractions"] != {
        name: cfg[f"{name}_fraction"] for name in ("train", "validation", "holdout")
    }:
        raise ValueError("Split configuration changed; select a new split version")
    for filename, key in (("membership.tsv", "membership"), ("membership.sqlite3", "database")):
        if data.fingerprint(directory / filename) != manifest[key]:
            raise ValueError("Frozen split artifact has changed")
    return directory, manifest


def baseline(cfg, setting):
    root = data.output_root(cfg)
    split_dir, split_manifest = load_split(cfg)
    verify_inputs(split_manifest)
    store = data.SplitStore(split_dir, cfg["settings"][setting])
    scope = split_manifest["historical_query_ids"]
    fitting, tuning, evaluation = (store.queries(role, scope) for role in ("fit", "tune", "evaluate"))
    if not fitting or not tuning or not evaluation:
        raise ValueError("Baseline needs nonempty fit/tune/evaluation groups")
    query_ids = sorted(set(fitting) | set(tuning) | set(evaluation))
    queries = data.query_records(query_ids)
    stage_config = {"method": "legacy_v1", "seed": cfg["seed"]}
    preprocessing = preprocess.fit(queries[queries.entity_id.isin(fitting)], stage_config)
    records = preprocess.transform(queries, preprocessing)
    index = retrieve.build_index(None, preprocessing, stage_config)
    cache, retrieval_runtime = retrieve.cached(records, index, root / "candidates",
                                               provenance(split_manifest, index, legacy_cache=True), reuse_legacy=True)
    candidates = pd.concat(retrieve.read_candidates(cache), ignore_index=True)
    tokens = pd.read_csv(cache / "tokens.tsv", sep="\t", keep_default_na=False)
    destination = new_run(root, setting)
    started = time.perf_counter()
    train_pairs = candidates[candidates.source1_entity_id.isin(fitting)].sort_values(data.PAIR_KEYS)
    state = match.fit(train_pairs, records, store.labels(fitting, "fit"), stage_config, tokens)
    fit_seconds = time.perf_counter() - started
    scores, scoring_runtime = match.score(candidates, records, state, tokens)
    scoring_runtime["scope"] = "all fit, tune, and evaluation candidates scored in this run"
    tune_scores = scores[scores.source1_entity_id.isin(tuning)]
    policy, trials = decide.tune(tune_scores, store.labels(tuning, "tune"), tuning, stage_config)
    eval_scores = scores[scores.source1_entity_id.isin(evaluation)]
    predictions = decide.apply(eval_scores, evaluation, policy)
    eval_candidates = candidates[candidates.source1_entity_id.isin(evaluation)]
    per_business, errors = evaluate.evaluate(evaluation, store.labels(evaluation, "evaluate"), eval_candidates, predictions)
    metrics = evaluate.summarize(per_business)
    with (destination / "matcher.pkl").open("wb") as stream:
        pickle.dump(state, stream)
    data.write_json(destination / "preprocessing.json", preprocessing)
    data.write_json(destination / "policy.json", policy)
    data.write_json(destination / "threshold_trials.json", trials)
    records.to_csv(destination / "derived_queries.tsv.gz", sep="\t", index=False, compression="gzip")
    scores.to_csv(destination / "scores.tsv.gz", sep="\t", index=False, compression="gzip")
    per_business.to_csv(destination / "per_business.tsv", sep="\t", index=False)
    errors.to_csv(destination / "errors.tsv.gz", sep="\t", index=False, compression="gzip")
    export(destination, evaluation, eval_candidates, predictions)
    data.write_json(destination / "metrics.json", metrics)
    manifest = {
        "status": "development feedback", "setting": setting, "query_scope": "frozen legacy cohort within full-population splits",
        "configuration": cfg, "stage_config": stage_config, "code": code_identity(),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "sklearn": __import__("sklearn").__version__},
        "split_manifest": data.fingerprint(split_dir / "manifest.json"),
        "candidate_cache": str(cache.relative_to(root)), "candidate_cache_id": retrieval_runtime["cache_id"],
        "fit_queries": sorted(fitting), "tune_queries": sorted(tuning), "evaluation_queries": sorted(evaluation),
        "seed": cfg["seed"], "changed_component": "protocol: old evaluated test fold becomes development validation",
        "runtime": {"retrieval": retrieval_runtime, "matcher_fit_seconds": fit_seconds,
                    "matcher_inference": scoring_runtime},
        "historical_baseline": json.loads((data.ROOT / "eda/reports/full_pool/metrics.json").read_text())["model"]["splits"],
        "state_artifacts": {name: data.fingerprint(destination / name)
                            for name in ("matcher.pkl", "preprocessing.json", "policy.json", "derived_queries.tsv.gz")},
    }
    data.write_json(destination / "manifest.json", manifest)
    evaluate.report(destination / "report.md", manifest, metrics)
    store.close()
    print(f"Development run: {destination}")
    print(json.dumps(metrics["all"], indent=2))


def export(destination, query_ids, candidates, predictions, *, append=False):
    import csv
    candidate_map = candidates.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    for filename, column, values in (("predictions.tsv", "matched_entity_ids", predictions),
                                     ("candidate_pairs.tsv", "candidate_entity_ids", candidate_map)):
        with (destination / filename).open("a" if append else "w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            if not append:
                writer.writerow(["source1_entity_id", column])
            for q in query_ids:
                if not predictions[q] <= candidate_map.get(q, set()):
                    raise ValueError("Accepted ID absent from candidates")
                writer.writerow([q, ",".join(sorted(values.get(q, set())))])


def audit(cfg):
    """Reproduce historical scorer and baseline adapter without touching EDA outputs."""
    directory = new_run(data.output_root(cfg), "legacy-audit")
    anchors = pd.read_csv(data.ROOT / "eda/reports/eda_data/anchors.tsv", sep="\t", keep_default_na=False)
    pairs = pd.read_csv(data.ROOT / "eda/reports/full_pool/candidate_pairs.tsv.gz", sep="\t", keep_default_na=False)
    tokens = pd.read_csv(data.ROOT / "eda/reports/full_pool/query_tokens.tsv", sep="\t", keep_default_na=False)
    splits = data.legacy_membership()
    truth = {q: set() for q in anchors.entity_id}
    for row in data.rows(data.ROOT / "eda/reports/eda_data/truth.tsv"):
        truth[row["source1_entity_id"]].add(row["candidate_entity_id"])
    fitting = splits.loc[splits.split.eq("train"), "entity_id"].tolist()
    tuning = splits.loc[splits.split.eq("validation"), "entity_id"].tolist()
    candidates = pairs[retrieve.LEGACY_COLUMNS].copy()
    for col in ("name_rank", "address_rank", "name_bm25", "address_bm25"):
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    state = match.fit(candidates[candidates.source1_entity_id.isin(fitting)], anchors,
                      {q: truth[q] for q in fitting}, {"method": "legacy_v1", "seed": 42}, tokens)
    scores, runtime = match.score(candidates, anchors, state, tokens)
    np.testing.assert_allclose(scores.score, pairs.probability.astype(float), rtol=1e-9, atol=1e-10)
    policy, _ = decide.tune(scores[scores.source1_entity_id.isin(tuning)], {q: truth[q] for q in tuning},
                            tuning, {"method": "legacy_v1"})
    saved = json.loads((data.ROOT / "eda/reports/full_pool/metrics.json").read_text())
    if not np.isclose(policy["threshold"], saved["model"]["selected_threshold"]):
        raise ValueError("Legacy threshold changed")
    checks = {}
    for split in ("train", "validation", "test"):
        ids = splits.loc[splits.split.eq(split), "entity_id"].tolist()
        predictions = decide.apply(scores[scores.source1_entity_id.isin(ids)], ids, policy)
        score = evaluate.macro_f05(ids, {q: truth[q] for q in ids}, predictions)
        expected = saved["model"]["splits"][split]["macro_f0_5"]
        if not np.isclose(score, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Legacy {split} score changed")
        checks[split] = {"expected": expected, "reproduced": score}
    data.write_json(directory / "audit.json", {"legacy_scores": checks, "policy": policy,
                    "all_pair_scores_reproduced": True, "runtime": runtime,
                    "note": "Historical test fold is exposed development data, not the new final holdout."})
    print(json.dumps(checks, indent=2))
    print(f"Legacy audit: {directory}")


def freeze(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("kind") == "blocking":
        return blocking_freeze(run_dir, manifest)
    if manifest["status"] != "development feedback" or manifest["code"] != code_identity():
        raise ValueError("Freeze requires a completed development run using current code")
    split_dir, _ = load_split(manifest["configuration"])
    if list(split_dir.glob("final_access_*.json")):
        raise ValueError("Holdout already accessed; cannot freeze a new design for an untouched final check")
    filenames = ("matcher.pkl", "preprocessing.json", "policy.json", "manifest.json")
    frozen = {"files": {n: data.fingerprint(run_dir / n) for n in filenames}, "code": code_identity(),
              "created_at": datetime.now(timezone.utc).isoformat()}
    # Exclusive creation: freezing cannot silently replace an earlier decision.
    with (run_dir / "frozen.json").open("x") as stream:
        json.dump(frozen, stream, indent=2)
    print(f"Frozen for explicit final evaluation: {run_dir}")


def final(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("kind") == "blocking":
        return blocking_final(run_dir, manifest)
    frozen = json.loads((run_dir / "frozen.json").read_text())
    if frozen["code"] != code_identity():
        raise ValueError("Code changed after freezing")
    for name, expected in frozen["files"].items():
        if data.fingerprint(run_dir / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg, setting = manifest["configuration"], manifest["setting"]
    split_dir, split_manifest = load_split(cfg)
    if data.fingerprint(split_dir / "manifest.json") != manifest["split_manifest"]:
        raise ValueError("Split manifest changed after fitting")
    verify_inputs(split_manifest)
    root = data.output_root(cfg)
    # Ledger created before label access; later runs cannot claim an untouched holdout.
    marker = split_dir / f"final_access_{setting}.json"
    prior_access = [json.loads(p.read_text()) for p in sorted(split_dir.glob("final_access_*.json"))]
    if any(entry["started_at"] < frozen["created_at"] for entry in prior_access):
        raise ValueError("This design was frozen after holdout results became available")
    with marker.open("x") as stream:
        json.dump({"frozen_run": str(run_dir), "started_at": datetime.now(timezone.utc).isoformat()}, stream)
    store = data.SplitStore(split_dir, cfg["settings"][setting], final=True)
    queries = store.queries("evaluate")
    records = data.query_records(queries)
    preprocessing = json.loads((run_dir / "preprocessing.json").read_text())
    records = preprocess.transform(records, preprocessing)
    index = retrieve.build_index(None, preprocessing, {"method": "legacy_v1"})
    cache, retrieval_runtime = retrieve.cached(records, index, root / "candidates",
                                               provenance(split_manifest, index, legacy_cache=False))
    # Only unpickle our local, hash-verified frozen artifact.
    with (run_dir / "matcher.pkl").open("rb") as stream:
        state = pickle.load(stream)
    policy = json.loads((run_dir / "policy.json").read_text())
    tokens = pd.read_csv(cache / "tokens.tsv", sep="\t", keep_default_na=False)
    destination = new_run(root, setting + "-final")
    parts, seen, carry = [], set(), None
    elapsed, pair_count, first = 0.0, 0, True

    def process(candidates, ids):
        nonlocal elapsed, pair_count, first
        subset = records[records.entity_id.isin(ids)]
        scores, runtime = match.score(candidates, subset, state, tokens[tokens.source1_entity_id.isin(ids)])
        predictions = decide.apply(scores, ids, policy)
        per_query, errors = evaluate.evaluate({q: queries[q] for q in ids}, store.labels(ids, "evaluate"), candidates, predictions)
        parts.append(per_query)
        for name, frame in (("scores.tsv.gz", scores), ("errors.tsv.gz", errors)):
            frame.to_csv(destination / name, sep="\t", index=False, mode="w" if first else "a",
                         header=first, compression="gzip")
        export(destination, ids, candidates, predictions, append=not first)
        first = False
        elapsed += runtime["seconds"]
        pair_count += runtime["pairs"]
        seen.update(ids)

    for chunk in retrieve.read_candidates(cache):
        chunk = pd.concat([carry, chunk], ignore_index=True) if carry is not None else chunk
        if chunk.empty:
            continue
        last_id = chunk.source1_entity_id.iloc[-1]
        carry = chunk[chunk.source1_entity_id.eq(last_id)]
        complete = chunk[~chunk.source1_entity_id.eq(last_id)]
        if len(complete):
            process(complete, complete.source1_entity_id.drop_duplicates().tolist())
    if carry is not None and len(carry):
        process(carry, carry.source1_entity_id.drop_duplicates().tolist())
    missing = sorted(set(queries) - seen)
    if missing:
        process(pd.DataFrame(columns=[*retrieve.LEGACY_COLUMNS, "rank"]), missing)
    per_business = pd.concat(parts, ignore_index=True)
    metrics = evaluate.summarize(per_business)
    per_business.to_csv(destination / "per_business.tsv", sep="\t", index=False)
    manifest = {**manifest, "status": "final evaluation; holdout now exposed", "query_scope": "all eligible final holdout businesses",
                "frozen_run": str(run_dir), "evaluation_queries_hash": data.digest(sorted(queries)),
                "evaluation_queries": sorted(queries), "candidate_cache": str(cache.relative_to(root)),
                "candidate_cache_id": retrieval_runtime["cache_id"],
                "prior_final_access": prior_access,
                "runtime": {"retrieval": retrieval_runtime, "matcher_inference": {"seconds": elapsed, "pairs": pair_count}}}
    data.write_json(destination / "manifest.json", manifest)
    data.write_json(destination / "metrics.json", metrics)
    evaluate.report(destination / "report.md", manifest, metrics)
    store.close()
    print(f"Final evaluation: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=data.ROOT / "configs/experiments.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Freeze full-population splits and preserve input fingerprints")
    commands.add_parser("audit-legacy", help="Reproduce historical baseline without rewriting it")
    commands.add_parser("blocking-freeze-evaluator", help="Test and snapshot the independent blocking evaluator")
    for name in ("blocking-preflight", "blocking-build"):
        sub = commands.add_parser(name)
        sub.add_argument("--branch", choices=["lexical", "dense"], default="lexical")
    sub = commands.add_parser("blocking-retrieve")
    sub.add_argument("--setting", choices=["mixed", "india_to_us", "us_to_india"], default="mixed")
    sub.add_argument("--purpose", choices=["tune", "evaluate"], default="tune")
    sub.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="lexical")
    sub = commands.add_parser("blocking-evaluate")
    sub.add_argument("--run", required=True, type=Path)
    sub.add_argument("--compare", action="append", default=[], type=Path)
    baseline_parser = commands.add_parser("baseline", help="Run existing baseline on frozen legacy development queries")
    baseline_parser.add_argument("--setting", choices=["mixed", "india_to_us", "us_to_india"], default="mixed")
    for name in ("freeze", "final"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    cfg = data.config(args.config)
    if args.command == "blocking-freeze-evaluator":
        blocking_evaluator(data.output_root(cfg), create=True)
    elif args.command == "blocking-preflight":
        blocking_preflight(cfg, args.branch)
    elif args.command == "blocking-build":
        blocking_build(cfg, args.branch)
    elif args.command == "blocking-retrieve":
        blocking_retrieve(cfg, args.setting, args.purpose, args.mode)
    elif args.command == "blocking-evaluate":
        with retrieve.ResourceWatch(cfg["blocking"]):
            blocking_evaluate(args.run, args.compare)
    elif args.command == "prepare":
        prepare(cfg)
    elif args.command == "baseline":
        baseline(cfg, args.setting)
    elif args.command == "audit-legacy":
        audit(cfg)
    elif args.command == "freeze":
        freeze(args.run)
    elif args.command == "final":
        final(args.run)



def blocking_evaluator(root, *, create=False):
    """Verify the separately frozen evaluator and only its metric fixture class."""
    import ast
    fixture_path = data.ROOT / "tests/test_protocol.py"
    source = fixture_path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "BlockingMetricTests")
    fixture = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
    directory = root / "evaluators" / evaluate.BLOCKING_VERSION
    if create:
        import subprocess
        subprocess.run([__import__("sys").executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_protocol.py"], check=True)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "evaluate.py").write_bytes((data.ROOT / "src/er/evaluate.py").read_bytes())
        (directory / "metric_fixtures.py").write_text(fixture)
        data.write_json(directory / "manifest.json", {"version": evaluate.BLOCKING_VERSION,
                        "evaluator": data.fingerprint(directory / "evaluate.py"), "fixtures": data.fingerprint(directory / "metric_fixtures.py")})
    manifest = json.loads((directory / "manifest.json").read_text())
    import hashlib
    expected_fixture = {"bytes": len(fixture.encode()), "sha256": hashlib.sha256(fixture.encode()).hexdigest()}
    if manifest["evaluator"] != data.fingerprint(data.ROOT / "src/er/evaluate.py") or manifest["fixtures"] != expected_fixture:
        raise ValueError("Frozen evaluator/fixtures changed: separately review, version, and reevaluate compared runs")
    return manifest


def blocking_context(cfg):
    root = data.output_root(cfg)
    blocking_evaluator(root)
    split_dir, split = load_split(cfg)
    verify_inputs(split)
    with data.pool_connection() as db:
        # Existing record IDs are globally unique by a UNIQUE index and source-qualified.
        indexes = db.execute("PRAGMA index_list(records)").fetchall()
        if not any(row[1] == "records_entity_id" and row[2] == 1 for row in indexes):
            raise ValueError("Full-pool record ID uniqueness is not established")
        count = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    pool = {p: fp for p, fp in split["preserved_inputs"].items()
            if p in {"dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv",
                     "eda/reports/full_pool/candidate_index.sqlite3"}}
    identity = retrieve.blocking_identity(cfg["blocking"], {"files": pool, "count": count,
                                                          "id_mapping": "source-qualified ID; read-only records.rowid"})
    return root, split_dir, split, identity


def blocking_queries(cfg, split_dir, split, setting, purpose):
    if cfg["blocking"]["query_subset"] != "historical-development":
        raise ValueError("Unknown fixed development query subset")
    store = data.SplitStore(split_dir, cfg["settings"][setting])
    queries = store.queries(purpose, split["historical_query_ids"])
    selected = {}
    for country in sorted(set(queries.values())):
        ids = sorted((q for q, c in queries.items() if c == country), key=lambda q: data.digest([cfg["seed"], q]))
        selected.update((q, country) for q in ids[:cfg["blocking"]["queries_per_country"]])
    store.close()
    if not selected:
        raise ValueError("Empty authorized query subset")
    return selected


def blocking_preflight(cfg, branch):
    root, split_dir, split, identity = blocking_context(cfg)
    config = cfg["blocking"]
    pilot_identity = {**identity, "scope": "reduced-pool engineering pilot", "pilot_rows": config["pilot_rows"]}
    destination = root / "indexes" / data.digest(pilot_identity)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"preflight-{branch}.json"
    if report_path.exists():
        print(f"Completed preflight: {report_path}")
        return
    with retrieve.ResourceWatch(config) as resources:
        lexical_dir = destination / "lexical"
        lexical = retrieve.lexical_build(lexical_dir, data.pool_records(config["shard_rows"], pilot_rows=config["pilot_rows"]),
                                         config, pilot_identity)
        queries = blocking_queries(cfg, split_dir, split, "mixed", "tune")
        records = preprocess.blocking_records(data.query_records(queries))
        # Score verification is also covered by a brute-force unit fixture before this pilot.
        started = time.perf_counter()
        retrieve.lexical_search(lexical_dir, records.iloc[:config["query_batch_size"]], config["allocations"]["lexical"], config, lexical)
        search_seconds = time.perf_counter() - started
        report = {"identity": identity, "scope": pilot_identity["scope"], "pilot_pool_records": lexical["pool_records"],
                  "full_pool_records": identity["pool"]["count"], "lexical_manifest": data.fingerprint(lexical_dir / "manifest.json"),
                  "lexical_runtime": lexical["runtime"], "lexical_batch_seconds": search_seconds}
        if branch == "dense":
            model = retrieve.load_encoder(config, root / "models")
            texts = []
            for name in lexical["shards"]:
                frame = pd.read_csv(lexical_dir / name / "records.tsv.gz", sep="\t", keep_default_na=False)
                texts.extend(frame.loc[frame.dense_text.ne(""), "dense_text"])
            lengths = []
            for start in range(0, len(texts), config["encoder"]["batch_size"]):
                lengths.extend(map(len, model.tokenizer(texts[start:start + config["encoder"]["batch_size"]],
                                                      truncation=False, padding=False, verbose=False)["input_ids"]))
            p99 = float(np.quantile(lengths, .99))
            model.max_seq_length = min(config["encoder"]["max_length_cap"], max(32, int(np.ceil(p99 / 32) * 32)))
            dense_dir = destination / "dense"
            dense = retrieve.dense_build(dense_dir, lexical_dir, lexical, config, model, pilot_identity, pilot=True)
            usable = records[records.dense_text.ne("")]
            vectors, runtime = retrieve.encode_texts(model, usable.dense_text.tolist(), config)
            fidelity = retrieve.dense_fidelity(dense_dir, dense, vectors, config)
            report.update({"max_length": model.max_seq_length, "observed_token_p99": p99,
                           "dense_manifest": data.fingerprint(dense_dir / "manifest.json"),
                           "dense_runtime": dense["runtime"], "query_encoding": runtime, "fidelity": fidelity})
        size = retrieve.disk_check(destination, config)
        scale = identity["pool"]["count"] / lexical["pool_records"]
        estimate = int(size * scale)
        report.update({"disk_bytes": size, "projected_full_disk_bytes": estimate,
                       "projection_caveat": "linear pilot extrapolation; vocabulary/index growth may differ; runtime guard remains active",
                       "projected_lexical_build_hours": lexical["runtime"]["index_build_seconds"] * scale / 3600,
                       "runtime": resources.stats()})
        if branch == "dense":
            report["projected_dense_encoding_hours"] = report["dense_runtime"]["encoding_seconds"] * scale / 3600
        disk_ok = estimate < config["max_disk_gib"] * 1024 ** 3
        fidelity_ok = branch != "dense" or report["fidelity"]["ann_exact_recall_at_k"] >= config["ann"]["minimum_pilot_recall"]
        report["approved_for_full_build"] = disk_ok and fidelity_ok
        retrieve.atomic_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"Preflight: {report_path}")


def preflight_for(root, identity, branch):
    pilot_identity = {**identity, "scope": "reduced-pool engineering pilot", "pilot_rows": identity["config"]["pilot_rows"]}
    directory = root / "indexes" / data.digest(pilot_identity)
    value = json.loads((directory / f"preflight-{branch}.json").read_text())
    if value["identity"] != identity or not value["approved_for_full_build"]:
        raise ValueError("Full build requires a passing preflight for these exact inputs/configuration")
    retrieve.verified_artifact(directory / "lexical")
    if branch == "dense":
        retrieve.verified_artifact(directory / "dense")
    return value


def blocking_build(cfg, branch):
    root, _, _, identity = blocking_context(cfg)
    preflight = preflight_for(root, identity, branch)
    config = cfg["blocking"]
    destination = root / "indexes" / data.digest(identity)
    destination.mkdir(parents=True, exist_ok=True)
    retrieve.disk_check(destination, config, preflight["projected_full_disk_bytes"])
    with retrieve.ResourceWatch(config) as resources:
        lexical_dir = destination / "lexical"
        lexical = retrieve.lexical_build(lexical_dir, data.pool_records(config["shard_rows"]), config, identity)
        if lexical["pool_records"] != identity["pool"]["count"]:
            raise ValueError("Index does not cover the complete eligible pool")
        if branch == "dense":
            model = retrieve.load_encoder(config, root / "models", preflight["max_length"])
            retrieve.dense_build(destination / "dense", lexical_dir, lexical, config, model, identity)
        resources_path = destination / f"resources-{branch}.json"
        if not resources_path.exists():
            data.write_json(resources_path, {**resources.stats(), "disk_bytes": retrieve.disk_check(destination, config),
                                            "build_timing_scope": "successful invocation; earlier interrupted attempts are excluded"})
    print(f"Full-pool {branch} index: {destination}")


def blocking_retrieve(cfg, setting, purpose, mode):
    root, split_dir, split, identity = blocking_context(cfg)
    queries = blocking_queries(cfg, split_dir, split, setting, purpose)
    records = data.query_records(queries)
    indexes = root / "indexes" / data.digest(identity)
    cache_id = data.digest({"identity": identity, "queries": sorted(queries), "mode": mode})
    destination = root / "candidates" / ("blocking-" + cache_id)
    config = cfg["blocking"]
    with retrieve.ResourceWatch(config) as resources:
        model = None
        if config["allocations"][mode]["dense"]:
            dense = retrieve.verified_artifact(indexes / "dense")
            model = retrieve.load_encoder(config, root / "models", dense["max_length"])
        manifest, cache_access = retrieve.blocking_candidates(destination, records, indexes / "lexical", indexes / "dense", config, mode, identity, model)
        run_dir = new_run(root, "blocking-" + setting + "-" + mode)
        data.write_json(run_dir / "manifest.json", {
            "kind": "blocking", "status": "candidates complete; not evaluated", "setting": setting, "purpose": purpose,
            "scope": "full eligible Source 2/3 pool; fixed development queries", "configuration": cfg,
            "mode": mode, "candidate_cache": str(destination), "candidate_manifest": data.fingerprint(destination / "manifest.json"),
            "indexes": str(indexes), "identity": identity, "queries": queries, "query_list_hash": data.digest(sorted(queries)),
            "split_manifest": data.fingerprint(split_dir / "manifest.json"), "truth_version": split["inputs" ] if "inputs" in split else split["preserved_inputs"]["dataset/train/train_ground_truth.tsv"],
            "evaluator": blocking_evaluator(root), "code": code_identity(),
            "runtime": {**manifest["runtime"], **resources.stats(), "cache_access": cache_access,
                        "retrieval_timing_scope": "original candidate generation; cache access measured separately",
                        "candidate_disk_bytes": retrieve.disk_check(destination, config),
                        "index_build": {b: json.loads((indexes / f"resources-{b}.json").read_text()) for b in ("lexical", "dense") if (indexes / f"resources-{b}.json").exists()}}})
    print(f"Candidate run: {run_dir}")
    return run_dir


def read_blocking_pairs(path):
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    for col in ("rank", "fusion_score", *[f"{r}_{s}" for r in retrieve.BLOCKING_ROUTES for s in ("rank", "score")]):
        frame[col] = pd.to_numeric(frame[col].replace("", np.nan), errors="raise")
    for route in retrieve.BLOCKING_ROUTES:
        frame[f"in_{route}"] = frame[f"in_{route}"].map(lambda x: x if isinstance(x, bool) else {"True": True, "False": False}[x])
    return frame


def blocking_evaluate(run_dir, compare=(), *, final_store=None):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg, config = manifest["configuration"], manifest["configuration"]["blocking"]
    root = data.output_root(cfg)
    if (run_dir / "blocking_metrics.json").exists():
        raise ValueError("Completed reports are immutable; retrieve into a new run to reevaluate")
    if manifest["evaluator"] != blocking_evaluator(root):
        raise ValueError("Evaluator version changed")
    split_dir, split = load_split(cfg)
    verify_inputs(split)
    if data.fingerprint(split_dir / "manifest.json") != manifest["split_manifest"]:
        raise ValueError("Split/truth version changed")
    store = final_store or data.SplitStore(split_dir, cfg["settings"][manifest["setting"]])
    queries, purpose = manifest["queries"], manifest["purpose"]
    if set(store.queries(purpose, queries)) != set(queries):
        raise ValueError("Run queries are outside the profile's authorized group")
    cache = Path(manifest["candidate_cache"])
    if data.fingerprint(cache / "manifest.json") != manifest["candidate_manifest"]:
        raise ValueError("Candidate completion manifest changed")
    started = time.perf_counter()
    candidate_manifest = retrieve.verified_artifact(cache)
    cache_read_seconds = time.perf_counter() - started
    if set(candidate_manifest["completed_queries"]) != set(queries):
        raise ValueError("Missing completed queries")
    parts, first, all_truth = [], True, {}
    # Query-complete batches keep evaluation memory bounded for the explicit final command.
    metadata = pd.read_csv(cache / "queries.tsv.gz", sep="\t", keep_default_na=False)
    missing = metadata.set_index("entity_id")[["missing_name", "missing_address"]].to_dict("index")
    for name in candidate_manifest["shards"]:
        batch_dir = cache / name
        batch_manifest = retrieve.verified_artifact(batch_dir)
        ids = batch_manifest["completed_queries"]
        read_started = time.perf_counter()
        pairs = read_blocking_pairs(batch_dir / "pairs.tsv.gz")
        cache_read_seconds += time.perf_counter() - read_started
        subset = {q: queries[q] for q in ids}
        truth = store.labels(ids, purpose)
        all_truth.update({q: data.digest(sorted(t)) for q, t in truth.items()})
        _, per, misses = evaluate.blocking(subset, truth, pairs, completed_queries=ids, known_ids=data.pool_has_ids,
                                           budget=config["budget"], route_budgets=config["allocations"][manifest["mode"]],
                                           missing_fields={q: missing[q] for q in ids})
        parts.append(per)
        misses.to_csv(run_dir / "missed_matches.tsv.gz", sep="\t", index=False, mode="w" if first else "a", header=first, compression="gzip")
        first = False
    frame = pd.concat(parts, ignore_index=True)
    if frame.source1_entity_id.duplicated().any() or set(frame.source1_entity_id) != set(queries):
        raise ValueError("Duplicate/incomplete evaluated query coverage")
    # The frozen evaluator owns every metric calculation; this only selects groups.
    groups = {"all": frame, "matched": frame[frame.true_links.gt(0)], "no_matches": frame[frame.true_links.eq(0)]}
    groups.update({f"country={c}": frame[frame.country.eq(c)] for c in ("India", "US")})
    groups.update({f"{field}={flag}": frame[frame[field].eq(flag)] for field in ("missing_name", "missing_address") for flag in (True, False)})
    metrics = {"version": evaluate.BLOCKING_VERSION, "budget": config["budget"],
               "groups": {name: evaluate._blocking_summary(part, (10, 25, 50, 100)) for name, part in groups.items()}}
    frame.to_csv(run_dir / "per_query.tsv", sep="\t", index=False)
    manifest["truth_hash"] = data.digest(all_truth)
    comparisons = []
    for other_dir in compare:
        other = json.loads((Path(other_dir) / "manifest.json").read_text())
        for key in ("query_list_hash", "truth_hash", "split_manifest", "evaluator", "scope"):
            if manifest[key] != other[key]:
                raise ValueError("Comparison requires identical queries, truth, evaluator and pool scope")
        if manifest["identity"]["pool"] != other["identity"]["pool"]:
            raise ValueError("Comparison requires the identical searchable pool version")
        other_metrics = json.loads((Path(other_dir) / "blocking_metrics.json").read_text())
        if other_metrics["budget"] != metrics["budget"]:
            raise ValueError("Different budgets need an explicit quality/cost curve")
        prior = pd.read_csv(Path(other_dir) / "per_query.tsv", sep="\t").set_index("source1_entity_id")
        differences = frame.set_index("source1_entity_id").ceiling - prior.ceiling
        comparisons.append({"run": str(other_dir), "mode": other["mode"], "queries": len(differences),
                            "paired_ceiling_difference": float(differences.mean()),
                            "improved": int(differences.gt(1e-12).sum()), "worsened": int(differences.lt(-1e-12).sum())})
    # Existing BM25 cache is a named historical comparator, disclosed at its saved rank prefix B.
    if final_store is None:
        legacy_pairs = []
        for chunk in pd.read_csv(data.ROOT / "eda/reports/full_pool/candidate_pairs.tsv.gz", sep="\t", chunksize=50000, keep_default_na=False,
                                 usecols=data.PAIR_KEYS + ["name_rank", "address_rank"]):
            part = chunk[chunk.source1_entity_id.isin(queries)].copy()
            for col in ("name_rank", "address_rank"):
                part[col] = pd.to_numeric(part[col], errors="coerce")
            legacy_pairs.append(part)
        pairs = retrieve._rank(pd.concat(legacy_pairs, ignore_index=True))
        pairs = pairs[pairs["rank"].le(config["budget"])][data.PAIR_KEYS + ["rank"]]
        old, old_per, _ = evaluate.blocking(queries, store.labels(queries, purpose), pairs, completed_queries=list(queries),
                                           known_ids=data.pool_has_ids, budget=config["budget"])
        delta = frame.set_index("source1_entity_id").ceiling - old_per.set_index("source1_entity_id").ceiling
        comparisons.append({"mode": "legacy BM25 saved deterministic prefix (original union up to 200)", "budget": config["budget"],
                            "ceiling": old["groups"]["all"]["ceiling"], "paired_ceiling_difference": float(delta.mean()),
                            "note": "Saved min-route-rank prefix, not a BM25 rerun with changed route allocation"})
    manifest["runtime"]["evaluation_and_cache_read_seconds"] = time.perf_counter() - started
    manifest["runtime"]["cache_read_seconds"] = cache_read_seconds
    manifest["status"] = "final evaluation; holdout now exposed" if final_store else "development feedback"
    manifest["selection_eligible"] = final_store is None and purpose == "tune"
    data.write_json(run_dir / "blocking_metrics.json", metrics)
    data.write_json(run_dir / "comparisons.json", comparisons)
    data.write_json(run_dir / "manifest.json", manifest)
    evaluate.blocking_report(run_dir / "report.md", manifest, metrics, comparisons)
    if final_store is None:
        store.close()
    print(f"Blocking evaluation: {run_dir / 'report.md'}")
    print(json.dumps(metrics["groups"]["all"], indent=2))


def blocking_freeze(run_dir, manifest):
    cfg = manifest["configuration"]
    blocking_evaluator(data.output_root(cfg))
    if manifest["status"] != "development feedback" or not manifest.get("selection_eligible"):
        raise ValueError("Freeze blocking settings using the permitted tuning group only")
    if manifest["code"] != code_identity():
        raise ValueError("Code changed since candidate generation")
    split_dir, _ = load_split(cfg)
    if list(split_dir.glob("final_access_*.json")):
        raise ValueError("Holdout already accessed; cannot freeze a new design")
    retrieve.verified_artifact(Path(manifest["candidate_cache"]))
    names = ("manifest.json", "blocking_metrics.json", "comparisons.json", "per_query.tsv")
    with (run_dir / "frozen.json").open("x") as stream:
        json.dump({"files": {name: data.fingerprint(run_dir / name) for name in names}, "code": code_identity(),
                   "created_at": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
    print(f"Frozen blocking configuration: {run_dir}")


def blocking_final(run_dir, manifest):
    frozen = json.loads((run_dir / "frozen.json").read_text())
    if frozen["code"] != code_identity():
        raise ValueError("Code changed after freezing")
    for name, expected in frozen["files"].items():
        if data.fingerprint(run_dir / name) != expected:
            raise ValueError("Frozen blocking artifact changed")
    cfg, setting = manifest["configuration"], manifest["setting"]
    root, split_dir, split, identity = blocking_context(cfg)
    if identity != manifest["identity"] or data.fingerprint(split_dir / "manifest.json") != manifest["split_manifest"]:
        raise ValueError("Frozen inputs changed")
    prior = [json.loads(p.read_text()) for p in split_dir.glob("final_access_*.json")]
    if any(entry["started_at"] < frozen["created_at"] for entry in prior):
        raise ValueError("Configuration frozen after earlier holdout access")
    with (split_dir / f"final_access_{setting}.json").open("x") as stream:
        json.dump({"frozen_run": str(run_dir), "started_at": datetime.now(timezone.utc).isoformat()}, stream)
    store = data.SplitStore(split_dir, cfg["settings"][setting], final=True)
    queries = store.queries("evaluate")
    config, mode = cfg["blocking"], manifest["mode"]
    indexes = Path(manifest["indexes"])
    cache = root / "candidates" / ("blocking-final-" + data.digest({"identity": identity, "queries": sorted(queries), "mode": mode}))
    with retrieve.ResourceWatch(config) as resources:
        model = None
        if config["allocations"][mode]["dense"]:
            dense = retrieve.verified_artifact(indexes / "dense")
            model = retrieve.load_encoder(config, root / "models", dense["max_length"])
        candidate_manifest, _ = retrieve.blocking_candidates(cache, data.query_records(queries), indexes / "lexical", indexes / "dense",
                                                             config, mode, identity, model)
        destination = new_run(root, "blocking-" + setting + "-final")
        final_manifest = {**manifest, "queries": queries, "query_list_hash": data.digest(sorted(queries)), "purpose": "evaluate",
                          "scope": "full eligible Source 2/3 pool; all eligible final holdout queries", "candidate_cache": str(cache),
                          "candidate_manifest": data.fingerprint(cache / "manifest.json"), "frozen_run": str(run_dir),
                          "prior_final_access": prior, "runtime": {**candidate_manifest["runtime"], **resources.stats()},
                          "status": "candidates complete; not evaluated", "selection_eligible": False}
        data.write_json(destination / "manifest.json", final_manifest)
        blocking_evaluate(destination, final_store=store)
    store.close()


if __name__ == "__main__":
    main()
