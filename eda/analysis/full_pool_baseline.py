"""Full-pool candidate retrieval and a small entity-resolution baseline.

Run from the repository root:
    uv run python eda/analysis/full_pool_baseline.py

The raw TSVs are read only. A reusable SQLite FTS index and all derived outputs
are written under eda/reports/full_pool/.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import threading
import unicodedata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/amazon-er-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from eda_first_pass import (
    CHUNK,
    LEGAL,
    ROOT,
    EDA_ROOT,
    STOP_ADDRESS,
    STOP_NAME,
    BASELINE_NORMALIZATION,
    TARGETED_NORMALIZATION,
    CLEANUP_ABLATIONS,
    clean,
    clean_targeted,
    core_name,
    expanded,
    expanded_targeted,
    number_status,
    read_tsv,
)

SEED = 42
OUT = EDA_ROOT / "reports/full_pool"
EDA_CACHE = EDA_ROOT / "reports/eda_data"
DB_PATH = OUT / "candidate_index.sqlite3"
INDEX_VERSION = 2
TOP_KS = (10, 25, 50, 100)
FINAL_ROUTE_K = 100
MAX_QUERY_TOKENS = 6
EXPECTED_ROWS = 10_320_219
SPLITS = {0: "test", 1: "validation", 2: "train", 3: "train", 4: "train"}

FEATURES = [
    "name_similarity",
    "expanded_name_similarity",
    "address_similarity",
    "name_token_jaccard",
    "address_token_jaccard",
    "distinctive_name_coverage",
    "distinctive_address_coverage",
    "exact_normalized_name",
    "exact_expanded_name",
    "exact_normalized_address",
    "numeric_same",
    "numeric_partial",
    "numeric_disjoint",
    "numeric_one_missing",
    "numeric_neither",
    "missing_candidate_address",
    "name_length_ratio",
    "address_length_ratio",
    "script_overlap",
    "script_exact",
    "cross_script",
]


def source_fingerprints():
    result = {}
    for source in (2, 3):
        path = ROOT / f"dataset/train/train_source{source}.tsv"
        stat = path.stat()
        result[str(path.relative_to(ROOT))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def index_signature():
    return {"version": INDEX_VERSION, "sources": source_fingerprints(), "expected_rows": EXPECTED_ROWS}


def validate_raw_inputs():
    inventory = json.loads((EDA_ROOT / "analysis/initial_inventory.json").read_text())
    changed = []
    for entry in inventory["sources"] + [inventory["ground_truth"]]:
        path = ROOT / entry["file"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
            changed.append(entry["file"])
    if changed:
        raise RuntimeError(f"Raw dataset fingerprints changed: {changed}")


def index_valid():
    if not DB_PATH.exists():
        return False
    try:
        connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        stored_signature = json.loads(connection.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()[0])
        count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
        connection.close()
        return stored_signature == index_signature() and count == EXPECTED_ROWS and integrity == "ok"
    except (OSError, sqlite3.Error, json.JSONDecodeError):
        return False


def build_index(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if not force and index_valid():
        print(f"Using validated full-pool index: {DB_PATH}", flush=True)
        return {"reused": True, "seconds": 0.0, "rows": EXPECTED_ROWS, "bytes": DB_PATH.stat().st_size}

    temporary = OUT / "candidate_index.building.sqlite3"
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    connection = sqlite3.connect(temporary)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-262144;
        CREATE TABLE records(
            rowid INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL,
            source TEXT NOT NULL,
            country TEXT NOT NULL,
            name_normalized TEXT NOT NULL,
            address_normalized TEXT NOT NULL,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL
        );
        """
    )
    inserted = 0
    for source in (2, 3):
        path = ROOT / f"dataset/train/train_source{source}.tsv"
        print(f"Indexing Source {source} records...", flush=True)
        for chunk in read_tsv(path, chunksize=CHUNK):
            names = chunk.business_name.map(expanded)
            addresses = chunk.business_address.map(clean)
            rows = zip(
                chunk.entity_id,
                [f"S{source}"] * len(chunk),
                chunk.country,
                names,
                addresses,
                chunk.business_name,
                chunk.business_address,
            )
            connection.executemany(
                """INSERT INTO records(
                    entity_id, source, country, name_normalized, address_normalized,
                    business_name, business_address
                ) VALUES(?,?,?,?,?,?,?)""",
                rows,
            )
            connection.commit()
            inserted += len(chunk)
            if inserted % 1_000_000 < CHUNK:
                print(f"  inserted {inserted:,}/{EXPECTED_ROWS:,}", flush=True)
    if inserted != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} indexed rows, found {inserted:,}")
    print("Creating unique ID index and full-text index...", flush=True)
    connection.execute("CREATE UNIQUE INDEX records_entity_id ON records(entity_id)")
    connection.execute(
        """CREATE VIRTUAL TABLE records_fts USING fts5(
            country, name_normalized, address_normalized,
            content='records', content_rowid='rowid',
            tokenize="unicode61 remove_diacritics 0 categories 'L* N* Co M*'"
        )"""
    )
    connection.execute("INSERT INTO records_fts(records_fts) VALUES('rebuild')")
    connection.execute("INSERT INTO records_fts(records_fts) VALUES('optimize')")
    connection.execute("CREATE VIRTUAL TABLE token_vocab USING fts5vocab(records_fts, 'col')")
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO metadata VALUES('signature', ?)", (json.dumps(index_signature(), sort_keys=True),))
    connection.commit()
    count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    unique_count = connection.execute("SELECT COUNT(*) FROM records INDEXED BY records_entity_id").fetchone()[0]
    integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
    connection.close()
    if count != EXPECTED_ROWS or unique_count != count or integrity != "ok":
        raise RuntimeError(f"Index validation failed: rows={count}, unique={unique_count}, check={integrity}")
    os.replace(temporary, DB_PATH)
    elapsed = time.perf_counter() - started
    result = {"reused": False, "seconds": elapsed, "rows": count, "bytes": DB_PATH.stat().st_size}
    print(f"Built index in {elapsed / 60:.1f} minutes ({result['bytes'] / 2**30:.2f} GiB).", flush=True)
    return result


def selectable_tokens(value, field):
    text = expanded(value) if field == "name" else clean(value)
    stop = STOP_NAME if field == "name" else STOP_ADDRESS
    result = []
    for token in set(text.split()):
        if token in stop:
            continue
        if token.isdigit() or len(token) >= 3:
            result.append(token)
    return result


def fetch_document_frequencies(connection, anchors):
    wanted = set()
    for row in anchors.itertuples():
        wanted.update(selectable_tokens(row.business_name, "name"))
        wanted.update(selectable_tokens(row.business_address, "address"))
    result = {}
    wanted = sorted(wanted)
    for start in range(0, len(wanted), 500):
        block = wanted[start : start + 500]
        placeholders = ",".join("?" for _ in block)
        rows = connection.execute(
            f"SELECT term, col, doc FROM token_vocab WHERE term IN ({placeholders})", block
        )
        for term, column, document_count in rows:
            if column in {"name_normalized", "address_normalized"}:
                result[(column, term)] = int(document_count)
    return result


def choose_query_tokens(value, field, document_frequency):
    column = "name_normalized" if field == "name" else "address_normalized"
    tokens = selectable_tokens(value, field)
    present = [(document_frequency.get((column, token), 0), token) for token in tokens]
    present = [item for item in present if item[0] > 0]
    return [token for _, token in sorted(present)[:MAX_QUERY_TOKENS]]


def quote_fts(value):
    return '"' + value.replace('"', '""') + '"'


def retrieve_route(connection, country, tokens, route, limit=FINAL_ROUTE_K):
    if not tokens:
        return []
    column = "name_normalized" if route == "name" else "address_normalized"
    weights = "0.0, 1.0, 0.0" if route == "name" else "0.0, 0.0, 1.0"
    expression = f"country:{quote_fts(country)} AND {column}:(" + " OR ".join(quote_fts(token) for token in tokens) + ")"
    return connection.execute(
        f"""SELECT r.entity_id, r.source, r.business_name, r.business_address,
                   bm25(records_fts, {weights}) AS score
            FROM records_fts
            JOIN records r ON r.rowid = records_fts.rowid
            WHERE records_fts MATCH ?
            ORDER BY score ASC, r.entity_id ASC
            LIMIT ?""",
        (expression, limit),
    ).fetchall()


def retrieve(anchors):
    token_path = OUT / "query_tokens.tsv"
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    document_frequency = fetch_document_frequencies(connection, anchors)
    connection.close()
    started = time.perf_counter()
    candidate_rows, token_rows = [], []
    thread_state = threading.local()

    def retrieve_reference(row):
        if not hasattr(thread_state, "connection"):
            thread_state.connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            thread_state.connection.execute("PRAGMA query_only=ON")
        merged = {}
        local_tokens = []
        for route, value in (("name", row.business_name), ("address", row.business_address)):
            tokens = choose_query_tokens(value, route, document_frequency)
            local_tokens.append(
                {
                    "source1_entity_id": row.entity_id,
                    "route": route,
                    "tokens": "|".join(tokens),
                    "token_document_frequencies": "|".join(
                        str(document_frequency[("name_normalized" if route == "name" else "address_normalized", token)])
                        for token in tokens
                    ),
                }
            )
            for rank, result in enumerate(retrieve_route(thread_state.connection, row.country, tokens, route), start=1):
                entity_id, source, business_name, business_address, score = result
                item = merged.setdefault(
                    entity_id,
                    {
                        "source1_entity_id": row.entity_id,
                        "candidate_entity_id": entity_id,
                        "country": row.country,
                        "candidate_source": source,
                        "candidate_name": business_name,
                        "candidate_address": business_address,
                        "name_rank": np.nan,
                        "name_bm25": np.nan,
                        "address_rank": np.nan,
                        "address_bm25": np.nan,
                    },
                )
                item[f"{route}_rank"] = rank
                item[f"{route}_bm25"] = score
        return list(merged.values()), local_tokens

    with ThreadPoolExecutor(max_workers=4) as executor:
        for query_number, (local_candidates, local_tokens) in enumerate(
            executor.map(retrieve_reference, anchors.itertuples(index=False)), start=1
        ):
            candidate_rows.extend(local_candidates)
            token_rows.extend(local_tokens)
            if query_number % 200 == 0:
                print(f"  retrieved {query_number:,}/{len(anchors):,} references", flush=True)
    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(["source1_entity_id", "candidate_entity_id"]).reset_index(drop=True)
    tokens = pd.DataFrame(token_rows)
    if candidates.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise RuntimeError("Duplicate candidate pairs after route union")
    if candidates.groupby("source1_entity_id").size().max() > FINAL_ROUTE_K * 2:
        raise RuntimeError("Candidate union exceeded configured maximum")
    tokens.to_csv(token_path, sep="\t", index=False)
    elapsed = time.perf_counter() - started
    print(f"Retrieved {len(candidates):,} pairs in {elapsed:.1f} seconds.", flush=True)
    return candidates, tokens, {"reused": False, "seconds": elapsed}


@lru_cache(maxsize=600_000)
def trigram_counter(value):
    if len(value) < 3:
        return Counter()
    return Counter(value[i : i + 3] for i in range(len(value) - 2))


def trigram_cosine(left, right):
    a, b = trigram_counter(left), trigram_counter(right)
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(count * b.get(token, 0) for token, count in a.items())
    return dot / math.sqrt(sum(x * x for x in a.values()) * sum(x * x for x in b.values()))


def token_jaccard(left, right):
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a or b else 0.0


def length_ratio(left, right):
    if not left and not right:
        return 1.0
    return min(len(left), len(right)) / max(len(left), len(right)) if max(len(left), len(right)) else 0.0


def scripts(value):
    result = set()
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if not name:
            continue
        result.add(name.split()[0])
    return result


def feature_row(reference, candidate, selected_name_tokens=(), selected_address_tokens=(), *, normalization=BASELINE_NORMALIZATION):
    if normalization not in {BASELINE_NORMALIZATION, TARGETED_NORMALIZATION, *CLEANUP_ABLATIONS}:
        raise ValueError(f"Unknown normalization: {normalization}")
    targeted = normalization != BASELINE_NORMALIZATION
    rules = CLEANUP_ABLATIONS.get(normalization)
    name_clean = (lambda value: clean_targeted(value, field="name", rules=rules)) if targeted else clean
    address_clean = (lambda value: clean_targeted(value, rules=rules)) if targeted else clean
    name_expand = (lambda value: expanded_targeted(value, rules=rules)) if targeted else expanded
    reference_name = name_clean(reference["business_name"])
    candidate_name = name_clean(candidate["business_name"])
    reference_expanded = name_expand(reference["business_name"])
    candidate_expanded = name_expand(candidate["business_name"])
    reference_address = address_clean(reference["business_address"])
    candidate_address = address_clean(candidate["business_address"])
    numeric = number_status(reference["business_address"], candidate["business_address"], strip_zeroes=True)
    reference_scripts, candidate_scripts = scripts(reference_name), scripts(candidate_name)
    script_union = reference_scripts | candidate_scripts
    # Reuse the originally selected rare tokens; do not consult/rebuild the index.
    name_selected = ({name_expand(t) for t in selected_name_tokens} - {""}) if targeted else set(selected_name_tokens)
    address_selected = ({address_clean(t) for t in selected_address_tokens} - {""}) if targeted else set(selected_address_tokens)
    candidate_name_tokens = set(candidate_expanded.split())
    candidate_address_tokens = set(candidate_address.split())
    return {
        "name_similarity": trigram_cosine(reference_name, candidate_name),
        "expanded_name_similarity": trigram_cosine(reference_expanded, candidate_expanded),
        "address_similarity": trigram_cosine(reference_address, candidate_address) if reference_address and candidate_address else np.nan,
        "name_token_jaccard": token_jaccard(reference_expanded, candidate_expanded),
        "address_token_jaccard": token_jaccard(reference_address, candidate_address),
        "distinctive_name_coverage": len(name_selected & candidate_name_tokens) / len(name_selected) if name_selected else 0.0,
        "distinctive_address_coverage": len(address_selected & candidate_address_tokens) / len(address_selected) if address_selected else 0.0,
        "exact_normalized_name": float(bool(reference_name) and reference_name == candidate_name),
        "exact_expanded_name": float(bool(reference_expanded) and reference_expanded == candidate_expanded),
        "exact_normalized_address": float(bool(reference_address) and reference_address == candidate_address),
        "numeric_same": float(numeric == "same_numeric_tokens"),
        "numeric_partial": float(numeric == "partial_overlap"),
        "numeric_disjoint": float(numeric == "disjoint_numeric_tokens"),
        "numeric_one_missing": float(numeric == "one_side_has_no_numbers"),
        "numeric_neither": float(numeric == "neither_has_numbers"),
        "missing_candidate_address": float(not candidate_address),
        "name_length_ratio": length_ratio(reference_name, candidate_name),
        "address_length_ratio": length_ratio(reference_address, candidate_address),
        "script_overlap": len(reference_scripts & candidate_scripts) / len(script_union) if script_union else 0.0,
        "script_exact": float(bool(reference_scripts) and reference_scripts == candidate_scripts),
        "cross_script": float(bool(reference_scripts and candidate_scripts) and not (reference_scripts & candidate_scripts)),
        "numeric_status": numeric,
    }


def token_maps(query_tokens):
    result = defaultdict(dict)
    for row in query_tokens.itertuples():
        result[row.source1_entity_id][row.route] = tuple(row.tokens.split("|")) if row.tokens else tuple()
    return result


def add_candidate_features(candidates, anchors, query_tokens, truth, *, normalization=BASELINE_NORMALIZATION):
    anchors_by_id = anchors.set_index("entity_id").to_dict("index")
    token_map = token_maps(query_tokens)
    truth_keys = set(zip(truth.source1_entity_id, truth.candidate_entity_id))
    rows = []
    started = time.perf_counter()
    for number, row in enumerate(candidates.itertuples(index=False), start=1):
        reference = anchors_by_id[row.source1_entity_id]
        candidate = {"business_name": row.candidate_name, "business_address": row.candidate_address}
        features = feature_row(
            reference,
            candidate,
            token_map[row.source1_entity_id].get("name", ()),
            token_map[row.source1_entity_id].get("address", ()),
            normalization=normalization,
        )
        base = row._asdict()
        base.update(features)
        base["label"] = int((row.source1_entity_id, row.candidate_entity_id) in truth_keys)
        rows.append(base)
        if number % 50_000 == 0:
            print(f"  featured {number:,}/{len(candidates):,} candidates", flush=True)
    result = pd.DataFrame(rows)
    print(f"Computed candidate features in {time.perf_counter() - started:.1f} seconds.", flush=True)
    return result


def lookup_records(entity_ids):
    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    result = {}
    entity_ids = sorted(set(entity_ids))
    for start in range(0, len(entity_ids), 800):
        block = entity_ids[start : start + 800]
        placeholders = ",".join("?" for _ in block)
        for row in connection.execute(
            f"SELECT entity_id, source, country, business_name, business_address FROM records WHERE entity_id IN ({placeholders})",
            block,
        ):
            result[row[0]] = {"source": row[1], "country": row[2], "business_name": row[3], "business_address": row[4]}
    connection.close()
    if set(result) != set(entity_ids):
        raise RuntimeError(f"Missing indexed target records: {len(set(entity_ids) - set(result))}")
    return result


def truth_diagnostics(anchors, truth, query_tokens, candidates):
    anchors_by_id = anchors.set_index("entity_id").to_dict("index")
    targets = lookup_records(truth.candidate_entity_id)
    token_map = token_maps(query_tokens)
    candidate_lookup = candidates.set_index(["source1_entity_id", "candidate_entity_id"])[
        ["name_rank", "address_rank"]
    ].to_dict("index")
    rows = []
    for row in truth.itertuples(index=False):
        reference = anchors_by_id[row.source1_entity_id]
        candidate = targets[row.candidate_entity_id]
        features = feature_row(
            reference,
            candidate,
            token_map[row.source1_entity_id].get("name", ()),
            token_map[row.source1_entity_id].get("address", ()),
        )
        ranks = candidate_lookup.get((row.source1_entity_id, row.candidate_entity_id), {})
        rows.append(
            {
                "source1_entity_id": row.source1_entity_id,
                "candidate_entity_id": row.candidate_entity_id,
                "country": reference["country"],
                "candidate_source": candidate["source"],
                "reference_name": reference["business_name"],
                "candidate_name": candidate["business_name"],
                "reference_address": reference["business_address"],
                "candidate_address": candidate["business_address"],
                "name_rank": ranks.get("name_rank", np.nan),
                "address_rank": ranks.get("address_rank", np.nan),
                "retrieved": bool(ranks),
                **features,
            }
        )
    result = pd.DataFrame(rows)
    result["low_name_similarity"] = result.name_similarity.lt(0.3)
    return result


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def grouped_splits(anchors, truth):
    ids = anchors.entity_id.tolist()
    union = UnionFind(ids)
    for key_series in (
        anchors.business_name.map(core_name),
        anchors.business_address.map(clean),
    ):
        groups = defaultdict(list)
        for entity_id, key in zip(ids, key_series):
            if key:
                groups[key].append(entity_id)
        for members in groups.values():
            for member in members[1:]:
                union.union(members[0], member)
    truth_owners = defaultdict(list)
    for row in truth.itertuples(index=False):
        truth_owners[row.candidate_entity_id].append(row.source1_entity_id)
    for members in truth_owners.values():
        for member in members[1:]:
            union.union(members[0], member)
    groups = np.array([union.find(entity_id) for entity_id in ids])
    strata = anchors.country.astype(str) + "|" + anchors.match_count.astype(int).eq(0).map({True: "singleton", False: "matched"})
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    assignments = np.empty(len(anchors), dtype=object)
    for fold, (_, test_indices) in enumerate(splitter.split(np.zeros(len(anchors)), strata, groups)):
        assignments[test_indices] = SPLITS[fold]
    result = anchors[["entity_id", "country", "match_count"]].copy()
    result["group_id"] = groups
    result["split"] = assignments
    if result.groupby("group_id").split.nunique().max() != 1:
        raise RuntimeError("Connected reference group leaked across splits")
    return result


def pair_fbeta(truth_ids, predicted_ids, beta=0.5):
    truth_ids, predicted_ids = set(truth_ids), set(predicted_ids)
    if not truth_ids:
        return 1.0 if not predicted_ids else 0.0
    if not predicted_ids:
        return 0.0
    true_positive = len(truth_ids & predicted_ids)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted_ids)
    recall = true_positive / len(truth_ids)
    beta_squared = beta * beta
    return (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)


def score_references(reference_ids, truth_map, predictions):
    return float(np.mean([pair_fbeta(truth_map.get(entity_id, set()), predictions.get(entity_id, set())) for entity_id in reference_ids]))


def retrieval_metrics(anchors, truth, candidates, diagnostics):
    truth_map = defaultdict(set)
    for row in truth.itertuples(index=False):
        truth_map[row.source1_entity_id].add(row.candidate_entity_id)
    rows = []
    for k in TOP_KS:
        for route in ("name", "address", "union"):
            if route == "name":
                selected = candidates[candidates.name_rank.le(k)]
            elif route == "address":
                selected = candidates[candidates.address_rank.le(k)]
            else:
                selected = candidates[candidates.name_rank.le(k) | candidates.address_rank.le(k)]
            selected_map = selected.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
            links_retrieved = sum(len(truth_map[entity_id] & selected_map.get(entity_id, set())) for entity_id in anchors.entity_id)
            nonsingletons = [entity_id for entity_id in anchors.entity_id if truth_map[entity_id]]
            complete = sum(truth_map[entity_id].issubset(selected_map.get(entity_id, set())) for entity_id in nonsingletons)
            ceilings = []
            for entity_id in anchors.entity_id:
                g = len(truth_map[entity_id])
                if g == 0:
                    ceilings.append(1.0)
                else:
                    t = len(truth_map[entity_id] & selected_map.get(entity_id, set()))
                    ceilings.append(1.25 * t / (0.25 * g + t) if t else 0.0)
            counts = selected.groupby("source1_entity_id").size().reindex(anchors.entity_id, fill_value=0)
            rows.append(
                {
                    "route": route,
                    "per_route_k": k,
                    "true_links_retrieved": links_retrieved,
                    "true_links": len(truth),
                    "link_recall": links_retrieved / len(truth),
                    "complete_match_sets": complete,
                    "nonsingleton_references": len(nonsingletons),
                    "complete_match_set_rate": complete / len(nonsingletons),
                    "ideal_macro_f0_5_ceiling": float(np.mean(ceilings)),
                    "mean_candidates": float(counts.mean()),
                    "median_candidates": float(counts.median()),
                    "p95_candidates": float(counts.quantile(0.95)),
                    "zero_candidate_references": int(counts.eq(0).sum()),
                }
            )
    summary = pd.DataFrame(rows)

    fixed = diagnostics.copy()
    fixed["retrieved_name"] = fixed.name_rank.le(FINAL_ROUTE_K)
    fixed["retrieved_address"] = fixed.address_rank.le(FINAL_ROUTE_K)
    fixed["retrieved_union"] = fixed.retrieved_name | fixed.retrieved_address
    breakdown = []
    masks = {
        "all": pd.Series(True, index=fixed.index),
        "country=India": fixed.country.eq("India"),
        "country=US": fixed.country.eq("US"),
        "source=S2": fixed.candidate_source.eq("S2"),
        "source=S3": fixed.candidate_source.eq("S3"),
        "missing_candidate_address": fixed.missing_candidate_address.eq(1),
        "low_name_similarity": fixed.low_name_similarity,
    }
    all_truth_by_ref = fixed.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    union_retrieved = fixed[fixed.retrieved_union].groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    for subgroup, mask in masks.items():
        group = fixed[mask]
        flagged_refs = sorted(group.source1_entity_id.unique())
        for route in ("name", "address", "union"):
            retrieved_column = f"retrieved_{route}"
            complete = 0
            if flagged_refs:
                if route == "union":
                    route_map = union_retrieved
                else:
                    route_map = fixed[fixed[retrieved_column]].groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
                complete = sum(all_truth_by_ref[ref].issubset(route_map.get(ref, set())) for ref in flagged_refs)
            breakdown.append(
                {
                    "subgroup": subgroup,
                    "route": route,
                    "links": len(group),
                    "links_retrieved": int(group[retrieved_column].sum()),
                    "link_recall": float(group[retrieved_column].mean()) if len(group) else np.nan,
                    "flagged_references": len(flagged_refs),
                    "flagged_references_with_all_entity_matches": complete,
                    "complete_entity_rate": complete / len(flagged_refs) if flagged_refs else np.nan,
                }
            )
    return summary, pd.DataFrame(breakdown)


def train_matcher(features, anchors, truth, splits, *, evaluated_splits=("train", "validation", "test")):
    split_map = splits.set_index("entity_id").split.to_dict()
    features = features.copy()
    features["split"] = features.source1_entity_id.map(split_map)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=SEED,
    )
    train = features[features.split.eq("train")]
    if train.label.nunique() != 2:
        raise RuntimeError("Training candidates do not contain both classes")
    model.fit(train[FEATURES], train.label)
    features["probability"] = model.predict_proba(features[FEATURES])[:, 1]
    truth_map = defaultdict(set)
    for row in truth.itertuples(index=False):
        truth_map[row.source1_entity_id].add(row.candidate_entity_id)
    validation_refs = splits[splits.split.eq("validation")].entity_id.tolist()
    validation = features[features.split.eq("validation")]
    probability_quantiles = validation.probability.quantile(np.linspace(0, 1, 101)).to_numpy()
    thresholds = sorted(set(np.r_[np.linspace(0.01, 0.99, 99), probability_quantiles, 1.000001]))
    threshold_rows = []
    for threshold in thresholds:
        accepted = validation[validation.probability.ge(threshold)]
        predictions = accepted.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
        threshold_rows.append({"threshold": threshold, "macro_f0_5": score_references(validation_refs, truth_map, predictions), "predicted_pairs": len(accepted)})
    threshold_frame = pd.DataFrame(threshold_rows)
    best = threshold_frame.sort_values(["macro_f0_5", "threshold"], ascending=[False, False]).iloc[0]
    threshold = float(best.threshold)
    features["accepted"] = features.probability.ge(threshold)

    result = {"selected_threshold": threshold, "features": FEATURES, "model": model.get_params(), "splits": {}}
    prediction_rows = []
    for split_name in evaluated_splits:
        reference_ids = splits[splits.split.eq(split_name)].entity_id.tolist()
        part = features[features.split.eq(split_name)]
        accepted = part[part.accepted]
        predictions = accepted.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
        true_positive = int(accepted.label.sum())
        predicted_pairs = len(accepted)
        total_truth = sum(len(truth_map[entity_id]) for entity_id in reference_ids)
        singleton_ids = [entity_id for entity_id in reference_ids if not truth_map[entity_id]]
        false_accept_singletons = sum(bool(predictions.get(entity_id)) for entity_id in singleton_ids)
        split_metrics = {
            "references": len(reference_ids),
            "macro_f0_5": score_references(reference_ids, truth_map, predictions),
            "pair_precision": true_positive / predicted_pairs if predicted_pairs else 1.0,
            "pair_recall": true_positive / total_truth if total_truth else 1.0,
            "true_positive_pairs": true_positive,
            "predicted_pairs": predicted_pairs,
            "true_pairs": total_truth,
            "singleton_references": len(singleton_ids),
            "singleton_false_acceptances": false_accept_singletons,
            "singleton_false_acceptance_rate": false_accept_singletons / len(singleton_ids) if singleton_ids else np.nan,
        }
        result["splits"][split_name] = split_metrics
        if split_name == "test":
            for entity_id in reference_ids:
                ids = sorted(predictions.get(entity_id, set()))
                prediction_rows.append({"source1_entity_id": entity_id, "matched_entity_ids": ",".join(ids)})
    return model, features, result, threshold_frame, pd.DataFrame(prediction_rows)


def error_analysis(scored, diagnostics, splits, model_metrics):
    split_map = splits.set_index("entity_id").split.to_dict()
    diagnostics = diagnostics.copy()
    diagnostics["split"] = diagnostics.source1_entity_id.map(split_map)
    test_truth = diagnostics[diagnostics.split.eq("test")]
    test_scored = scored[scored.split.eq("test")]
    scored_lookup = test_scored.set_index(["source1_entity_id", "candidate_entity_id"])[["probability", "accepted"]].to_dict("index")
    diagnostic_rows = []
    for row in test_truth.itertuples(index=False):
        scoring = scored_lookup.get((row.source1_entity_id, row.candidate_entity_id))
        if not row.retrieved:
            error_type = "retrieval_miss"
            probability, accepted = np.nan, False
        elif scoring and not scoring["accepted"]:
            error_type = "retrieved_positive_rejected"
            probability, accepted = scoring["probability"], False
        else:
            continue
        diagnostic_rows.append({**row._asdict(), "error_type": error_type, "probability": probability, "accepted": accepted})
    accepted_negatives = test_scored[test_scored.accepted & test_scored.label.eq(0)].sort_values("probability", ascending=False).head(40)
    for row in accepted_negatives.itertuples(index=False):
        diagnostic_rows.append({**row._asdict(), "error_type": "accepted_negative"})
    errors = pd.DataFrame(diagnostic_rows)
    test_references = set(splits.loc[splits.split.eq("test"), "entity_id"])
    retrieval_misses = int((test_truth.retrieved.eq(False)).sum())
    retrieved_rejected = sum(row.get("error_type") == "retrieved_positive_rejected" for row in diagnostic_rows)
    accepted_negative_count = int((test_scored.accepted & test_scored.label.eq(0)).sum())
    breakdown = {
        "test_references": len(test_references),
        "retrieval_missed_true_links": retrieval_misses,
        "retrieved_but_rejected_true_links": int(retrieved_rejected),
        "accepted_negative_pairs": accepted_negative_count,
        "recommendation": (
            "Improve retrieval before considering a larger matcher."
            if retrieval_misses > retrieved_rejected
            else "The candidate set contains more missed decisions than retrieval misses; improve matching features or model next."
        ),
    }
    return errors, breakdown


def singleton_diagnostics(anchors, features):
    singleton_ids = set(anchors.loc[anchors.match_count.astype(int).eq(0), "entity_id"])
    rows = []
    for entity_id in sorted(singleton_ids):
        part = features[features.source1_entity_id.eq(entity_id)]
        rows.append(
            {
                "source1_entity_id": entity_id,
                "country": anchors.loc[anchors.entity_id.eq(entity_id), "country"].iloc[0],
                "candidate_count": len(part),
                "max_name_similarity": part.name_similarity.max() if len(part) else np.nan,
                "max_address_similarity": part.address_similarity.max() if len(part) else np.nan,
                "max_probability": part.probability.max() if "probability" in part and len(part) else np.nan,
                "accepted_candidates": int(part.accepted.sum()) if "accepted" in part else 0,
            }
        )
    result = pd.DataFrame(rows)
    return result


def plot_results(features):
    valid = features[features.address_similarity.notna()]
    positives = valid[valid.label.eq(1)]
    negatives = valid[valid.label.eq(0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained")
    for ax, frame, title in ((axes[0], positives, "Retrieved true links"), (axes[1], negatives, "Retrieved nonmatches")):
        mesh = ax.hexbin(frame.name_similarity, frame.address_similarity, gridsize=35, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="Pairs per hexagon (log scale)")
        ax.set(title=f"{title}\nn={len(frame):,}", xlabel="Name similarity", ylabel="Address similarity", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    fig.suptitle("Candidates retrieved from the complete Source 2/3 training pool")
    fig.savefig(OUT / "retrieved_candidate_similarity.png", dpi=160)
    plt.close(fig)


def write_outputs(index_stats, retrieval_stats, candidates, features, retrieval_summary, retrieval_breakdown, model_metrics, threshold_frame, predictions, errors, error_breakdown, singletons, splits):
    candidate_counts = candidates.groupby("source1_entity_id").size()
    split_counts = splits.assign(singleton=splits.match_count.astype(int).eq(0)).groupby(
        ["split", "country", "singleton"]
    ).size().rename("references").reset_index()
    summary = {
        "seed": SEED,
        "reference_queries": 2000,
        "indexed_candidate_records": EXPECTED_ROWS,
        "index": index_stats,
        "retrieval": {
            **retrieval_stats,
            "candidate_pairs": len(candidates),
            "references_with_candidates": int(candidates.source1_entity_id.nunique()),
            "candidate_count_percentiles": {str(q): float(candidate_counts.quantile(q)) for q in (0, 0.5, 0.9, 0.95, 0.99, 1)},
            "fixed_union": retrieval_summary.query("route == 'union' and per_route_k == 100").iloc[0].to_dict(),
        },
        "split_counts": split_counts.to_dict("records"),
        "model": model_metrics,
        "error_breakdown": error_breakdown,
        "singletons": {
            "references": len(singletons),
            "mean_candidates": float(singletons.candidate_count.mean()),
            "p95_max_name_similarity": float(singletons.max_name_similarity.quantile(0.95)),
            "p95_max_address_similarity": float(singletons.max_address_similarity.quantile(0.95)),
        },
        "feature_pairs": len(features),
        "retrieval_metrics": retrieval_summary.to_dict("records"),
        "retrieval_breakdown": retrieval_breakdown.to_dict("records"),
        "validation_thresholds": threshold_frame.to_dict("records"),
        "singleton_diagnostics": singletons.to_dict("records"),
    }
    features.to_csv(OUT / "candidate_pairs.tsv.gz", sep="\t", index=False, compression="gzip")
    predictions.to_csv(OUT / "test_predictions.tsv", sep="\t", index=False)
    if errors.empty:
        errors = pd.DataFrame(columns=["error_type"])
    errors.to_csv(OUT / "errors.tsv", sep="\t", index=False)
    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value)}")

    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")


def validate_outputs(anchors, truth):
    validate_raw_inputs()
    if not index_valid():
        raise RuntimeError("Full-pool index failed its signature, row-count, or integrity check")
    candidates = pd.read_csv(OUT / "candidate_pairs.tsv.gz", sep="\t")
    predictions = read_tsv(OUT / "test_predictions.tsv")
    query_tokens = read_tsv(OUT / "query_tokens.tsv")
    if candidates.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise RuntimeError("Duplicate candidate pair in saved output")
    if not candidates.candidate_entity_id.str.startswith(("S2-", "S3-")).all():
        raise RuntimeError("Candidate output contains a non-S2/S3 identifier")
    if candidates.name_rank.dropna().gt(FINAL_ROUTE_K).any() or candidates.address_rank.dropna().gt(FINAL_ROUTE_K).any():
        raise RuntimeError("Saved route rank exceeds the configured limit")
    if query_tokens.groupby(["source1_entity_id", "route"]).size().ne(1).any() or len(query_tokens) != len(anchors) * 2:
        raise RuntimeError("Query-token audit table is incomplete or duplicated")
    truth_keys = set(zip(truth.source1_entity_id, truth.candidate_entity_id))
    saved_positive_keys = set(
        zip(candidates.loc[candidates.label.eq(1), "source1_entity_id"], candidates.loc[candidates.label.eq(1), "candidate_entity_id"])
    )
    if not saved_positive_keys.issubset(truth_keys):
        raise RuntimeError("A saved positive label is absent from ground truth")
    split_by_ref = candidates.groupby("source1_entity_id").split.first().to_dict()
    for values in (anchors.business_name.map(core_name), anchors.business_address.map(clean)):
        check = pd.DataFrame({"key": values, "split": anchors.entity_id.map(split_by_ref)})
        check = check[check.key.ne("")]
        if check.groupby("key").split.nunique().gt(1).any():
            raise RuntimeError("Canonical reference family leaked across splits")
    truth_splits = truth.assign(split=truth.source1_entity_id.map(split_by_ref))
    if truth_splits.groupby("candidate_entity_id").split.nunique().gt(1).any():
        raise RuntimeError("Shared true target leaked across splits")
    test_refs = set(entity_id for entity_id, split in split_by_ref.items() if split == "test")
    if set(predictions.source1_entity_id) != test_refs or predictions.source1_entity_id.duplicated().any():
        raise RuntimeError("Test predictions are not one row per held-out reference")
    accepted = candidates[candidates.accepted].groupby("source1_entity_id").candidate_entity_id.agg(lambda values: ",".join(sorted(values))).to_dict()
    if any(row.matched_entity_ids != accepted.get(row.source1_entity_id, "") for row in predictions.itertuples(index=False)):
        raise RuntimeError("Saved predictions disagree with accepted candidate rows")
    print("Validated index, candidate labels/ranks, grouped split isolation, predictions, and raw-data fingerprints.", flush=True)


def cleanup_row_audit(anchors, candidates, *, rules=None):
    """Count unique source records, never query/candidate pair repetitions."""
    references = anchors[["entity_id", "business_name", "business_address"]].copy()
    references["population"] = "sampled_references"
    records = candidates[["candidate_entity_id", "candidate_name", "candidate_address"]].drop_duplicates("candidate_entity_id")
    records = records.rename(columns={"candidate_entity_id": "entity_id", "candidate_name": "business_name", "candidate_address": "business_address"})
    records["population"] = "retrieved_candidate_records"
    counts, totals, changed_ids = [], [], set()
    for population, group in pd.concat([references, records]).groupby("population"):
        field_counts = {field: Counter() for field in ("name", "address")}
        any_changed = 0
        for row in group.itertuples(index=False):
            record_changed = False
            for field in ("name", "address"):
                raw = getattr(row, f"business_{field}")
                changes = {}
                new = clean_targeted(raw, field=field, changes=changes, rules=rules)
                for rule, changed in changes.items():
                    field_counts[field][rule] += int(changed)
                differs = clean(raw) != new
                if field == "name":
                    differs |= expanded(raw) != expanded_targeted(raw, rules=rules)
                field_counts[field]["final_differs_from_baseline"] += int(differs)
                record_changed |= differs
            if record_changed:
                any_changed += 1
                changed_ids.add(row.entity_id)
        totals.append({"population": population, "changed_rows": any_changed, "rows": len(group), "rate": any_changed / len(group)})
        for field, rule_counts in field_counts.items():
            for rule, count in rule_counts.items():
                counts.append({"population": population, "field": field, "rule": rule, "changed_rows": count, "rows": len(group), "rate": count / len(group)})
    return counts, totals, changed_ids


def update_cleanup_report(comparison):
    before = comparison["baseline_model"]["splits"]["validation"]
    after = comparison["model"]["splits"]["validation"]
    validation_changes = next(r for r in comparison["transitions"] if r["split"] == "validation" and r["scope"] == "all")
    affected_changes = next(r for r in comparison["transitions"] if r["split"] == "validation" and r["scope"] == "cleanup_affected")
    training = next(r for r in comparison["transitions"] if r["split"] == "train" and r["scope"] == "all")
    false_before = before["predicted_pairs"] - before["true_positive_pairs"]
    false_after = after["predicted_pairs"] - after["true_positive_pairs"]
    lines = [
        "<!-- targeted-cleanup:start -->", "## Targeted cleanup: fixed-candidate comparison", "",
        f"**Finding:** cleanup changes a meaningful fraction of the sampled text, but the measured matching benefit is small and comes with more false matches. Across {before['references']:,} validation references, macro F₀.₅ moves from {before['macro_f0_5']:.4f} to {after['macro_f0_5']:.4f}; false-positive pairs rise from {false_before:,} to {false_after:,}. Retain this as a versioned experiment rather than promoting it as a demonstrated precision improvement.", "",
        "Raw datasets and the full-pool index remain unchanged. This experiment applies `targeted_cleanup_v2` to the original train/validation candidate lists and reference splits; retrieval and baseline outputs still use `baseline_v1`. Test references were not rescored. The effect on retrieval is unmeasured.", "",
        "Cleanup preserves letters and combining marks across scripts, case-folds text, preserves numeric separators, removes integer leading zeros, removes appended websites (using the domain stem for website-only names), and joins dotted legal suffixes. Decimal fractional digits are preserved; comma-separated numeric components are treated conservatively. No address imputation, transliteration, new model features, or classifier hyperparameter changes were introduced.", "",
        "### How much text changed?", "",
        "Counts use unique source records in the train/validation experiment, not repeated pairs or the full training pool. Empty fields remain in denominators. Per-rule counts compare consecutive cleanup stages and overlap; numeric-separator preservation counts rows retaining separators that v1 removed. Lowercase and punctuation handling were already present in v1. The final-difference row measures actual changes relative to baseline normalized or expanded values.", "",
    ]
    for row in comparison["row_counts"]:
        lines.append(f"- {row['population'].replace('_', ' ')}: **{row['changed_rows']:,}/{row['rows']:,} ({row['rate']:.2%})** have at least one changed final name/address representation.")
    lookup = {(r["population"], r["field"], r["rule"]): r for r in comparison["rule_counts"]}
    groups = [(p, f) for p in ("sampled_references", "retrieved_candidate_records") for f in ("name", "address")]
    lines += ["", "| Rule | Reference names | Reference addresses | Candidate names | Candidate addresses |", "|---|---:|---:|---:|---:|"]
    for rule in dict.fromkeys(r["rule"] for r in comparison["rule_counts"]):
        cells = []
        for p, f in groups:
            row = lookup[(p, f, rule)]
            cells.append(f"{row['changed_rows']:,}/{row['rows']:,} ({row['rate']:.2%})")
        lines.append("| " + rule.replace("_", " ") + " | " + " | ".join(cells) + " |")
    lines += ["", "### Validation results", "", f"The experiment reused {comparison['fixed_candidate_pairs']:,} train/validation candidate pairs. The same classifier was fitted on {training['pairs']:,} training pairs and evaluated on {validation_changes['pairs']:,} validation pairs. The original validation threshold-selection procedure, including predict-nothing, was retained. Unretrieved positives still count as false negatives.", "", "| Metric | Baseline | Cleanup |", "|---|---:|---:|"]
    for label, key in (("References", "references"), ("Macro F₀.₅", "macro_f0_5"), ("Pair precision", "pair_precision"), ("Pair recall", "pair_recall"), ("True positives", "true_positive_pairs"), ("Predicted pairs", "predicted_pairs"), ("Labeled true pairs", "true_pairs")):
        a, b = before[key], after[key]
        lines.append(f"| {label} | {a:.6f} | {b:.6f} |" if isinstance(a, float) else f"| {label} | {a:,} | {b:,} |")
    lines += [f"| Selected threshold | {comparison['baseline_model']['selected_threshold']:.6f} | {comparison['model']['selected_threshold']:.6f} |", f"| Singleton false acceptance | {before['singleton_false_acceptances']}/{before['singleton_references']} | {after['singleton_false_acceptances']}/{after['singleton_references']} |", "", "| Validation scope | Positive / negative pairs | Recovered positives | Newly rejected positives | Corrected false acceptances | New false acceptances |", "|---|---:|---:|---:|---:|---:|"]
    for row in comparison["transitions"]:
        if row["split"] == "validation":
            lines.append(f"| {row['scope']} | {row['positive_pairs']:,} / {row['negative_pairs']:,} | {row['recovered_positives']} | {row['newly_rejected_positives']} | {row['corrected_false_acceptances']} | {row['new_false_acceptances']} |")
    delta = after["macro_f0_5"] - before["macro_f0_5"]
    lines += ["", f"Validation macro F₀.₅ changed by **{delta:+.6f}**. Pair precision changed from {before['pair_precision']:.2%} to {after['pair_precision']:.2%}, and pair recall from {before['pair_recall']:.2%} to {after['pair_recall']:.2%}. This is a tradeoff, not a uniform improvement. These are paired experiment results after retraining and threshold selection; an individual corrected decision cannot be attributed solely to its text edit. Small validation movements do not establish a population-level improvement.", "", "### Examples", "", "The notebook and metrics include original and cleaned text, before/after similarities, probabilities, and decisions for corrected errors, regressions, and persistent errors.", ""]
    seen = set()
    for example in comparison["examples"]:
        if example["category"] in seen:
            continue
        seen.add(example["category"])
        lines.append(f"- **{example['category'].replace('_', ' ')}** ({example['source1_entity_id']} → {example['candidate_entity_id']}): reference {example['reference_name_raw']!r}; candidate {example['candidate_name_raw']!r} → {example['candidate_name_after']!r}; address {example['candidate_address_raw']!r} → {example['candidate_address_after']!r}. Acceptance: {example['accepted_before']} → {example['accepted_after']}.")
    lines += ["", "### What the results establish", "",
        "**Numeric cleanup has broader reach than website or legal-suffix cleanup.** The rule table shows that preserving numeric separators affects more candidate addresses than either website removal or dotted legal suffixes affects names. This justifies preserving numeric structure in derived text, but does not establish that this rule improved classifier accuracy: the rules were evaluated together, without individual ablations.", "",
        f"**Some observed errors improved, and some worsened.** Among {affected_changes['positive_pairs']:,} retrieved validation positives whose text representation changed, {affected_changes['recovered_positives']} previously rejected pairs became accepted and {affected_changes['newly_rejected_positives']} previously accepted pairs became rejected. Among {affected_changes['negative_pairs']:,} affected negatives, {affected_changes['corrected_false_acceptances']} false acceptance was corrected and {affected_changes['new_false_acceptances']} new false acceptances appeared. These counts describe affected pairs, not causal effects of a specific rule.", "",
        f"**The acceptance threshold is part of the result.** Validation selected {comparison['model']['selected_threshold']:.2f} after cleanup, versus {comparison['baseline_model']['selected_threshold']:.2f} before. Retraining also changed the score distribution. Of the {validation_changes['recovered_positives']} recovered validation positives, {validation_changes['recovered_positives'] - affected_changes['recovered_positives']} had unchanged normalized/expanded text. The overall recall increase therefore cannot be described as direct correction of that many dirty records. Scores from the two models are not assumed to be calibrated or directly comparable.", "",
        "**Website removal has a concrete failure case.** The saved regression `Mr lbrothers.com` becomes `mr`, because the remaining qualifier prevents the website-only fallback. This loses the informative domain stem. Even a correctly retained stem, such as `firstindia.com → firstindia`, need not improve the matching decision. Domain removal should not be treated as universally beneficial.", "",
        "**Formatting cleanup cannot repair damaged information.** Converting `000 Brandenburg Road` to `0 brandenburg road` standardizes zeros but cannot recover a missing building-number digit. Keeping `12-3` distinct from `123` preserves a distinction without establishing which address is correct. Original fields remain necessary for inspection.", "",
        f"**False acceptance deserves attention before adopting the variant.** Singleton false acceptance changed from {before['singleton_false_acceptances']}/{before['singleton_references']} to {after['singleton_false_acceptances']}/{after['singleton_references']}. These are small counts, but together with the precision decrease they argue against claiming an unqualified improvement from the small macro-score increase.", "",
        "### Recommendation and limits", "",
        "The combined experiment supports keeping the baseline as the default and preserving cleanup as a comparison variant. Its next recommended check was to review the domain-plus-qualifier regression and compare the website, numeric, and dotted-suffix rules separately on the same candidates and splits. The single-rule follow-up, when available below, addresses that comparison.", "",
        "These results cover a selected train/validation shortlist, not all training records or production candidate prevalence. The 400-reference validation set also selected the threshold; it is not an independent final performance estimate. Test performance and retrieval gains remain unmeasured for this variant. Missing-address and cross-script matching remain outside the cleanup scope.", ""]
    lines += ["", "Reproduce with `uv run python eda/analysis/full_pool_baseline.py --cleanup-comparison`. The comparison is stored under `cleanup_comparison` in the existing metrics JSON; `cleanup_*` columns in the candidate table contain train/validation decisions only. Original candidate fields, scores, baseline metrics, and test predictions are preserved. Raw-file size/mtime fingerprints were verified before and after execution; index size/mtime were unchanged.", "<!-- targeted-cleanup:end -->"]
    path = EDA_ROOT / "reports/eda_first_pass.md"
    text = path.read_text()
    start, end = "<!-- targeted-cleanup:start -->", "<!-- targeted-cleanup:end -->"
    if start in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix + "\n".join(lines) + suffix
    else:
        text = text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    path.write_text(text)


def run_cleanup_comparison(anchors, truth):
    """Fixed-candidate train/validation experiment; no index writes or test scoring."""
    validate_raw_inputs()
    index_stat = (DB_PATH.stat().st_size, DB_PATH.stat().st_mtime_ns)
    metrics_path = OUT / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    original = pd.read_csv(OUT / "candidate_pairs.tsv.gz", sep="\t", keep_default_na=False)
    # Added cleanup columns are deliberately ignored on repeat runs.
    baseline = original[[c for c in original if not c.startswith("cleanup_")]].copy()
    baseline[FEATURES] = baseline[FEATURES].apply(pd.to_numeric, errors="raise")
    # Empty address similarity was serialized as an empty field.
    split_map = baseline.groupby("source1_entity_id").split.agg(lambda s: s.unique().tolist())
    if not split_map.map(len).eq(1).all():
        raise RuntimeError("Saved reference splits are inconsistent")
    splits = anchors[["entity_id", "country", "match_count"]].copy()
    splits["split"] = splits.entity_id.map(split_map.map(lambda s: s[0]))
    if splits.split.isna().any():
        raise RuntimeError("Cannot recover the fixed split for a reference")
    development_splits = splits[splits.split.isin(["train", "validation"])]
    development_anchors = anchors[anchors.entity_id.isin(development_splits.entity_id)]
    old = baseline[baseline.split.isin(["train", "validation"])].reset_index(drop=True)
    if old.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise RuntimeError("Duplicate saved candidates")
    truth_keys = set(zip(truth.source1_entity_id, truth.candidate_entity_id))
    if old.label.tolist() != [int(key in truth_keys) for key in zip(old.source1_entity_id, old.candidate_entity_id)]:
        raise RuntimeError("Saved candidate labels differ from truth")
    print("Auditing cleanup on unique train/validation source records...", flush=True)
    rule_counts, row_counts, changed_ids = cleanup_row_audit(development_anchors, old)
    tokens = read_tsv(OUT / "query_tokens.tsv")
    features = add_candidate_features(old, development_anchors, tokens, truth, normalization=TARGETED_NORMALIZATION)
    print("Training the unchanged matcher on cleaned features...", flush=True)
    _, scored, model_metrics, thresholds, _ = train_matcher(
        features, development_anchors, truth, development_splits, evaluated_splits=("train", "validation")
    )
    assert model_metrics["model"] == metrics["model"]["model"]
    assert old[["source1_entity_id", "candidate_entity_id", "split", "label"]].equals(scored[["source1_entity_id", "candidate_entity_id", "split", "label"]])
    affected = old.source1_entity_id.isin(changed_ids) | old.candidate_entity_id.isin(changed_ids)
    transitions = []
    for split in ("train", "validation"):
        for scope in ("all", "cleanup_affected"):
            mask = old.split.eq(split) & (affected if scope == "cleanup_affected" else True)
            part, after = old[mask], scored[mask]
            positive = part.label.eq(1)
            transitions.append({
                "split": split, "scope": scope, "pairs": len(part), "positive_pairs": int(positive.sum()),
                "negative_pairs": int((~positive).sum()),
                "recovered_positives": int((positive & ~part.accepted & after.accepted).sum()),
                "newly_rejected_positives": int((positive & part.accepted & ~after.accepted).sum()),
                "corrected_false_acceptances": int((~positive & part.accepted & ~after.accepted).sum()),
                "new_false_acceptances": int((~positive & ~part.accepted & after.accepted).sum()),
            })
    references = anchors.set_index("entity_id").to_dict("index")
    example_rows = []
    categories = {
        "recovered_positive": old.label.eq(1) & ~old.accepted & scored.accepted,
        "newly_rejected_positive": old.label.eq(1) & old.accepted & ~scored.accepted,
        "corrected_false_acceptance": old.label.eq(0) & old.accepted & ~scored.accepted,
        "new_false_acceptance": old.label.eq(0) & ~old.accepted & scored.accepted,
        "still_rejected_positive": old.label.eq(1) & ~old.accepted & ~scored.accepted,
    }
    for category, mask in categories.items():
        selected = old[mask & affected & old.split.eq("validation")].head(3)
        for i, row in selected.iterrows():
            reference = references[row.source1_entity_id]
            example = {"category": category, "source1_entity_id": row.source1_entity_id, "candidate_entity_id": row.candidate_entity_id, "split": row.split}
            for side, name, address in (("reference", reference["business_name"], reference["business_address"]), ("candidate", row.candidate_name, row.candidate_address)):
                example.update({f"{side}_name_raw": name, f"{side}_name_before": clean(name), f"{side}_name_after": clean_targeted(name, field="name"), f"{side}_expanded_before": expanded(name), f"{side}_expanded_after": expanded_targeted(name), f"{side}_address_raw": address, f"{side}_address_before": clean(address), f"{side}_address_after": clean_targeted(address)})
            for field in ("name_similarity", "expanded_name_similarity", "address_similarity", "probability", "accepted"):
                example[f"{field}_before"] = row[field]
                example[f"{field}_after"] = scored.loc[i, field]
            example_rows.append(example)
    comparison = {
        "normalization": TARGETED_NORMALIZATION, "baseline_normalization": BASELINE_NORMALIZATION,
        "index_normalization": BASELINE_NORMALIZATION, "index_version": INDEX_VERSION,
        "scope": "Fixed original candidates and splits; train/validation only; retrieval effects unmeasured.",
        "rule_count_method": "Unique train/validation reference rows and unique retrieved candidate rows. Rule counts compare successive text stages, overlap, and do not sum; numeric_separator_preservation counts rows retaining separators v1 removed. final_differs_from_baseline compares final normalized/expanded values with v1. Denominators include empty fields.",
        "row_counts": row_counts, "rule_counts": rule_counts,
        "baseline_model": metrics["model"], "model": model_metrics,
        "validation_thresholds": thresholds.to_dict("records"),
        "transitions": transitions, "examples": example_rows,
        "fixed_candidate_pairs": len(old), "raw_fingerprints_verified": True,
        "index_unchanged": True,
    }
    # Append only cleanup decisions; baseline columns and test predictions persist.
    indexed = scored.set_index(["source1_entity_id", "candidate_entity_id"])
    keys = pd.MultiIndex.from_frame(baseline[["source1_entity_id", "candidate_entity_id"]])
    for column in ("probability", "accepted"):
        original[f"cleanup_{column}"] = indexed[column].reindex(keys).to_numpy()
    original["cleanup_normalization"] = np.where(original.split.eq("test"), "", TARGETED_NORMALIZATION)
    metrics["cleanup_comparison"] = comparison
    validate_raw_inputs()
    if index_stat != (DB_PATH.stat().st_size, DB_PATH.stat().st_mtime_ns):
        raise RuntimeError("Retrieval index changed during cleanup experiment")
    original.to_csv(OUT / "candidate_pairs.tsv.gz", sep="\t", index=False, compression="gzip")
    def serializable(value):
        if isinstance(value, dict):
            return {k: serializable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [serializable(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    metrics_path.write_text(json.dumps(serializable(metrics), indent=2, allow_nan=False) + "\n")
    update_cleanup_report(comparison)
    print(json.dumps({"validation_before": metrics["model"]["splits"]["validation"], "validation_after": model_metrics["splits"]["validation"], "changed_rows": row_counts, "transitions": transitions}, indent=2), flush=True)


def ablation_decisions(old, scored, affected):
    """Pairwise decision changes against v1, preserving baseline row order."""
    positive = old.label.eq(1)
    masks = {
        "recovered_positives": positive & ~old.accepted & scored.accepted,
        "newly_rejected_positives": positive & old.accepted & ~scored.accepted,
        "corrected_false_acceptances": ~positive & old.accepted & ~scored.accepted,
        "new_false_acceptances": ~positive & ~old.accepted & scored.accepted,
    }
    summaries = []
    for scope, selection in (("all", old.split.eq("validation")), ("cleanup_affected", old.split.eq("validation") & affected)):
        summaries.append({"scope": scope, "pairs": int(selection.sum()), "positive_pairs": int((selection & positive).sum()), "negative_pairs": int((selection & ~positive).sum()), **{name: int((mask & selection).sum()) for name, mask in masks.items()}})
    changes = []
    for name, mask in masks.items():
        for i in old.index[mask & old.split.eq("validation")]:
            changes.append({"source1_entity_id": old.loc[i, "source1_entity_id"], "candidate_entity_id": old.loc[i, "candidate_entity_id"], "change": name, "text_changed": bool(affected.loc[i]), "probability_before": float(old.loc[i, "probability"]), "probability_after": float(scored.loc[i, "probability"])})
    return summaries, changes


def combined_cleanup_threshold_check(original, truth, threshold):
    """Rescore saved combined-cleanup probabilities; no additional model fit."""
    old = original[original.split.eq("validation")].reset_index(drop=True)
    scored = old.copy()
    scored["accepted"] = pd.to_numeric(old.cleanup_probability).ge(threshold)
    transitions, changes = ablation_decisions(old, scored, pd.Series(True, index=old.index))
    truth_map = defaultdict(set)
    for row in truth.itertuples(index=False):
        truth_map[row.source1_entity_id].add(row.candidate_entity_id)
    ids = old.source1_entity_id.unique().tolist()
    accepted = scored[scored.accepted]
    predictions = accepted.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
    tp = int(accepted.label.sum())
    total = sum(len(truth_map[i]) for i in ids)
    singleton_ids = [i for i in ids if not truth_map[i]]
    return {"threshold": threshold, "references": len(ids), "macro_f0_5": score_references(ids, truth_map, predictions), "pair_precision": tp / len(accepted) if len(accepted) else 1.0, "pair_recall": tp / total, "true_positive_pairs": tp, "predicted_pairs": len(accepted), "true_pairs": total, "singleton_references": len(singleton_ids), "singleton_false_acceptances": sum(bool(predictions.get(i)) for i in singleton_ids), "transitions": transitions[0], "decision_changes": changes}


def run_cleanup_ablations(anchors, truth):
    """Four independent baseline-plus-one-rule experiments on saved candidates."""
    validate_raw_inputs()
    preserved_paths = [DB_PATH, OUT / "candidate_pairs.tsv.gz", OUT / "query_tokens.tsv", OUT / "test_predictions.tsv"]
    fingerprints = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in preserved_paths}
    metrics_path = OUT / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    original = pd.read_csv(OUT / "candidate_pairs.tsv.gz", sep="\t", keep_default_na=False)
    original = original[original.split.isin(["train", "validation"])].reset_index(drop=True)
    old = original[[c for c in original if not c.startswith("cleanup_")]].copy()
    old[FEATURES] = old[FEATURES].apply(pd.to_numeric, errors="raise")
    assignment = old[["source1_entity_id", "split"]].drop_duplicates()
    if assignment.source1_entity_id.duplicated().any() or old.duplicated(["source1_entity_id", "candidate_entity_id"]).any():
        raise RuntimeError("Saved candidate/split uniqueness failed")
    development_anchors = anchors[anchors.entity_id.isin(assignment.source1_entity_id)].copy()
    splits = development_anchors[["entity_id", "country", "match_count"]].copy()
    splits["split"] = splits.entity_id.map(assignment.set_index("source1_entity_id").split)
    tokens = read_tsv(OUT / "query_tokens.tsv")
    truth_map = defaultdict(set)
    for row in truth.itertuples(index=False):
        truth_map[row.source1_entity_id].add(row.candidate_entity_id)
    assert old.label.tolist() == [int(r.candidate_entity_id in truth_map[r.source1_entity_id]) for r in old.itertuples()]
    validation_ids = splits.loc[splits.split.eq("validation"), "entity_id"].tolist()
    refs = development_anchors.set_index("entity_id").to_dict("index")
    combined = original.cleanup_accepted.astype(str).str.lower().eq("true")
    combined_masks = {
        "recovered_positives": old.split.eq("validation") & old.label.eq(1) & ~old.accepted & combined,
        "new_false_acceptances": old.split.eq("validation") & old.label.eq(0) & ~old.accepted & combined,
    }
    def pairs_for(mask):
        return set(zip(old.loc[mask, "source1_entity_id"], old.loc[mask, "candidate_entity_id"]))
    combined_sets = {k: pairs_for(v) for k, v in combined_masks.items()}
    variants = {}
    for name, rules in CLEANUP_ABLATIONS.items():
        started = time.perf_counter()
        print(f"\nAblation: {name}", flush=True)
        _, row_counts, changed_ids = cleanup_row_audit(development_anchors, old, rules=rules)
        affected = old.source1_entity_id.isin(changed_ids) | old.candidate_entity_id.isin(changed_ids)
        featured = add_candidate_features(old, development_anchors, tokens, truth, normalization=name)
        assert old[["source1_entity_id", "candidate_entity_id", "split", "label"]].equals(featured[["source1_entity_id", "candidate_entity_id", "split", "label"]])
        _, scored, model_metrics, thresholds, _ = train_matcher(featured, development_anchors, truth, splits, evaluated_splits=("train", "validation"))
        assert model_metrics["model"] == metrics["model"]["model"]
        transitions, changes = ablation_decisions(old, scored, affected)
        overlap = {}
        for change, combined_set in combined_sets.items():
            variant_set = {(r["source1_entity_id"], r["candidate_entity_id"]) for r in changes if r["change"] == change}
            overlap[change] = {"reproduced": len(variant_set & combined_set), "combined_total": len(combined_set), "variant_total": len(variant_set)}
        # A secondary, pre-specified threshold check distinguishes decision-policy
        # movement from the main validation-selected-threshold comparison.
        fixed = scored.copy()
        fixed["accepted"] = fixed.probability.ge(metrics["model"]["selected_threshold"])
        fixed_transitions, _ = ablation_decisions(old, fixed, affected)
        accepted = fixed[fixed.split.eq("validation") & fixed.accepted]
        predictions = accepted.groupby("source1_entity_id").candidate_entity_id.agg(set).to_dict()
        tp = int(accepted.label.sum())
        fixed_metrics = {
            "threshold": metrics["model"]["selected_threshold"], "references": len(validation_ids),
            "macro_f0_5": score_references(validation_ids, truth_map, predictions),
            "true_positive_pairs": tp, "predicted_pairs": len(accepted),
            "pair_precision": tp / len(accepted) if len(accepted) else 1.0,
            "pair_recall": tp / sum(len(truth_map[i]) for i in validation_ids),
            "transitions": fixed_transitions,
        }
        examples = []
        for kind in ("recovered_positives", "new_false_acceptances", "newly_rejected_positives", "corrected_false_acceptances"):
            selected = [r for r in changes if r["change"] == kind and r["text_changed"]][:2]
            for change in selected:
                reference = refs[change["source1_entity_id"]]
                candidate = old[(old.source1_entity_id == change["source1_entity_id"]) & (old.candidate_entity_id == change["candidate_entity_id"])].iloc[0]
                example = dict(change)
                for side, n, a in (("reference", reference["business_name"], reference["business_address"]), ("candidate", candidate.candidate_name, candidate.candidate_address)):
                    example.update({f"{side}_name_raw": n, f"{side}_name_before": expanded(n), f"{side}_name_after": expanded_targeted(n, rules=rules), f"{side}_address_raw": a, f"{side}_address_before": clean(a), f"{side}_address_after": clean_targeted(a, rules=rules)})
                examples.append(example)
        variants[name] = {"enabled_rules": sorted(rules), "seconds": time.perf_counter() - started, "row_counts": row_counts, "model": model_metrics, "transitions": transitions, "combined_overlap": overlap, "at_baseline_threshold": fixed_metrics, "decision_changes": changes, "examples": examples, "validation_thresholds": thresholds.to_dict("records")}
        print(json.dumps({"variant": name, "validation": model_metrics["splits"]["validation"], "transitions": transitions, "overlap": overlap}, indent=2), flush=True)
        del featured, scored, fixed
        trigram_counter.cache_clear()
    attribution = []
    for change, pairs in combined_sets.items():
        for ref, cand in sorted(pairs):
            reproducing = [name for name, result in variants.items() if any(r["change"] == change and r["source1_entity_id"] == ref and r["candidate_entity_id"] == cand for r in result["decision_changes"])]
            attribution.append({"change": change, "source1_entity_id": ref, "candidate_entity_id": cand, "reproduced_by": reproducing})
    validate_raw_inputs()
    assert all(fingerprints[p] == (p.stat().st_size, p.stat().st_mtime_ns) for p in preserved_paths)
    metrics["cleanup_ablations"] = {"design": "Baseline plus one rule at a time, not leave-one-out; fixed original train/validation pairs and splits; unchanged classifier and features; threshold selected independently on validation macro F0.5. Fixed baseline-threshold diagnostics are secondary. No test evaluation or retrieval rebuild.", "baseline_normalization": BASELINE_NORMALIZATION, "baseline_validation": metrics["model"]["splits"]["validation"], "baseline_threshold": metrics["model"]["selected_threshold"], "candidate_pairs": len(old), "validation_references": len(validation_ids), "variants": variants, "combined_decision_attribution": attribution, "raw_fingerprints_verified": True, "existing_candidate_index_and_prediction_files_unchanged": True}
    metrics["cleanup_ablations"]["combined_at_baseline_threshold"] = combined_cleanup_threshold_check(original, truth, metrics["model"]["selected_threshold"])
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    update_ablation_report(metrics["cleanup_ablations"])


def update_ablation_report(result):
    variants = result["variants"]
    best_name = max(variants, key=lambda n: variants[n]["model"]["splits"]["validation"]["macro_f0_5"])
    most_recovered = max(variants, key=lambda n: variants[n]["transitions"][0]["recovered_positives"])
    most_false = max(variants, key=lambda n: variants[n]["transitions"][0]["new_false_acceptances"])
    lines = ["<!-- cleanup-ablations:start -->", "## Single-rule cleanup ablations", "", f"**Finding:** `{best_name}` has the highest validation macro F₀.₅ among the four single-rule runs. `{most_recovered}` recovers the most previously rejected positives; `{most_false}` introduces the most new false acceptances. Counts below separate all decision changes from those on pairs whose text changed, and show overlap with the combined experiment.", "", result["design"], "", "Each row below retrains the same classifier with only that rule added to baseline normalization. Leading-zero cleanup preserves fractional digits before punctuation is removed. Numeric-separator preservation also standardizes numeric dash variants and surrounding spaces. The existing untyped numeric-agreement feature already ignores leading zeros in every variant; only the text-derived features change.", "", "| Variant | Threshold | Macro F₀.₅ | Precision | Recall | TP / predicted / true links | Singleton false acceptance |", "|---|---:|---:|---:|---:|---:|---:|"]
    baseline = result["baseline_validation"]
    def metric_row(name, threshold, m):
        return f"| {name} | {threshold:.4f} | {m['macro_f0_5']:.6f} | {m['pair_precision']:.2%} | {m['pair_recall']:.2%} | {m['true_positive_pairs']} / {m['predicted_pairs']} / {m['true_pairs']} | {m['singleton_false_acceptances']}/{m['singleton_references']} |"
    lines.append(metric_row("baseline", result["baseline_threshold"], baseline))
    for name, variant in result["variants"].items():
        lines.append(metric_row(name, variant["model"]["selected_threshold"], variant["model"]["splits"]["validation"]))
    lines += ["", "All main metrics cover 400 validation references. The same 1,383 labeled true links are the recall denominator, including 51 unretrieved positives. Candidate-level error transitions cover 1,332 retrieved positives and 77,483 negatives.", "", "| Variant | Recovered positives | Newly rejected positives | Corrected false acceptances | New false acceptances | Combined recoveries reproduced | Combined new false acceptances reproduced |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, variant in result["variants"].items():
        t = variant["transitions"][0]
        r, f = variant["combined_overlap"]["recovered_positives"], variant["combined_overlap"]["new_false_acceptances"]
        lines.append(f"| {name} | {t['recovered_positives']} | {t['newly_rejected_positives']} | {t['corrected_false_acceptances']} | {t['new_false_acceptances']} | {r['reproduced']}/{r['combined_total']} | {f['reproduced']}/{f['combined_total']} |")
    lines += ["", "### Changes on pairs whose text was affected", "", "| Variant | Changed references | Changed unique candidates | Affected validation positives / negatives | Recovered / newly rejected positives | Corrected / new false acceptances |", "|---|---:|---:|---:|---:|---:|"]
    for name, variant in result["variants"].items():
        rows = {r["population"]: r for r in variant["row_counts"]}
        r, c = rows["sampled_references"], rows["retrieved_candidate_records"]
        t = variant["transitions"][1]
        lines.append(f"| {name} | {r['changed_rows']}/{r['rows']} ({r['rate']:.2%}) | {c['changed_rows']}/{c['rows']} ({c['rate']:.2%}) | {t['positive_pairs']} / {t['negative_pairs']} | {t['recovered_positives']} / {t['newly_rejected_positives']} | {t['corrected_false_acceptances']} / {t['new_false_acceptances']} |")
    lines += ["", "### Fixed baseline-threshold diagnostic", "", "The following uses 0.99 for every model. This is a diagnostic comparison, not a separately optimized policy; retrained probabilities need not have identical calibration.", "", "| Variant | Macro F₀.₅ | Precision | Recall | Recovered / newly rejected positives | Corrected / new false acceptances |", "|---|---:|---:|---:|---:|---:|"]
    for name, variant in result["variants"].items():
        m = variant["at_baseline_threshold"]
        t = m["transitions"][0]
        lines.append(f"| {name} | {m['macro_f0_5']:.6f} | {m['pair_precision']:.2%} | {m['pair_recall']:.2%} | {t['recovered_positives']} / {t['newly_rejected_positives']} | {t['corrected_false_acceptances']} / {t['new_false_acceptances']} |")
    combined = result["combined_at_baseline_threshold"]
    t = combined["transitions"]
    lines.append(f"| combined cleanup | {combined['macro_f0_5']:.6f} | {combined['pair_precision']:.2%} | {combined['pair_recall']:.2%} | {t['recovered_positives']} / {t['newly_rejected_positives']} | {t['corrected_false_acceptances']} / {t['new_false_acceptances']} |")
    lines += ["", f"All four single-rule runs independently selected 0.99, so their fixed-threshold results equal their main results. The combined run originally selected 0.98. Applying 0.99 to its **same saved scores** gives {t['recovered_positives']} recovered positives and {t['new_false_acceptances']} new false acceptances, compared with 114 and 54 at 0.98. Thus {114 - t['recovered_positives']} of those recoveries and {54 - t['new_false_acceptances']} of those new false acceptances depend on the lower threshold. This is a direct within-model threshold comparison; it does not assign interactions to individual cleanup rules."]
    lines += ["", "### Examples and attribution limits", ""]
    for name, variant in result["variants"].items():
        for kind in ("recovered_positives", "new_false_acceptances", "corrected_false_acceptances"):
            examples = [e for e in variant["examples"] if e["change"] == kind]
            if examples:
                e = examples[0]
                lines.append(f"- **{name}, {kind.replace('_', ' ')}:** {e['source1_entity_id']} → {e['candidate_entity_id']}; reference name {e['reference_name_before']!r} → {e['reference_name_after']!r}; candidate name {e['candidate_name_before']!r} → {e['candidate_name_after']!r}; candidate address {e['candidate_address_before']!r} → {e['candidate_address_after']!r}.")
    for kind in ("recovered_positives", "new_false_acceptances"):
        rows = [r for r in result["combined_decision_attribution"] if r["change"] == kind]
        none = sum(not r["reproduced_by"] for r in rows)
        multiple = sum(len(r["reproduced_by"]) > 1 for r in rows)
        lines.append(f"\nFor combined-cleanup {kind.replace('_', ' ')}, {none}/{len(rows)} are reproduced by no single-rule model and {multiple}/{len(rows)} by multiple single-rule models. Counts cannot be added or assigned exclusively to one rule.")
    lines += ["", "These controlled pipeline comparisons show the effect of enabling each rule together with retraining and validation threshold selection. They do not isolate a direct causal effect on individual examples or interactions among rules. Retrieval and test performance remain unmeasured. Raw datasets, index, original candidate file, and test predictions were preserved.", "", "Reproduce with `uv run python eda/analysis/full_pool_baseline.py --cleanup-ablations`. Metrics, changed validation pair IDs, overlap attribution, and examples are contained in the existing `full_pool/metrics.json` under `cleanup_ablations`.", "<!-- cleanup-ablations:end -->"]
    path = EDA_ROOT / "reports/eda_first_pass.md"
    text = path.read_text()
    start, end = "<!-- cleanup-ablations:start -->", "<!-- cleanup-ablations:end -->"
    if start in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        text = prefix + "\n".join(lines) + suffix
    else:
        text = text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cleanup-comparison", action="store_true", help="Compare targeted cleanup on saved train/validation candidates without rebuilding retrieval")
    parser.add_argument("--cleanup-ablations", action="store_true", help="Run four baseline-plus-one-rule cleanup experiments on saved candidates")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    validate_raw_inputs()
    anchors = read_tsv(EDA_CACHE / "anchors.tsv")
    anchors["match_count"] = anchors.match_count.astype(int)
    truth = read_tsv(EDA_CACHE / "truth.tsv")
    if len(anchors) != 2000:
        raise RuntimeError(f"Expected the existing 2,000-reference sample, found {len(anchors)}")
    if args.cleanup_ablations:
        if args.cleanup_comparison or args.rebuild_index or args.validate_only:
            parser.error("--cleanup-ablations cannot be combined with other modes")
        run_cleanup_ablations(anchors, truth)
        return
    if args.cleanup_comparison:
        if args.rebuild_index or args.validate_only:
            parser.error("--cleanup-comparison cannot be combined with other modes")
        run_cleanup_comparison(anchors, truth)
        return
    if args.validate_only:
        validate_outputs(anchors, truth)
        return
    index_stats = build_index(args.rebuild_index)
    candidates, query_tokens, retrieval_stats = retrieve(anchors)
    features = add_candidate_features(candidates, anchors, query_tokens, truth)
    diagnostics = truth_diagnostics(anchors, truth, query_tokens, candidates)
    retrieval_summary, retrieval_breakdown = retrieval_metrics(anchors, truth, candidates, diagnostics)
    splits = grouped_splits(anchors, truth)
    _, scored, model_metrics, threshold_frame, predictions = train_matcher(features, anchors, truth, splits)
    errors, error_breakdown = error_analysis(scored, diagnostics, splits, model_metrics)
    singleton_results = singleton_diagnostics(anchors, scored)
    plot_results(scored)
    write_outputs(index_stats, retrieval_stats, candidates, scored, retrieval_summary, retrieval_breakdown, model_metrics, threshold_frame, predictions, errors, error_breakdown, singleton_results, splits)
    validate_outputs(anchors, truth)
    print(f"Complete. Outputs written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
