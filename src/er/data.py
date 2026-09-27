"""Read-only raw inputs, frozen business splits, scoped labels, artifact identity.

Only split preparation inspects all labels, for grouping/stratification. Stage
fitting receives labels through SplitStore.labels(), never the raw label path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
PAIR_KEYS = ["source1_entity_id", "candidate_entity_id"]


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fingerprint(path):
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return {"bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def inventory(paths):
    return {str(Path(p).relative_to(ROOT)): fingerprint(p) for p in sorted(paths)}


def config(path=ROOT / "configs/experiments.toml"):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def output_root(cfg):
    path = (ROOT / cfg["artifact_root"]).resolve()
    for protected in (ROOT / "dataset", ROOT / "eda", ROOT / "student_resource", ROOT / "src"):
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError(f"Artifact output overlaps protected inputs: {path}")
    return path


@lru_cache(maxsize=1)
def legacy():
    # Import definitions only. Never call the legacy runner or its output writers.
    sys.path.insert(0, str(ROOT / "eda/analysis"))
    return importlib.import_module("full_pool_baseline")


def validate_pairs(frame, query_ids):
    if frame.duplicated(PAIR_KEYS).any():
        raise ValueError("Duplicate candidate pairs")
    if not set(frame.source1_entity_id) <= set(query_ids):
        raise ValueError("Pairs contain an unknown query")
    if not frame.candidate_entity_id.str.startswith(("S2-", "S3-")).all():
        raise ValueError("Candidates must be Source 2/3 records")


def legacy_membership():
    import pandas as pd
    anchors = pd.read_csv(ROOT / "eda/reports/eda_data/anchors.tsv", sep="\t", keep_default_na=False)
    truth = pd.read_csv(ROOT / "eda/reports/eda_data/truth.tsv", sep="\t", keep_default_na=False)
    recovered = legacy().grouped_splits(anchors, truth)
    saved = {}
    for row in pd.read_csv(ROOT / "eda/reports/full_pool/candidate_pairs.tsv.gz", sep="\t",
                           usecols=["source1_entity_id", "split"]).drop_duplicates().itertuples(index=False):
        if row.source1_entity_id in saved:
            raise ValueError("Inconsistent historical query assignment")
        saved[row.source1_entity_id] = row.split
    for row in recovered.itertuples(index=False):
        if saved.get(row.entity_id, row.split) != row.split:
            raise ValueError("Recovered legacy grouping disagrees with saved assignment")
    return recovered


def exposure_ids():
    exposed = set()
    for name in ("anchors.tsv", "branch_candidates.tsv", "reviewed_examples.tsv"):
        for row in rows(ROOT / "eda/reports/eda_data" / name):
            for key in ("entity_id", "source1_entity_id", "other_reference_id"):
                if row.get(key, "").startswith("S1-"):
                    exposed.add(row[key])
    # records.tsv also contains references shown in the original EDA.
    for row in rows(ROOT / "eda/reports/eda_data/records.tsv"):
        if row["entity_id"].startswith("S1-"):
            exposed.add(row["entity_id"])
    return exposed


def prepare_splits(destination, source1, truth_path, cfg, historical=None, exposed=()):
    """Disk-backed grouping; retain every query, including zero-match businesses.

    Stable hash assignment targets country/singleton proportions in expectation;
    manifests record realized counts rather than claiming exact quotas.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(destination / "membership.sqlite3")
    db.executescript("""
        CREATE TABLE queries(id TEXT PRIMARY KEY, country TEXT NOT NULL,
          truth TEXT, group_id TEXT, split TEXT, transfer_eligible INTEGER);
        CREATE TABLE links(target TEXT NOT NULL, owner TEXT NOT NULL);
    """)
    with db:
        db.executemany("INSERT INTO queries(id,country) VALUES (?,?)",
                       ((r["entity_id"], r["country"]) for r in rows(source1)))
    print("Loaded Source 1 records; scanning labels for shared business groups...", flush=True)
    batch = []
    for row in rows(truth_path):
        batch.append(row)
        if len(batch) == 25000:
            _insert_truth(db, batch)
            batch = []
    _insert_truth(db, batch)
    if db.execute("SELECT count(*) FROM queries WHERE truth IS NULL").fetchone()[0]:
        raise ValueError("Missing ground-truth query rows")
    db.execute("CREATE INDEX link_target ON links(target)")
    # Only references participating in a shared label or legacy group need a UF node.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    shared = db.execute("""SELECT owner, min_owner FROM links JOIN
        (SELECT target, min(owner) min_owner FROM links GROUP BY target
         HAVING count(*) > 1) USING(target)""")
    for owner, first in shared:
        union(owner, first)
    historical = historical or []
    first_by_group = {}
    for row in historical:
        first = first_by_group.setdefault(row["group_id"], row["entity_id"])
        union(first, row["entity_id"])
    old = {r["entity_id"]: r["split"] for r in historical}
    forced = {}
    for entity_id, split in old.items():
        group = find(entity_id)
        assigned = "validation" if split in ("validation", "test") else "train"
        if assigned == "validation" or group not in forced:
            forced[group] = assigned
    exposed_groups = {find(x) for x in exposed}
    countries = defaultdict(set)
    for entity_id in parent:
        row = db.execute("SELECT country FROM queries WHERE id=?", (entity_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown historical/exposed business: {entity_id}")
        countries[find(entity_id)].add(row[0])
    group_matched = {}
    for entity_id in parent:
        value = db.execute("SELECT truth FROM queries WHERE id=?", (entity_id,)).fetchone()[0]
        group = find(entity_id)
        group_matched[group] = group_matched.get(group, False) or bool(value)
    counts, reassigned = Counter(), []
    print("Assigning frozen splits...", flush=True)
    cursor = db.execute("SELECT id,country,truth FROM queries ORDER BY id")
    while batch := cursor.fetchmany(25000):
        updates = []
        for entity_id, country, truth in batch:
            group = find(entity_id) if entity_id in parent else entity_id
            eligible = len(countries.get(group, {country})) == 1
            stratum = [sorted(countries.get(group, {country})), group_matched.get(group, bool(truth))]
            value = int(digest([cfg["seed"], group, stratum])[:16], 16) / 2**64
            split = forced.get(group)
            if split is None:
                if group in exposed_groups:
                    split = "validation" if value < cfg["validation_fraction"] / (1 - cfg["holdout_fraction"]) else "train"
                elif value < cfg["holdout_fraction"]:
                    split = "holdout"
                elif value < cfg["holdout_fraction"] + cfg["validation_fraction"]:
                    split = "validation"
                else:
                    split = "train"
            if entity_id in old and old[entity_id] != split:
                reassigned.append({"entity_id": entity_id, "old": old[entity_id], "new": split})
            counts[(country, split, bool(truth))] += 1
            updates.append((group, split, int(eligible), entity_id))
        with db:
            db.executemany("UPDATE queries SET group_id=?,split=?,transfer_eligible=? WHERE id=?", updates)
    if db.execute("SELECT group_id FROM queries GROUP BY group_id HAVING count(DISTINCT split)>1 LIMIT 1").fetchone():
        raise ValueError("Business group crosses splits")
    db.executescript("DROP TABLE links; CREATE INDEX query_split ON queries(split,country);")
    with (destination / "membership.tsv").open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["source1_entity_id", "group_id", "country", "split", "transfer_eligible"])
        writer.writerows(db.execute("SELECT id,group_id,country,split,transfer_eligible FROM queries ORDER BY id"))
    db.commit()
    db.execute("VACUUM")
    db.close()
    manifest = {
        "version": cfg["split_version"], "seed": cfg["seed"],
        "target_fractions": {s: cfg[f"{s}_fraction"] for s in ("train", "validation", "holdout")},
        "assignment": "stable group hash stratified by country set and match status; approximate quotas",
        "grouping": "Source 1 identity + shared labeled targets + preserved legacy groups",
        "inputs": {str(p): fingerprint(p) for p in (source1, truth_path)},
        "exposure_ids_hash": digest(sorted(exposed)), "exposed_queries": len(set(exposed)),
        "counts": [{"country": c, "split": s, "has_matches": m, "queries": n}
                   for (c, s, m), n in sorted(counts.items())],
        "reassignments": reassigned,
        "membership": fingerprint(destination / "membership.tsv"),
        "database": fingerprint(destination / "membership.sqlite3"),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def _insert_truth(db, batch):
    with db:
        for row in batch:
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            if len(ids) != len(set(ids)) or any(not x.startswith(("S2-", "S3-")) for x in ids):
                raise ValueError("Invalid truth target list")
            result = db.execute("UPDATE queries SET truth=? WHERE id=? AND truth IS NULL",
                                (row["matched_entity_ids"], row["source1_entity_id"]))
            if result.rowcount != 1:
                raise ValueError("Unknown or duplicate ground-truth query")
            db.executemany("INSERT INTO links VALUES (?,?)", ((x, row["source1_entity_id"]) for x in ids))


class SplitStore:
    def __init__(self, directory, profile, *, final=False):
        self.directory = Path(directory)
        self.profile = profile
        self.final = final
        self.db = sqlite3.connect(f"file:{self.directory / 'membership.sqlite3'}?mode=ro", uri=True)
        self.db.execute("PRAGMA query_only=ON")

    def close(self):
        self.db.close()

    def queries(self, purpose, ids=None):
        roles = {"fit": ("train", "train_countries"), "tune": ("validation", "tune_countries"),
                 "evaluate": ("holdout" if self.final else "validation", "evaluate_countries")}
        if purpose not in roles:
            raise ValueError(f"Unknown label purpose: {purpose}")
        if self.final and purpose != "evaluate":
            raise ValueError("Final evaluation cannot fit or tune")
        split, country_key = roles[purpose]
        countries = self.profile[country_key]
        placeholders = ",".join("?" for _ in countries)
        transfer = self.profile["train_countries"] != self.profile["evaluate_countries"]
        query = f"SELECT id,country FROM queries WHERE split=? AND country IN ({placeholders})"
        if transfer:
            query += " AND transfer_eligible=1"
        allowed = set(ids) if ids is not None else None
        return {i: c for i, c in self.db.execute(query + " ORDER BY id", [split, *countries])
                if allowed is None or i in allowed}

    def labels(self, query_ids, purpose):
        requested = set(query_ids)
        allowed = self.queries(purpose, requested)
        if requested != set(allowed):
            raise ValueError("Label access outside the authorized split/country group")
        result = {}
        ordered = sorted(requested)
        for start in range(0, len(ordered), 500):
            block = ordered[start:start + 500]
            placeholders = ",".join("?" for _ in block)
            for i, value in self.db.execute(f"SELECT id,truth FROM queries WHERE id IN ({placeholders})", block):
                result[i] = set(value.split(",")) if value else set()
        return result


def query_records(query_ids):
    import pandas as pd
    wanted = set(query_ids)
    result = []
    for chunk in pd.read_csv(ROOT / "dataset/train/train_source1.tsv", sep="\t", dtype=str,
                             keep_default_na=False, chunksize=100000):
        result.append(chunk[chunk.entity_id.isin(wanted)])
    frame = pd.concat(result, ignore_index=True)
    if set(frame.entity_id) != wanted or frame.entity_id.duplicated().any():
        raise ValueError("Missing/duplicate query records")
    return frame


def pool_connection():
    """Reuse only the original record store; FTS/BM25 scoring is not reused."""
    db = sqlite3.connect(f"file:{legacy().DB_PATH}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


def pool_records(batch_size, *, pilot_rows=0):
    """Deterministic, label-free full pool or evenly spaced engineering pilot."""
    import numpy as np
    import pandas as pd
    with pool_connection() as db:
        columns = "rowid,entity_id,source,country,business_name,business_address"
        if pilot_rows:
            maximum, count = db.execute("SELECT MAX(rowid),COUNT(*) FROM records").fetchone()
            if maximum != count:
                raise ValueError("Pilot sampling expects contiguous original pool row IDs")
            selected = np.linspace(1, count, min(count, pilot_rows), dtype=np.int64).tolist()
            for start in range(0, len(selected), batch_size):
                values = []
                for offset in range(start, min(start + batch_size, len(selected)), 500):
                    ids = selected[offset:min(offset + 500, start + batch_size, len(selected))]
                    values.extend(db.execute(f"SELECT {columns} FROM records WHERE rowid IN ({','.join('?' for _ in ids)}) ORDER BY rowid", ids))
                yield pd.DataFrame(values, columns=columns.split(","))
        else:
            cursor = db.execute(f"SELECT {columns} FROM records ORDER BY rowid")
            while values := cursor.fetchmany(batch_size):
                yield pd.DataFrame(values, columns=columns.split(","))


def pool_has_ids(ids):
    ordered = sorted(ids)
    with pool_connection() as db:
        for start in range(0, len(ordered), 500):
            batch = ordered[start:start + 500]
            count = db.execute(f"SELECT COUNT(*) FROM records WHERE entity_id IN ({','.join('?' for _ in batch)})", batch).fetchone()[0]
            if count != len(batch):
                return False
    return True
