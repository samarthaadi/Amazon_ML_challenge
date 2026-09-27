"""Stage 2: label-free full-pool retrieval and versioned candidate caches.

Implementation TBD: new retrieval methods, eligibility, ranking, budgets/losses.
The legacy adapter uses the existing two-route union, at K=100 per route.
Its stable `rank` is min(route rank), then ID; it is not a historical fused score.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

import pandas as pd

from .data import ROOT, digest, fingerprint, legacy, validate_pairs, write_json

LEGACY_COLUMNS = ["source1_entity_id", "candidate_entity_id", "country", "candidate_source",
                  "candidate_name", "candidate_address", "name_rank", "name_bm25",
                  "address_rank", "address_bm25"]


def cache_key(query_ids, provenance):
    return digest({"query_ids": sorted(query_ids), **provenance})


def build_index(records, state, config):
    if config["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: retrieval index")
    baseline = legacy()
    with sqlite3.connect(f"file:{baseline.DB_PATH}?mode=ro", uri=True) as db:
        stored = json.loads(db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()[0])
    if stored != baseline.index_signature():
        raise ValueError("Legacy full-pool index does not match raw inputs")
    return {"path": str(baseline.DB_PATH), "signature": stored, "method": "legacy_v1"}


def _rank(frame):
    result = frame.copy()
    result["_best_route_rank"] = result[["name_rank", "address_rank"]].min(axis=1)
    result = result.sort_values(["source1_entity_id", "_best_route_rank", "candidate_entity_id"])
    result["rank"] = result.groupby("source1_entity_id").cumcount() + 1
    return result.drop(columns="_best_route_rank").reset_index(drop=True)


def retrieve(queries, index, config):
    """Return the exact matcher input without consulting labels or adding positives."""
    if config["method"] != "legacy_v1":
        raise NotImplementedError("Implementation TBD: retrieval")
    baseline = legacy()
    candidate_rows, token_rows = [], []
    with sqlite3.connect(f"file:{index['path']}?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        frequencies = baseline.fetch_document_frequencies(db, queries)
        for row in queries.itertuples(index=False):
            merged = {}
            for route, value in (("name", row.business_name), ("address", row.business_address)):
                tokens = baseline.choose_query_tokens(value, route, frequencies)
                token_rows.append({"source1_entity_id": row.entity_id, "route": route,
                                   "tokens": "|".join(tokens)})
                for rank, (eid, source, name, address, score) in enumerate(
                    baseline.retrieve_route(db, row.country, tokens, route), 1
                ):
                    item = merged.setdefault(eid, {
                        "source1_entity_id": row.entity_id, "candidate_entity_id": eid,
                        "country": row.country, "candidate_source": source,
                        "candidate_name": name, "candidate_address": address,
                        "name_rank": float("nan"), "address_rank": float("nan"),
                        "name_bm25": float("nan"), "address_bm25": float("nan"),
                    })
                    item[f"{route}_rank"], item[f"{route}_bm25"] = rank, score
            candidate_rows.extend(merged.values())
    return _rank(pd.DataFrame(candidate_rows, columns=LEGACY_COLUMNS)), pd.DataFrame(
        token_rows, columns=["source1_entity_id", "route", "tokens"])


def cached(queries, index, directory, provenance, *, reuse_legacy=False):
    """Cache exact candidates, without labels/scores, plus the full query list."""
    query_ids = queries.entity_id.tolist()
    key = cache_key(query_ids, provenance)
    destination = Path(directory) / key
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest["key"] != key or manifest["provenance"] != provenance:
            raise ValueError("Candidate cache identity mismatch")
        for name, expected in manifest["files"].items():
            if fingerprint(destination / name) != expected:
                raise ValueError(f"Candidate cache was modified: {name}")
        return destination, {"cache_id": key, "cache_reused": True, "seconds": 0.0,
                             "original_retrieval": manifest["runtime"]}
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    chunks = []
    tokens = []
    if reuse_legacy:
        known = set(pd.read_csv(ROOT / "eda/reports/eda_data/anchors.tsv", sep="\t", usecols=["entity_id"]).entity_id)
        if not set(query_ids) <= known:
            raise ValueError("Legacy cache cannot serve new queries")
        for chunk in pd.read_csv(ROOT / "eda/reports/full_pool/candidate_pairs.tsv.gz", sep="\t",
                                 usecols=LEGACY_COLUMNS, keep_default_na=False, chunksize=50000):
            part = chunk[chunk.source1_entity_id.isin(query_ids)].copy()
            for name in ("name_rank", "address_rank", "name_bm25", "address_bm25"):
                part[name] = pd.to_numeric(part[name], errors="coerce")
            chunks.append(part)
        frame = _rank(pd.concat(chunks, ignore_index=True))
        token_frame = pd.read_csv(ROOT / "eda/reports/full_pool/query_tokens.tsv", sep="\t", keep_default_na=False)
        token_frame = token_frame[token_frame.source1_entity_id.isin(query_ids)]
        runtime = {"seconds": None, "kind": "imported saved candidates; no retrieval executed"}
    else:
        # Bounded retrieval batches; written directly to disk, never all candidates in RAM.
        for start in range(0, len(queries), 200):
            frame, token_frame = retrieve(queries.iloc[start:start + 200], index, {"method": "legacy_v1"})
            validate_pairs(frame, query_ids)
            frame.to_csv(destination / "pairs.tsv.gz", sep="\t", index=False,
                         mode="w" if start == 0 else "a", header=start == 0, compression="gzip")
            tokens.append(token_frame)
        frame = None
        token_frame = pd.concat(tokens, ignore_index=True) if tokens else pd.DataFrame(columns=["source1_entity_id", "route", "tokens"])
        runtime = {"seconds": time.perf_counter() - started, "kind": "retrieval"}
    if frame is not None:
        validate_pairs(frame, query_ids)
        frame.to_csv(destination / "pairs.tsv.gz", sep="\t", index=False, compression="gzip")
    elif not query_ids:
        pd.DataFrame(columns=[*LEGACY_COLUMNS, "rank"]).to_csv(destination / "pairs.tsv.gz", sep="\t", index=False)
    token_frame.to_csv(destination / "tokens.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": query_ids}).to_csv(destination / "queries.tsv", sep="\t", index=False)
    write_json(destination / "manifest.json", {
        "key": key, "provenance": provenance, "runtime": runtime,
        "ranking": "min(name_rank,address_rank), candidate_entity_id; legacy route union remains separate",
        "files": {n: fingerprint(destination / n) for n in ("pairs.tsv.gz", "tokens.tsv", "queries.tsv")},
    })
    return destination, {"cache_id": key, "cache_reused": reuse_legacy,
                         "seconds": time.perf_counter() - started, "original_retrieval": runtime}


def read_candidates(path):
    for chunk in pd.read_csv(Path(path) / "pairs.tsv.gz", sep="\t", keep_default_na=False, chunksize=50000):
        for column in ("rank", "name_rank", "address_rank", "name_bm25", "address_bm25"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        yield chunk


# Blocking implementation. Legacy adapters above remain unchanged.
BLOCKING_ROUTES = ("name", "address", "dense")
BLOCKING_COLUMNS = ["source1_entity_id", "candidate_entity_id", "country", "candidate_source",
                    "candidate_name", "candidate_address", "rank", "fusion_score",
                    *[f"{r}_{suffix}" for r in BLOCKING_ROUTES for suffix in ("rank", "score")],
                    *[f"in_{r}" for r in BLOCKING_ROUTES]]


class ResourceWatch:
    """Bound single-process CLI workers (native threads share this RSS).

    Native allocators cannot be capped per process on macOS. Terminate at the
    12-GiB working limit, leaving 4 GiB headroom below the requested 16 GiB.
    Partial artifacts have no completion marker and can be resumed.
    """
    def __init__(self, config):
        import threading
        self.config = config
        self.stop = threading.Event()
        self.peak = 0
        if not 0 < config["working_memory_gib"] < config["max_memory_gib"] <= 16:
            raise ValueError("Memory budgets must satisfy 0 < working < hard <= 16 GiB")

    def __enter__(self):
        import threading
        import os
        import psutil
        import fcntl
        # Do not let concurrent blocking commands each consume the whole allowance.
        (ROOT / "artifacts").mkdir(exist_ok=True)
        self.lock = (ROOT / "artifacts/.blocking.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("Another blocking workload is active; the 16-GiB allowance is shared") from None
        process = psutil.Process()
        self.peak = process.memory_info().rss
        def watch():
            while not self.stop.is_set():
                try:
                    rss = process.memory_info().rss
                    self.peak = max(self.peak, rss)
                    if rss > self.config["working_memory_gib"] * 1024 ** 3:
                        print("Blocking stopped: working memory limit reached; incomplete artifacts retained.", flush=True)
                        os._exit(75)
                except (psutil.Error, OSError):
                    print("Blocking stopped: memory monitoring failed.", flush=True)
                    os._exit(75)
                self.stop.wait(.05)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.lock.close()

    def stats(self):
        return {"peak_process_rss_bytes": self.peak, "ram_limit_gib": self.config["max_memory_gib"],
                "abort_at_gib": self.config["working_memory_gib"], "sampling_seconds": .05,
                "peak_vram_bytes": 0, "device": "cpu"}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def complete_artifact(directory, metadata):
    files = {str(p.relative_to(directory)): fingerprint(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".tmp")}
    atomic_json(directory / "manifest.json", {**metadata, "complete": True, "files": files})


def verified_artifact(directory):
    directory = Path(directory)
    value = json.loads((directory / "manifest.json").read_text())
    if value.get("complete") is not True:
        raise ValueError("Artifact is incomplete")
    for name, expected in value["files"].items():
        if fingerprint(directory / name) != expected:
            raise ValueError(f"Corrupt/incomplete artifact: {directory / name}")
    return value


def blocking_identity(config, pool_fingerprint):
    from . import data, preprocess
    if config["normalization"] != preprocess.LEXICAL_VERSION or config["serialization"] != preprocess.DENSE_VERSION:
        raise ValueError("Unknown normalization/serialization version")
    if config["encoder"]["device"] != "cpu":
        raise ValueError("This measured 16-GiB implementation supports CPU encoding only")
    if config["encoder"]["compute_precision"] != "float32" or config["encoder"]["storage_precision"] != "float16":
        raise ValueError("Unmeasured embedding precision")
    return {"config": config, "pool": pool_fingerprint,
            "code": data.inventory([ROOT / "src/er" / f for f in ("data.py", "preprocess.py", "retrieve.py")]
                                   + [ROOT / "pyproject.toml", ROOT / "uv.lock"])}


def disk_check(directory, config, expected_extra=0):
    import shutil
    size = sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
    if size + expected_extra > config["max_disk_gib"] * 1024 ** 3:
        raise ValueError("Declared disk budget exceeded")
    if shutil.disk_usage(directory).free < expected_extra + 2 * 1024 ** 3:
        raise ValueError("Insufficient free disk space")
    return size


def count_vectorizer(config, vocabulary=None):
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    return CountVectorizer(analyzer="char", ngram_range=(config["ngram_min"], config["ngram_max"]),
                           lowercase=False, dtype=np.float32, vocabulary=vocabulary)


def lexical_build(directory, batches, config, identity):
    """Two-pass, disk-sharded TF-IDF with exact global DF; no vocabulary pruning."""
    from scipy import sparse
    import numpy as np
    from sklearn.preprocessing import normalize
    from .preprocess import blocking_records
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        manifest = verified_artifact(directory)
        if manifest["identity"] != identity:
            raise ValueError("Index identity mismatch")
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    work = directory / "build.json"
    if work.exists() and json.loads(work.read_text()) != identity:
        raise ValueError("Cannot resume with different inputs/configuration")
    atomic_json(work, identity)
    started, count, shard_names = time.perf_counter(), 0, []
    for number, raw in enumerate(batches):
        shard = directory / f"shard-{number:05d}"
        shard_names.append(shard.name)
        count += len(raw)
        if (shard / "counts.json").exists():
            saved = json.loads((shard / "counts.json").read_text())
            if saved["ids_hash"] != digest(raw.entity_id.tolist()):
                raise ValueError("Resumed pool mapping changed")
            for name, expected in saved["files"].items():
                if fingerprint(shard / name) != expected:
                    raise ValueError("Partial lexical shard is corrupt")
            continue
        shard.mkdir(exist_ok=True)
        records = blocking_records(raw)
        records.to_csv(shard / "records.tsv.gz", sep="\t", index=False, compression="gzip")
        for route in ("name", "address"):
            vectorizer = count_vectorizer(config)
            texts = records[f"lexical_{route}"]
            if any(len(text) >= config["ngram_min"] for text in texts):
                matrix = vectorizer.fit_transform(texts)
                vocabulary = vectorizer.get_feature_names_out().tolist()
            else:
                matrix, vocabulary = sparse.csr_matrix((len(records), 0), dtype=np.float32), []
            sparse.save_npz(shard / f"{route}-counts.npz", matrix)
            atomic_json(shard / f"{route}-vocabulary.json", vocabulary)
        atomic_json(shard / "counts.json", {"ids_hash": digest(raw.entity_id.tolist()),
                    "files": {p.name: fingerprint(p) for p in shard.iterdir() if p.is_file() and p.name != "counts.json"}})
        disk_check(directory, config)
        print(f"Lexical count shard {number + 1}: {count:,} records", flush=True)
    if not count:
        raise ValueError("Empty searchable pool")
    # Rebuild only the global DF work file on resume, never completed indexes.
    df_path = directory / "df.sqlite3"
    if df_path.exists():
        df_path.unlink()
    with sqlite3.connect(df_path) as db:
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE TABLE df(route TEXT,term TEXT,n INTEGER NOT NULL,PRIMARY KEY(route,term)) WITHOUT ROWID")
        for name in shard_names:
            for route in ("name", "address"):
                shard = directory / name
                vocab = json.loads((shard / f"{route}-vocabulary.json").read_text())
                matrix = sparse.load_npz(shard / f"{route}-counts.npz")
                frequencies = np.bincount(matrix.indices, minlength=len(vocab))
                db.executemany("INSERT INTO df VALUES(?,?,?) ON CONFLICT(route,term) DO UPDATE SET n=n+excluded.n",
                               ((route, term, int(n)) for term, n in zip(vocab, frequencies)))
            db.commit()
        for name in shard_names:
            shard = directory / name
            for route in ("name", "address"):
                vocab = json.loads((shard / f"{route}-vocabulary.json").read_text())
                frequencies = lookup_df(db, route, vocab)
                idf = np.array([np.log((count + 1) / (frequencies[t] + 1)) + 1 for t in vocab], dtype=np.float32)
                matrix = sparse.load_npz(shard / f"{route}-counts.npz")
                weighted = normalize(matrix.multiply(idf).tocsr(), norm="l2") if len(vocab) else matrix
                sparse.save_npz(shard / f"{route}-tfidf.npz", weighted)
                np.save(shard / f"{route}-idf.npy", idf)
            print(f"Lexical weights {name}", flush=True)
    complete_artifact(directory, {"identity": identity, "pool_records": count, "shards": shard_names,
                                  "runtime": {"index_build_seconds": time.perf_counter() - started},
                                  "weighting": "raw TF; log((N+1)/(df+1))+1; L2; char ngrams; no pruning"})
    return json.loads((directory / "manifest.json").read_text())


def lookup_df(db, route, terms):
    found = {}
    for start in range(0, len(terms), 500):
        batch = terms[start:start + 500]
        found.update(db.execute(f"SELECT term,n FROM df WHERE route=? AND term IN ({','.join('?' for _ in batch)})", [route, *batch]))
    return found


def top_hits(scores, ids, k):
    """Only positive lexical cosine; include all ties before stable ID selection."""
    import numpy as np
    if not k or not len(scores):
        return []
    indices = np.flatnonzero(scores > 0)
    if len(indices) > k:
        threshold = np.partition(scores[indices], -k)[-k]
        indices = indices[scores[indices] >= threshold]
    return sorted(((str(ids[i]), float(scores[i])) for i in indices), key=lambda item: (-item[1], item[0]))[:k]


def lexical_search(directory, queries, allocation, config, manifest):
    from scipy import sparse
    import numpy as np
    from collections import Counter
    directory = Path(directory)
    output = {q: {"name": [], "address": []} for q in queries.entity_id}
    with sqlite3.connect(f"file:{directory / 'df.sqlite3'}?mode=ro", uri=True) as db:
        # Query norms use the full global vocabulary, not each shard's vocabulary.
        query_norms = {}
        analyzer = count_vectorizer(config).build_analyzer()
        for route in ("name", "address"):
            if not allocation[route]:
                continue
            norms = []
            for text in queries[f"lexical_{route}"]:
                counts = Counter(analyzer(text))
                frequencies = lookup_df(db, route, list(counts))
                norms.append(sum((counts[t] * (np.log((manifest["pool_records"] + 1) / (n + 1)) + 1)) ** 2
                                 for t, n in frequencies.items()) ** .5)
            query_norms[route] = np.array(norms)
        for shard_name in manifest["shards"]:
            shard = directory / shard_name
            ids = pd.read_csv(shard / "records.tsv.gz", sep="\t", usecols=["entity_id"], keep_default_na=False).entity_id.to_numpy()
            for route in ("name", "address"):
                k = allocation[route]
                if not k:
                    continue
                vocab = json.loads((shard / f"{route}-vocabulary.json").read_text())
                if not vocab:
                    continue
                vectorizer = count_vectorizer(config, {t: i for i, t in enumerate(vocab)})
                q = vectorizer.transform(queries[f"lexical_{route}"]).multiply(np.load(shard / f"{route}-idf.npy")).tocsr()
                norms = query_norms[route]
                q = q.multiply(np.divide(1, norms, out=np.zeros_like(norms), where=norms != 0)[:, None]).tocsr()
                matrix = sparse.load_npz(shard / f"{route}-tfidf.npz")
                # Only batch_size x shard_rows scores, never query x full_pool.
                similarities = (q @ matrix.T).tocsr()
                for i, query_id in enumerate(queries.entity_id):
                    row = similarities.getrow(i)
                    new = top_hits(row.data, ids[row.indices], k)
                    output[query_id][route] = sorted(output[query_id][route] + new, key=lambda item: (-item[1], item[0]))[:k]
    return output


def fuse_routes(query_id, routes, allocation, constant=60):
    """No refill, no union truncation; identity survives identical text."""
    import math
    union = {}
    for route in BLOCKING_ROUTES:
        hits = routes.get(route, [])
        if len(hits) > allocation[route] or len({i for i, _ in hits}) != len(hits):
            raise ValueError("Invalid route output")
        for rank, (candidate_id, score) in enumerate(hits, 1):
            if not math.isfinite(score):
                raise ValueError("Invalid route score")
            row = union.setdefault(candidate_id, {"source1_entity_id": query_id, "candidate_entity_id": candidate_id,
                                                    "fusion_score": 0.0})
            row[f"{route}_rank"], row[f"{route}_score"] = rank, score
            row["fusion_score"] += 1 / (constant + rank)
    result = sorted(union.values(), key=lambda r: (-r["fusion_score"], r["candidate_entity_id"]))
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
        for route in BLOCKING_ROUTES:
            row[f"in_{route}"] = f"{route}_rank" in row
            row.setdefault(f"{route}_rank", float("nan"))
            row.setdefault(f"{route}_score", float("nan"))
    return result


def load_encoder(config, cache_root, max_length=None):
    import os
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", config["encoder"]["revision"]):
        raise ValueError("Encoder revision must be an immutable commit SHA")
    os.environ.setdefault("HF_HOME", str(Path(cache_root).resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(config["threads"])
    model = SentenceTransformer(config["encoder"]["model"], revision=config["encoder"]["revision"],
                                cache_folder=str(cache_root), device="cpu", trust_remote_code=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.max_seq_length = max_length or config["encoder"]["max_length_cap"]
    return model


def encode_texts(model, texts, config):
    import numpy as np
    # Restore encoder parallelism after the single-threaded ANN boundary.
    if hasattr(model, "parameters"):
        import torch
        torch.set_num_threads(config["threads"])
    lengths = []
    for start in range(0, len(texts), config["encoder"]["batch_size"]):
        tokenized = model.tokenizer(texts[start:start + config["encoder"]["batch_size"]],
                                    truncation=False, padding=False, verbose=False)
        lengths.extend(map(len, tokenized["input_ids"]))
    started = time.perf_counter()
    vectors = model.encode(texts, batch_size=config["encoder"]["batch_size"], convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False) if texts else np.empty((0, 1024), dtype=np.float32)
    if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
        raise ValueError("Invalid BGE-M3 dense output")
    return vectors, {"encoding_seconds": time.perf_counter() - started, "encoded_records": len(texts),
                     "truncated_records": sum(n > model.max_seq_length for n in lengths),
                     "token_length_p50": float(np.median(lengths)) if lengths else None,
                     "token_length_p99": float(np.quantile(lengths, .99)) if lengths else None,
                     "token_length_max": max(lengths, default=0)}


def normalized_vectors(path):
    import numpy as np
    values = np.array(np.load(path, mmap_mode="r"), dtype=np.float32)
    if len(values):
        values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def train_ann(vectors, config, *, pilot=False):
    import faiss
    from threadpoolctl import threadpool_limits
    if config["ann"]["kind"] != "IVF-SQ8":
        raise ValueError("Unmeasured ANN backend")
    # macOS wheels load multiple libomp runtimes (sklearn, torch, FAISS).
    # Their multithreaded clustering deadlocks after encoding on this machine.
    threadpool_limits(limits=config["ann"]["threads"], user_api="openmp")
    faiss.omp_set_num_threads(config["ann"]["threads"])
    proposed = config["ann"]["pilot_nlist" if pilot else "nlist"]
    nlist = min(proposed, max(1, len(vectors) // 40))
    if not len(vectors):
        raise ValueError("No usable dense text in pool")
    index = faiss.IndexIVFScalarQuantizer(faiss.IndexFlatIP(vectors.shape[1]), vectors.shape[1], nlist,
                                         faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = config["seed"]
    index.nprobe = min(nlist, config["ann"]["pilot_nprobe" if pilot else "nprobe"])
    index.train(vectors)
    return index


def dense_build(directory, lexical_dir, lexical_manifest, config, model, identity, *, pilot=False):
    """Resumable embedding/ANN shards, keeping only one shard in RAM."""
    import faiss
    import numpy as np
    directory, lexical_dir = Path(directory), Path(lexical_dir)
    if (directory / "manifest.json").exists():
        value = verified_artifact(directory)
        if value["identity"] != identity:
            raise ValueError("Dense index identity mismatch")
        return value
    directory.mkdir(parents=True, exist_ok=True)
    build = {"identity": identity, "lexical_manifest": fingerprint(lexical_dir / "manifest.json"), "max_length": model.max_seq_length}
    if (directory / "build.json").exists() and json.loads((directory / "build.json").read_text()) != build:
        raise ValueError("Cannot resume embeddings with different inputs")
    atomic_json(directory / "build.json", build)
    started = time.perf_counter()
    stats, train_parts = [], []
    for name in lexical_manifest["shards"]:
        shard = directory / name
        shard.mkdir(exist_ok=True)
        mapping = lexical_dir / name / "records.tsv.gz"
        if (shard / "embeddings.json").exists():
            state = json.loads((shard / "embeddings.json").read_text())
            if state["records"] != fingerprint(mapping):
                raise ValueError("Embedding row mapping changed")
            for filename, expected in state["files"].items():
                if fingerprint(shard / filename) != expected:
                    raise ValueError("Corrupt embedding shard")
        else:
            records = pd.read_csv(mapping, sep="\t", keep_default_na=False)
            valid = records.dense_text.ne("")
            vectors, runtime = encode_texts(model, records.loc[valid, "dense_text"].tolist(), config)
            # A commit marker is written only after both vectors and row mappings.
            np.save(shard / "vectors.npy", vectors.astype(np.float16))
            records.loc[valid, ["rowid", "entity_id"]].to_csv(shard / "rows.tsv", sep="\t", index=False)
            runtime["unusable_records"] = int((~valid).sum())
            state = {"records": fingerprint(mapping), "runtime": runtime,
                     "files": {f: fingerprint(shard / f) for f in ("vectors.npy", "rows.tsv")}}
            atomic_json(shard / "embeddings.json", state)
        stats.append(state["runtime"])
        values = np.load(shard / "vectors.npy", mmap_mode="r")
        take = min(len(values), max(1, config["ann"]["train_rows"] // len(lexical_manifest["shards"])))
        if take:
            train_parts.append(np.array(values[np.linspace(0, len(values) - 1, take, dtype=int)], dtype=np.float32))
        disk_check(directory, config)
        print(f"Dense embeddings {name}: {state['runtime']['encoded_records']:,} records", flush=True)
    training = np.concatenate(train_parts) if train_parts else np.empty((0, 1024), dtype=np.float32)
    if len(training):
        training /= np.linalg.norm(training, axis=1, keepdims=True)
    ann_started = time.perf_counter()
    template = train_ann(training, config, pilot=pilot)
    del training, train_parts
    for name in lexical_manifest["shards"]:
        shard = directory / name
        index = faiss.clone_index(template)
        values = normalized_vectors(shard / "vectors.npy")
        if len(values):
            index.add(values)
        faiss.write_index(index, str(shard / "ann.faiss.tmp"))
        (shard / "ann.faiss.tmp").replace(shard / "ann.faiss")
    runtime = {k: sum(s[k] for s in stats) for k in ("encoding_seconds", "encoded_records", "truncated_records", "unusable_records")}
    runtime.update({"index_build_seconds": time.perf_counter() - ann_started, "total_seconds": time.perf_counter() - started})
    complete_artifact(directory, {**build, "shards": lexical_manifest["shards"], "pool_records": lexical_manifest["pool_records"],
                                  "max_length": model.max_seq_length, "nlist": template.nlist, "nprobe": template.nprobe,
                                  "ann_threads": config["ann"]["threads"],
                                  "ann_training": "evenly spaced unlabeled pool rows; no supervised fitting",
                                  "runtime": runtime})
    return json.loads((directory / "manifest.json").read_text())


def dense_search(directory, query_vectors, query_ids, k, manifest):
    import faiss
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=manifest["ann_threads"], user_api="openmp")
    faiss.omp_set_num_threads(manifest["ann_threads"])
    directory = Path(directory)
    results = {q: [] for q in query_ids}
    if not k:
        return results
    for name in manifest["shards"]:
        shard = directory / name
        index = faiss.read_index(str(shard / "ann.faiss"))
        index.nprobe = manifest["nprobe"]
        ids = pd.read_csv(shard / "rows.tsv", sep="\t", keep_default_na=False).entity_id.to_numpy()
        distances, neighbours = index.search(query_vectors, k)
        for q, scores, indices in zip(query_ids, distances, neighbours):
            hits = [(str(ids[i]), float(score)) for score, i in zip(scores, indices) if i >= 0]
            results[q] = sorted(results[q] + hits, key=lambda item: (-item[1], item[0]))[:k]
    return results


def dense_fidelity(directory, manifest, query_vectors, config, k=100):
    """Engineering pilot only: ANN vs exact on the identical reduced pool."""
    import faiss
    import numpy as np
    exact = faiss.IndexFlatIP(1024)
    ids = []
    for name in manifest["shards"]:
        shard = Path(directory) / name
        exact.add(normalized_vectors(shard / "vectors.npy"))
        ids.extend(pd.read_csv(shard / "rows.tsv", sep="\t", keep_default_na=False).entity_id)
    k = min(k, len(ids))
    if not len(query_vectors) or not k:
        raise ValueError("Fidelity pilot needs usable queries and candidates")
    _, exact_indices = exact.search(query_vectors, k)
    ann = dense_search(directory, query_vectors, list(range(len(query_vectors))), k, manifest)
    overlaps = [len(set(ids[i] for i in indices) & {i for i, _ in ann[q]}) / k
                for q, indices in enumerate(exact_indices)]
    return {"ann_exact_recall_at_k": float(np.mean(overlaps)), "k": k, "queries": len(query_vectors),
            "pool_records": len(ids), "scope": "same reduced pilot pool; search fidelity, not labeled recall",
            "nlist": manifest["nlist"], "nprobe": manifest["nprobe"]}


def blocking_candidates(directory, queries, lexical_dir, dense_dir, config, mode, identity, model=None):
    """Atomic cache completion and per-batch checkpoints; failures never mean zero."""
    from .preprocess import blocking_records
    from .data import pool_connection
    import numpy as np
    directory = Path(directory)
    allocation = config["allocations"][mode]
    if sum(allocation.values()) != config["budget"] or any(v < 0 for v in allocation.values()):
        raise ValueError("Route allocation must equal the declared total budget")
    records = blocking_records(queries).sort_values("entity_id").reset_index(drop=True)
    cache_identity = {"identity": identity, "mode": mode, "allocation": allocation,
                      "queries": digest(records.to_dict("records")), "ids": records.entity_id.tolist(),
                      "lexical_index": fingerprint(Path(lexical_dir) / "manifest.json")}
    lexical_manifest = verified_artifact(lexical_dir)
    dense_manifest = None
    if allocation["dense"]:
        dense_manifest = verified_artifact(dense_dir)
        cache_identity["dense_index"] = fingerprint(Path(dense_dir) / "manifest.json")
    if (directory / "manifest.json").exists():
        started = time.perf_counter()
        manifest = verified_artifact(directory)
        if manifest["identity"] != cache_identity:
            raise ValueError("Candidate cache identity mismatch")
        return manifest, {"cache_read_seconds": time.perf_counter() - started, "cache_reused": True}
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "build.json").exists() and json.loads((directory / "build.json").read_text()) != cache_identity:
        raise ValueError("Cannot resume a different candidate request")
    atomic_json(directory / "build.json", cache_identity)
    stats, shards = [], []
    for start in range(0, len(records), config["query_batch_size"]):
        batch = records.iloc[start:start + config["query_batch_size"]]
        part_dir = directory / f"batch-{start:06d}"
        shards.append(part_dir.name)
        if (part_dir / "manifest.json").exists():
            stats.append(verified_artifact(part_dir)["runtime"])
            continue
        part_dir.mkdir(exist_ok=True)
        began = time.perf_counter()
        routes = lexical_search(lexical_dir, batch, allocation, config, lexical_manifest)
        encoding_seconds = 0
        query_truncated = 0
        if allocation["dense"]:
            if model is None:
                raise ValueError("Dense encoder is required")
            usable = batch[batch.dense_text.ne("")]
            vectors, encoding = encode_texts(model, usable.dense_text.tolist(), config)
            encoding_seconds, query_truncated = encoding["encoding_seconds"], encoding["truncated_records"]
            hits = dense_search(dense_dir, vectors, usable.entity_id.tolist(), allocation["dense"], dense_manifest) if len(usable) else {}
            for q in batch.entity_id:
                routes[q]["dense"] = hits.get(q, [])
        pair_rows = [row for q in batch.entity_id for row in fuse_routes(q, routes[q], allocation, config["fusion_constant"])]
        countries = batch.set_index("entity_id").country.to_dict()
        targets = sorted({r["candidate_entity_id"] for r in pair_rows})
        originals = {}
        with pool_connection() as db:
            for offset in range(0, len(targets), 500):
                ids = targets[offset:offset + 500]
                originals.update((i, (s, n, a)) for i, s, n, a in db.execute(
                    f"SELECT entity_id,source,business_name,business_address FROM records WHERE entity_id IN ({','.join('?' for _ in ids)})", ids))
        for row in pair_rows:
            row["country"] = countries[row["source1_entity_id"]]
            row["candidate_source"], row["candidate_name"], row["candidate_address"] = originals[row["candidate_entity_id"]]
        frame = pd.DataFrame(pair_rows, columns=BLOCKING_COLUMNS)
        validate_pairs(frame, batch.entity_id)
        frame.to_csv(part_dir / "pairs.tsv.gz", sep="\t", index=False, compression="gzip")
        elapsed = time.perf_counter() - began
        runtime = {"queries": len(batch), "retrieval_seconds": elapsed - encoding_seconds,
                   "encoding_seconds": encoding_seconds, "query_truncated_records": query_truncated,
                   "amortized_query_seconds": elapsed / len(batch)}
        complete_artifact(part_dir, {"completed_queries": batch.entity_id.tolist(), "runtime": runtime})
        stats.append(runtime)
        print(f"Retrieved {min(start + len(batch), len(records))}/{len(records)} {mode} queries", flush=True)
    # Concatenate gzip members, retaining exactly one header. No in-memory full result.
    import gzip
    with gzip.open(directory / "pairs.tsv.gz.tmp", "wt") as output:
        output.write("\t".join(BLOCKING_COLUMNS) + "\n")
        for name in shards:
            with gzip.open(directory / name / "pairs.tsv.gz", "rt") as source:
                next(source)
                for line in source:
                    output.write(line)
    (directory / "pairs.tsv.gz.tmp").replace(directory / "pairs.tsv.gz")
    records.to_csv(directory / "queries.tsv.gz", sep="\t", index=False, compression="gzip")
    # Existing matcher consumes originals plus optional legacy token lists.
    pd.DataFrame(columns=["source1_entity_id", "route", "tokens"]).to_csv(directory / "tokens.tsv", sep="\t", index=False)
    runtime = {k: sum(s[k] for s in stats) for k in ("retrieval_seconds", "encoding_seconds", "query_truncated_records")}
    total = runtime["retrieval_seconds"] + runtime["encoding_seconds"]
    runtime.update({"queries_per_second": len(records) / total if total else None,
                    "p95_query_seconds": float(np.quantile([s["amortized_query_seconds"] for s in stats for _ in range(s["queries"])], .95)),
                    "latency_convention": "wall time per batch / actual batch size, including encoding and disk reads; not individual request latency",
                    "batch_size": config["query_batch_size"], "cache_read_seconds": 0})
    complete_artifact(directory, {"identity": cache_identity, "completed_queries": records.entity_id.tolist(),
                                  "shards": shards, "runtime": runtime})
    return json.loads((directory / "manifest.json").read_text()), {"cache_reused": False, "cache_read_seconds": 0}
