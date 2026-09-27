"""Small, reproducible ER analysis. Run with: uv run python eda/analysis/eda_first_pass.py.

Raw TSVs are streamed; cached samples live in eda/reports/eda_data/. This is an EDA
sample, not a matching pipeline or a full-corpus retrieval benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import unicodedata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/amazon-eda-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as sparse_normalize

EDA_ROOT = Path(__file__).resolve().parents[1]
ROOT = EDA_ROOT.parent
OUT = EDA_ROOT / "reports"
CACHE = OUT / "eda_data"
SEED = 42
CHUNK = 100_000
N_ANCHORS = 2_000
N_DISTRACTORS = 25_000
KINDS = ["true_match", "random_nonmatch", "hard_nonmatch"]
TITLES = ["True matches", "Random same-country nonmatches", "Harder sampled nonmatches"]
LEGAL = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited",
    "pvt": "private", "co": "company", "svc": "services", "svcs": "services",
}
STOP_NAME = set(LEGAL) | set(LEGAL.values()) | {"llc", "llp", "the", "and"}
STOP_ADDRESS = {
    "road", "street", "lane", "avenue", "drive", "floor", "flat", "plot",
    "near", "opp", "opposite", "block", "district", "india", "the", "and",
}


def read_tsv(path, **kwargs):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kwargs)


def clean(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    if text.isascii():
        return re.sub(r"[\W_]+", " ", text).strip()
    # Keep combining marks: deleting them corrupts Devanagari and other scripts.
    text = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text)
    return " ".join(text.split())


def expanded(value: str) -> str:
    return " ".join(LEGAL.get(t, t) for t in clean(value.replace("&", " and ")).split())


BASELINE_NORMALIZATION = "baseline_v1"
TARGETED_NORMALIZATION = "targeted_cleanup_v2"
CLEANUP_ABLATIONS = {
    "url_only": frozenset({"url"}),
    "dotted_suffix_only": frozenset({"dotted_suffix"}),
    "leading_zero_only": frozenset({"leading_zero"}),
    "numeric_separator_only": frozenset({"numeric_separator"}),
}
ALL_CLEANUP_RULES = frozenset().union(*CLEANUP_ABLATIONS.values())
# Explicit prefixes accept any domain suffix. Bare domains use a conservative
# suffix list; this avoids interpreting ordinary dotted initials as websites.
WEBSITE = re.compile(
    r"(?<![\w@])(?:https?://(?:www\.)?|www\.)"
    r"(?P<prefixed>[\w-]+(?:\.[\w-]+)+)(?::\d+)?(?:[/\?#][^\s<>\[\](){}]*)?"
    r"|(?<![\w@])(?P<bare>[\w-]+(?:\.[\w-]+)*"
    r"\.(?:com|org|net|edu|gov|biz|info|io|in|fr|uk|us))"
    r"(?![\w.-])(?:[/\?#][^\s<>\[\](){}]*)?",
    re.IGNORECASE,
)
DOTTED_LEGAL = re.compile(
    r"(?<!\w)(?:p\s*\.\s*l\s*\.\s*l\s*\.\s*c|"
    r"l\s*\.\s*l\s*\.\s*[cp]|p\s*\.\s*v\s*\.\s*t|"
    r"p\s*\.\s*c)(?:\.)?(?!\w)"
)


def clean_targeted(value: str, *, field="address", changes=None, expand_ampersand=False, rules=None) -> str:
    """Derived cleanup only; baseline/index normalization remains unchanged.

    Dot/comma components after the first are preserved to avoid changing
    decimal fractions or guessing a locale. Hyphen/slash integer components
    normalize independently. Decimal digits keep their original script.
    """
    if field not in {"name", "address"}:
        raise ValueError(f"Unknown cleanup field: {field}")
    enabled = ALL_CLEANUP_RULES if rules is None else frozenset(rules)
    if not enabled <= ALL_CLEANUP_RULES:
        raise ValueError(f"Unknown cleanup rules: {enabled - ALL_CLEANUP_RULES}")
    if not enabled:
        return clean(value.replace("&", " and ") if expand_ampersand else value)
    def step(rule, before, after):
        if changes is not None:
            changes[rule] = before != after
        return after

    text = step("unicode_normalization", value, unicodedata.normalize("NFKC", value))
    text = step("casefold", text, text.casefold())
    if changes is not None:
        changes.update(website=False, dotted_legal=False)
    if field == "name" and "url" in enabled:
        before = text
        matches = list(WEBSITE.finditer(text))
        remainder = WEBSITE.sub(" ", text)
        if matches and not any(unicodedata.category(c)[0] in "LMN" for c in remainder):
            stems = []
            for match in matches:
                labels = (match.group("prefixed") or match.group("bare")).split(".")
                # Handle common multi-part suffixes without a downloaded list.
                cut = 2 if len(labels) >= 3 and labels[-2] in {"co", "com", "org", "net", "ac", "gov"} and len(labels[-1]) == 2 else 1
                stems.append(labels[-cut - 1])
            text = " ".join(stems)
        else:
            text = remainder
        text = step("website", before, text)
    if field == "name" and "dotted_suffix" in enabled:
        text = step("dotted_legal", text, DOTTED_LEGAL.sub(lambda m: re.sub(r"[.\s]", "", m.group()), text))
    if expand_ampersand:
        text = text.replace("&", " and ")
    # Only canonicalize dash variants in numeric context. Other symbols become
    # boundaries; letters are never concatenated by punctuation deletion.
    if "numeric_separator" in enabled:
        text = step("numeric_separator_variants", text, re.sub(r"(?<=\d)\s*([/\-\u2010-\u2015\u2212])\s*(?=\d)", lambda m: "/" if m.group(1) == "/" else "-", text))

    def normalize_integer(match):
        start = match.start()
        if start >= 2 and text[start - 1] in ".," and text[start - 2].isdecimal():
            return match.group()  # fractional or ambiguous locale component
        digits = match.group()
        offset = 0
        while offset < len(digits) - 1 and unicodedata.decimal(digits[offset]) == 0:
            offset += 1
        return digits[offset:]

    # Normalize while decimal punctuation is still available, even when the
    # separator-preservation rule is disabled for a one-rule ablation.
    if "leading_zero" in enabled:
        text = step("leading_zeroes", text, re.sub(r"\d+", normalize_integer, text))
    if "numeric_separator" not in enabled:
        text = step("punctuation_symbols", text, clean(text))
    else:
        chars = []
        if changes is not None:
            changes["numeric_separator_preservation"] = False
        for i, c in enumerate(text):
            # Single-rule variants retain baseline character eligibility.
            keep = (unicodedata.category(c)[0] in "LM" or c.isdecimal()) if rules is None else (c.isalnum() or unicodedata.category(c).startswith("M"))
            numeric_separator = c in "-/.,:" and i > 0 and i + 1 < len(text) and text[i - 1].isdecimal() and text[i + 1].isdecimal()
            if changes is not None and numeric_separator:
                changes["numeric_separator_preservation"] = True
            chars.append(c if keep or numeric_separator else " ")
        text = step("punctuation_symbols", text, "".join(chars))
    return step("whitespace", text, " ".join(text.split()))


def expanded_targeted(value: str, *, rules=None) -> str:
    text = clean_targeted(value, field="name", expand_ampersand=True, rules=rules)
    return " ".join(LEGAL.get(token, token) for token in text.split())


def core_name(value: str) -> str:
    return " ".join(t for t in expanded(value).split() if t not in STOP_NAME)


def number_status(left: str, right: str, strip_zeroes=False) -> str:
    # Digit sequences are intentionally untyped; e.g. 1-101 becomes {1, 101}.
    a = set(re.findall(r"\d+", unicodedata.normalize("NFKC", left)))
    b = set(re.findall(r"\d+", unicodedata.normalize("NFKC", right)))
    if strip_zeroes:
        a = {str(int(t)) for t in a}
        b = {str(int(t)) for t in b}
    if not a and not b:
        return "neither_has_numbers"
    if not a or not b:
        return "one_side_has_no_numbers"
    if a == b:
        return "same_numeric_tokens"
    return "partial_overlap" if a & b else "disjoint_numeric_tokens"


def save_tsv(frame: pd.DataFrame, name: str):
    frame.to_csv(CACHE / name, sep="\t", index=False)


def fingerprint():
    return {
        str(p.relative_to(ROOT)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted((ROOT / "dataset").rglob("*.tsv"))
    }


def smallest(frame, n):
    return frame.nsmallest(n, "_priority") if len(frame) > n else frame


def build_sample(refresh=False):
    paths = ["anchors.tsv", "records.tsv", "truth.tsv", "strata.tsv", "branch_candidates.tsv"]
    manifest = CACHE / "manifest.json"
    current = {"seed": SEED, "anchors": N_ANCHORS, "distractors": N_DISTRACTORS,
               "files": fingerprint(), "sample_version": 1}
    if not refresh and manifest.exists() and json.loads(manifest.read_text()) == current:
        if all((CACHE / p).exists() for p in paths):
            print("Using validated cached sample.", flush=True)
            return
    print("Scanning labels and sampling reference entities...", flush=True)
    singles = set()
    gt_path = ROOT / "dataset/train/train_ground_truth.tsv"
    for chunk in read_tsv(gt_path, chunksize=CHUNK):
        singles.update(chunk.loc[chunk.matched_entity_ids.eq(""), "source1_entity_id"])
    rng = np.random.default_rng(SEED)
    reservoirs, counts = {}, Counter()
    source1_path = ROOT / "dataset/train/train_source1.tsv"
    for chunk in read_tsv(source1_path, chunksize=CHUNK):
        chunk["singleton"] = chunk.entity_id.isin(singles)
        chunk["_priority"] = rng.random(len(chunk))
        for key, part in chunk.groupby(["country", "singleton"], sort=True):
            counts[key] += len(part)
            prior = reservoirs.get(key)
            local = smallest(part, N_ANCHORS)
            reservoirs[key] = smallest(pd.concat([prior, local], ignore_index=True), N_ANCHORS)
    keys = sorted(counts)
    exact = {k: N_ANCHORS * counts[k] / sum(counts.values()) for k in keys}
    quotas = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - quotas[k]), k))[:N_ANCHORS - sum(quotas.values())]:
        quotas[k] += 1
    anchors = pd.concat([smallest(reservoirs[k], quotas[k]) for k in keys], ignore_index=True)
    anchors = anchors.drop(columns="_priority").sort_values("entity_id").reset_index(drop=True)
    strata = pd.DataFrame([
        {"country": k[0], "singleton": k[1], "population": counts[k], "sample": quotas[k]}
        for k in keys
    ])
    wanted_refs = set(anchors.entity_id)
    labels = []
    for chunk in read_tsv(gt_path, chunksize=CHUNK):
        labels.extend(chunk[chunk.source1_entity_id.isin(wanted_refs)].to_dict("records"))
    assert len(labels) == N_ANCHORS, "Missing or duplicate sampled label rows"
    truth_rows = []
    match_counts = {}
    for row in labels:
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        assert len(ids) == len(set(ids)), "Repeated match ID within sampled label"
        assert all(x.startswith(("S2-", "S3-")) for x in ids)
        match_counts[row["source1_entity_id"]] = len(ids)
        truth_rows.extend((row["source1_entity_id"], eid) for eid in ids)
    truth = pd.DataFrame(truth_rows, columns=["source1_entity_id", "candidate_entity_id"])
    wanted_matches = set(truth.candidate_entity_id)
    anchors["match_count"] = anchors.entity_id.map(match_counts)
    assert (anchors.singleton == anchors.match_count.eq(0)).all()
    print(f"Selected {len(anchors):,} references and {len(truth):,} true links.", flush=True)
    found, reservoir = [], None
    rng = np.random.default_rng(SEED + 1)
    for source in (2, 3):
        print(f"Streaming Source {source}: true records and random distractors...", flush=True)
        for chunk in read_tsv(ROOT / f"dataset/train/train_source{source}.tsv", chunksize=CHUNK):
            hit = chunk.entity_id.isin(wanted_matches)
            found.append(chunk[hit])
            part = chunk[~hit].copy()
            part["_priority"] = rng.random(len(part))
            reservoir = smallest(pd.concat([reservoir, smallest(part, N_DISTRACTORS)], ignore_index=True), N_DISTRACTORS)
    matched = pd.concat(found, ignore_index=True)
    assert len(matched) == len(wanted_matches), "Missing or duplicate sampled target ID"
    assert set(matched.entity_id) == wanted_matches
    distractors = reservoir.drop(columns="_priority").copy()
    assert len(distractors) == N_DISTRACTORS
    assert not set(distractors.entity_id) & wanted_matches
    records = pd.concat([
        anchors.assign(sample_role="reference"),
        matched.assign(sample_role="matched_record"),
        distractors.assign(sample_role="distractor"),
    ], ignore_index=True)[["entity_id", "business_name", "business_address", "country", "sample_role"]]
    assert records.entity_id.is_unique

    # Small targeted reference-name scan supplies branch/co-location evidence.
    # Only names with at least two meaningful tokens are considered; cap examples.
    print("Checking repeated distinctive reference names for branch examples...", flush=True)
    name_to_refs = defaultdict(list)
    for row in anchors.itertuples():
        key = core_name(row.business_name)
        if len(key.split()) >= 2:
            name_to_refs[(row.country, key)].append(row.entity_id)
    branch_rows, kept = [], Counter()
    for chunk in read_tsv(source1_path, chunksize=CHUNK):
        normalized = chunk.business_name.map(core_name)
        for row, core in zip(chunk.itertuples(), normalized):
            key = (row.country, core)
            if key not in name_to_refs:
                continue
            for ref in name_to_refs[key]:
                if row.entity_id != ref and kept[ref] < 3:
                    branch_rows.append({"source1_entity_id": ref, "other_reference_id": row.entity_id,
                                        "other_name": row.business_name, "other_address": row.business_address,
                                        "country": row.country, "core_name": core})
                    kept[ref] += 1
    branches = pd.DataFrame(branch_rows, columns=["source1_entity_id", "other_reference_id", "other_name", "other_address", "country", "core_name"])
    for frame, path in zip([anchors, records, truth, strata, branches], paths):
        save_tsv(frame, path)
    manifest.write_text(json.dumps(current, indent=2) + "\n")
    print(f"Cached {len(records):,} records and {len(branches):,} repeated-name reference examples.", flush=True)


def trigram_matrix(values):
    # Count cosine, not TF-IDF: corpus frequency cannot change an individual pair's score.
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False, dtype=np.float32)
    return sparse_normalize(vectorizer.fit_transform(values), norm="l2", copy=False)


def paired_cos(matrix, left, right):
    return np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel().clip(0, 1)


def analyze():
    records = read_tsv(CACHE / "records.tsv")
    anchors = read_tsv(CACHE / "anchors.tsv")
    truth = read_tsv(CACHE / "truth.tsv")
    index = {eid: i for i, eid in enumerate(records.entity_id)}
    record_ids = records.entity_id.to_numpy()
    names = records.business_name.map(clean)
    addresses = records.business_address.map(clean)
    print("Computing trigram similarities and sampled hard nonmatches...", flush=True)
    matrices = {
        "name_similarity": trigram_matrix(names),
        "address_similarity": trigram_matrix(addresses),
        "raw_name_similarity": trigram_matrix(records.business_name),
        "expanded_name_similarity": trigram_matrix(records.business_name.map(expanded)),
    }
    positives = defaultdict(set)
    for row in truth.itertuples():
        positives[row.source1_entity_id].add(row.candidate_entity_id)
    pool = records[records.sample_role.ne("reference")]
    distractors = records[records.sample_role.eq("distractor")]
    same_country = {c: d.index.to_numpy() for c, d in distractors.groupby("country")}
    rng = np.random.default_rng(SEED + 2)
    pair_rows = [(r.source1_entity_id, r.candidate_entity_id, "true_match") for r in truth.itertuples()]
    for row in anchors.itertuples():
        j = rng.choice(same_country[row.country])
        pair_rows.append((row.entity_id, record_ids[j], "random_nonmatch"))

    postings = defaultdict(list)
    for j, row in pool.iterrows():
        for field, text, stop in [("name", names[j], STOP_NAME), ("address", addresses[j], STOP_ADDRESS)]:
            for token in set(text.split()) - stop:
                if len(token) >= 3:
                    postings[(row.country, field, token)].append(j)
    country_sizes = pool.country.value_counts().to_dict()
    postings = {key: js for key, js in postings.items() if len(js) <= max(1, int(country_sizes[key[0]] * 0.01))}
    candidate_counts = []
    for row in anchors.itertuples():
        i = index[row.entity_id]
        candidate_ids = set()
        for field, text in [("name", names[i]), ("address", addresses[i])]:
            for token in set(text.split()):
                candidate_ids.update(postings.get((row.country, field, token), []))
        js = np.array(sorted(j for j in candidate_ids if record_ids[j] not in positives[row.entity_id]), dtype=int)
        candidate_counts.append({"source1_entity_id": row.entity_id, "available_hard_nonmatches": len(js)})
        if not len(js):
            continue
        # Rank by the mean of available fields; a missing address is not scored as zero.
        ns = paired_cos(matrices["name_similarity"], np.full(len(js), i), js)
        ads = paired_cos(matrices["address_similarity"], np.full(len(js), i), js)
        has_address = bool(addresses[i]) & addresses.iloc[js].ne("").to_numpy()
        score = np.where(has_address, (ns + ads) / 2, ns)
        ranked = sorted(range(len(js)), key=lambda k: (-float(score[k]), record_ids[js[k]]))[:5]
        pair_rows.extend((row.entity_id, record_ids[js[k]], "hard_nonmatch") for k in ranked)
    pairs = pd.DataFrame(pair_rows, columns=["source1_entity_id", "candidate_entity_id", "pair_type"])
    left = pairs.source1_entity_id.map(index).to_numpy()
    right = pairs.candidate_entity_id.map(index).to_numpy()
    for feature, matrix in matrices.items():
        pairs[feature] = paired_cos(matrix, left, right)
    pairs["country"] = records.country.iloc[left].to_numpy()
    pairs["candidate_source"] = records.entity_id.iloc[right].str[:2].to_numpy()
    pairs["reference_is_singleton"] = ~pairs.source1_entity_id.isin(set(truth.source1_entity_id))
    for side, indices in [("reference", left), ("candidate", right)]:
        pairs[f"{side}_name"] = records.business_name.iloc[indices].to_numpy()
        pairs[f"{side}_address"] = records.business_address.iloc[indices].to_numpy()
    pairs["missing_address"] = pairs.reference_address.str.strip().eq("") | pairs.candidate_address.str.strip().eq("")
    pairs.loc[pairs.missing_address, "address_similarity"] = np.nan
    pairs["numeric_status"] = [number_status(a, b) for a, b in zip(pairs.reference_address, pairs.candidate_address)]
    pairs["numeric_status_without_leading_zeroes"] = [number_status(a, b, strip_zeroes=True) for a, b in zip(pairs.reference_address, pairs.candidate_address)]
    pairs["exact_normalized_address"] = [bool(clean(a)) and clean(a) == clean(b) for a, b in zip(pairs.reference_address, pairs.candidate_address)]
    pairs["exact_normalized_name"] = [bool(clean(a)) and clean(a) == clean(b) for a, b in zip(pairs.reference_name, pairs.candidate_name)]
    true_keys = set(zip(truth.source1_entity_id, truth.candidate_entity_id))
    neg = pairs[pairs.pair_type.ne("true_match")]
    assert not true_keys & set(zip(neg.source1_entity_id, neg.candidate_entity_id))
    assert pairs[["source1_entity_id", "candidate_entity_id", "pair_type"]].duplicated().sum() == 0
    assert len(pairs[pairs.pair_type.eq("true_match")]) == len(truth)
    assert (records.country.iloc[left].to_numpy() == records.country.iloc[right].to_numpy()).all()
    assert pairs[pairs.pair_type.eq("hard_nonmatch")].groupby("source1_entity_id").size().max() <= 5
    assert pairs.loc[pairs.missing_address, "address_similarity"].isna().all()
    assert pairs.loc[~pairs.missing_address, "address_similarity"].between(0, 1).all()
    save_tsv(pairs, "pairs.tsv")
    save_tsv(pd.DataFrame(candidate_counts), "hard_candidate_counts.tsv")
    summarize(pairs, anchors, records)
    plot(pairs)
    select_examples(pairs)
    print(f"Validated and saved {len(pairs):,} pairs.", flush=True)


def summarize(pairs, anchors, records):
    positives = pairs[pairs.pair_type.eq("true_match")].copy()
    grouped_rows = []
    for (country, source), d in positives.groupby(["country", "candidate_source"]):
        grouped_rows.append({"country": country, "candidate_source": source, "true_links": len(d),
                             "missing_address": int(d.missing_address.sum()),
                             "name_below_0_3": int(d.name_similarity.lt(0.3).sum()),
                             "disjoint_numbers": int(d.numeric_status.eq("disjoint_numeric_tokens").sum())})
    save_tsv(pd.DataFrame(grouped_rows), "positive_breakdown.tsv")
    numeric = positives.groupby(["country", "numeric_status"]).size().rename("count").reset_index()
    save_tsv(numeric, "numeric_counts.tsv")
    records = records.assign(source=records.entity_id.str[:2], missing_address=records.business_address.str.strip().eq(""))
    quality = records.groupby(["sample_role", "country", "source"]).agg(records=("entity_id", "size"), missing_addresses=("missing_address", "sum")).reset_index()
    save_tsv(quality, "sample_quality.tsv")
    def rate(mask):
        count = int(mask.sum())
        return {"count": count, "denominator": len(positives), "fraction": count / len(positives)}
    normalized_equal = positives.exact_normalized_name
    expanded_equal = pd.Series([expanded(a) == expanded(b) for a, b in zip(positives.reference_name, positives.candidate_name)], index=positives.index)
    metrics = {
        "seed": SEED, "anchors": len(anchors), "sampled_singletons": int(anchors.match_count.astype(int).eq(0).sum()),
        "true_links": len(positives), "pair_counts": pairs.pair_type.value_counts().to_dict(),
        "numeric_status": positives.numeric_status.value_counts().to_dict(),
        "numeric_status_without_leading_zeroes": positives.numeric_status_without_leading_zeroes.value_counts().to_dict(),
        "missing_positive_address": rate(positives.missing_address),
        "exact_raw_name": rate(positives.reference_name.eq(positives.candidate_name)),
        "exact_normalized_name": rate(normalized_equal),
        "exact_expanded_name": rate(expanded_equal),
        "abbreviation_only_additional": rate(expanded_equal & ~normalized_equal),
        "low_name_similarity": rate(positives.name_similarity.lt(0.3)),
        "low_name_high_address": rate(positives.name_similarity.lt(0.3) & positives.address_similarity.gt(0.8)),
        "high_name_low_address": rate(positives.name_similarity.gt(0.8) & positives.address_similarity.lt(0.3)),
        "exact_address_negative_counts": pairs[pairs.pair_type.ne("true_match") & pairs.exact_normalized_address].pair_type.value_counts().to_dict(),
        "short_name_without_trigrams": int(records.business_name.map(clean).str.len().lt(3).sum()),
        "short_nonempty_address_without_trigrams": int((records.business_address.map(clean).str.len().between(1, 2)).sum()),
    }
    (CACHE / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def plot(pairs):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharex=True, sharey=True, layout="constrained")
    highlights = []
    true = pairs[pairs.pair_type.eq("true_match") & ~pairs.missing_address]
    choices = [
        true[true.name_similarity.lt(0.3) & true.address_similarity.gt(0.8)].sort_values(["address_similarity", "name_similarity"], ascending=[False, True]).head(1),
        true[true.name_similarity.gt(0.8) & true.address_similarity.lt(0.3)].sort_values(["name_similarity", "address_similarity"], ascending=[False, True]).head(1),
    ]
    for ax, kind, title in zip(axes, KINDS, TITLES):
        d = pairs[pairs.pair_type.eq(kind)]
        valid = d[~d.missing_address]
        if len(valid):
            mesh = ax.hexbin(valid.name_similarity, valid.address_similarity, gridsize=30, extent=(0, 1, 0, 1), mincnt=1, bins="log", cmap="viridis")
            fig.colorbar(mesh, ax=ax, label="Pairs per hexagon (log scale)", shrink=0.8)
        ax.set(title=f"{title}\nn={len(valid):,}; missing address={int(d.missing_address.sum()):,}", xlabel="Name similarity", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        if kind == "true_match":
            for letter, subset in zip(["A", "B"], choices):
                if subset.empty:
                    continue
                row = subset.iloc[0]
                ax.scatter([row.name_similarity], [row.address_similarity], facecolors="none", edgecolors="#ef4444", s=100, linewidths=1.8)
                ax.annotate(letter, (row.name_similarity, row.address_similarity), xytext=(8, -13), textcoords="offset points", color="#dc2626", weight="bold")
                highlights.append({"marker": letter, **row.to_dict()})
    axes[0].set_ylabel("Address similarity")
    fig.suptitle("Business entity resolution: names versus addresses\nCharacter-trigram count cosine; sampled negatives are not production prevalence", fontsize=14)
    fig.savefig(OUT / "name_address_similarity.png", dpi=160)
    plt.close(fig)
    save_tsv(pd.DataFrame(highlights), "plot_highlights.tsv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), sharex=True, layout="constrained")
    for ax, kind, title in zip(axes, KINDS, TITLES):
        d = pairs[pairs.pair_type.eq(kind) & pairs.missing_address]
        ax.hist(d.name_similarity, bins=np.linspace(0, 1, 21), color="#4d75a4")
        ax.set(title=f"{title}\nmissing address n={len(d):,}", xlabel="Name similarity", xlim=(0, 1), ylabel="Pair count")
    fig.suptitle("Missing-address pairs: name evidence only")
    fig.savefig(OUT / "missing_address_similarity.png", dpi=160)
    plt.close(fig)


def select_examples(pairs):
    selected, used = [], set()
    true = pairs[pairs.pair_type.eq("true_match")]
    hard = pairs[pairs.pair_type.eq("hard_nonmatch")]
    def add(category, ordered, n=1):
        if n <= 0:
            return 0
        added = 0
        for _, row in ordered.iterrows():
            key = (row.source1_entity_id, row.candidate_entity_id)
            if key in used:
                continue
            used.add(key)
            selected.append({"category": category, **row.to_dict()})
            added += 1
            if added == n:
                break
        return added
    add("true_matches", true[true.name_similarity.lt(0.3) & true.address_similarity.gt(0.8)].sort_values(["address_similarity", "name_similarity"], ascending=[False, True]))
    add("true_matches", true[true.name_similarity.gt(0.8) & true.address_similarity.lt(0.3)].sort_values(["name_similarity", "address_similarity"], ascending=[False, True]))
    add("true_matches", true[true.numeric_status_without_leading_zeroes.eq("disjoint_numeric_tokens")].sort_values("name_similarity", ascending=False))
    add("true_matches", true[true.numeric_status.eq("disjoint_numeric_tokens") & true.numeric_status_without_leading_zeroes.eq("same_numeric_tokens")].sort_values("name_similarity", ascending=False))
    equivalent = pd.Series([expanded(a) == expanded(b) for a,b in zip(true.reference_name, true.candidate_name)], index=true.index)
    add("true_matches", true[equivalent & ~true.exact_normalized_name].sort_values("name_similarity"))
    add("true_matches", true.sort_values("name_similarity", ascending=False), 5 - sum(x["category"] == "true_matches" for x in selected))
    single = hard[hard.reference_is_singleton].copy()
    single["score"] = single[["name_similarity", "address_similarity"]].mean(axis=1)
    add("singletons", single[single.country.eq("US") & ~single.missing_address].sort_values("score", ascending=False).drop_duplicates("source1_entity_id"), 1)
    add("singletons", single[single.country.eq("India") & ~single.missing_address].sort_values("score", ascending=False).drop_duplicates("source1_entity_id"), 2)
    add("singletons", single[single.missing_address].sort_values("score", ascending=False).drop_duplicates("source1_entity_id"), 2)
    add("confusing_names", hard.sort_values(["name_similarity", "address_similarity"], ascending=[False, False]), 3)
    add("confusing_names", true.sort_values("name_similarity"), 2)
    # Two exact shared-address positives, then exact/near-shared negative cases.
    add("shared_addresses", true[true.exact_normalized_address].sort_values("name_similarity"), 2)
    shared = hard[hard.exact_normalized_address].sort_values("name_similarity", ascending=False)
    n = add("shared_addresses", shared, 3)
    if n < 3:
        add("shared_addresses", hard[~hard.missing_address].sort_values("address_similarity", ascending=False), 3 - n)
    add("missing_fields", true[true.missing_address].sort_values("name_similarity", ascending=False), 2)
    add("missing_fields", true[true.missing_address].sort_values("name_similarity"), 2)
    add("missing_fields", hard[hard.missing_address].sort_values("name_similarity", ascending=False), 1)
    examples = pd.DataFrame(selected)
    notes_path = EDA_ROOT / "analysis/review_notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    examples["review_note"] = [notes.get(f"{r.category}|{r.source1_entity_id}|{r.candidate_entity_id}", "Pending manual review") for r in examples.itertuples()]
    assert examples.category.value_counts().to_dict() == {c: 5 for c in ["true_matches", "singletons", "confusing_names", "shared_addresses", "missing_fields"]}
    save_tsv(examples, "reviewed_examples.tsv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Rebuild the sample by rescanning training TSVs")
    parser.add_argument("--review-only", action="store_true", help="Apply review notes to existing pair outputs")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.review_only:
        pairs = pd.read_csv(CACHE / "pairs.tsv", sep="\t", keep_default_na=False)
        pairs["address_similarity"] = pd.to_numeric(pairs.address_similarity, errors="coerce")
        select_examples(pairs)
    else:
        build_sample(args.refresh)
        analyze()


if __name__ == "__main__":
    main()
