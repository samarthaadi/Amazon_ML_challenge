"""Protocol tests use synthetic businesses; never read final-holdout labels."""
import csv
import json
import pickle
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from er import data, decide, evaluate, retrieve


class ProtocolTests(unittest.TestCase):
    def test_official_score_and_missing_query(self):
        self.assertEqual(evaluate.f05(set(), set()), 1)
        self.assertEqual(evaluate.f05(set(), {"x"}), 0)
        self.assertEqual(evaluate.f05({"x"}, set()), 0)
        self.assertEqual(evaluate.f05({"x"}, {"y"}), 0)
        self.assertAlmostEqual(evaluate.f05({"a", "b"}, {"a", "b", "c"}), 5 / 7)
        self.assertAlmostEqual(evaluate.f05({"a", "b"}, {"a"}), 5 / 6)
        with self.assertRaises(ValueError):
            evaluate.macro_f05(["q"], {}, {"q": set()})

    def test_stages_include_zero_candidates_and_retrieval_misses(self):
        queries = {"S1-a": "India", "S1-b": "India", "S1-c": "US", "S1-d": "US"}
        truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": set(), "S1-c": {"S2-c"}, "S1-d": set()}
        candidates = pd.DataFrame([
            ("S1-a", "S2-a", 1), ("S1-a", "S2-wrong", 2), ("S1-b", "S2-wrong", 1)
        ], columns=[*data.PAIR_KEYS, "rank"])
        predictions = {"S1-a": set(), "S1-b": {"S2-wrong"}, "S1-c": set(), "S1-d": set()}
        per_query, errors = evaluate.evaluate(queries, truth, candidates, predictions, ks=(1, 2))
        metrics = evaluate.summarize(per_query)["all"]
        self.assertEqual(metrics["queries"], 4)
        self.assertEqual(metrics["retrieval"]["zero_candidate_queries"], 2)
        self.assertEqual(metrics["errors"], {"retrieval_miss": 2, "matcher_false_rejection": 1, "false_acceptance": 1})
        self.assertEqual(metrics["matching_decisions"]["singleton_false_acceptance_rate"], .5)
        self.assertEqual(metrics["end_to_end"]["macro_f0_5"], .25)
        oracle = {q: set(truth[q]) & set(candidates.loc[candidates.source1_entity_id.eq(q), "candidate_entity_id"]) for q in queries}
        self.assertAlmostEqual(metrics["retrieval"]["oracle_macro_f0_5"], evaluate.macro_f05(queries, truth, oracle))
        self.assertEqual(set(errors.error_type), {"retrieval_miss", "matcher_false_rejection", "false_acceptance"})
        predictions["S1-c"] = {"S2-c"}
        with self.assertRaises(ValueError):
            evaluate.evaluate(queries, truth, candidates, predictions)

    def test_retrieved_recall_has_different_denominator(self):
        pairs = pd.DataFrame([("S1-a", "S2-a", 1)], columns=[*data.PAIR_KEYS, "rank"])
        per_query, _ = evaluate.evaluate({"S1-a": "US"}, {"S1-a": {"S2-a", "S2-b"}}, pairs, {"S1-a": {"S2-a"}})
        summary = evaluate.summarize(per_query)["all"]
        self.assertEqual(summary["matching_decisions"]["recall_on_retrieved"], 1)
        self.assertEqual(summary["end_to_end"]["pair_recall_including_retrieval_misses"], .5)

    def test_no_candidate_decisions_and_undefined_metrics(self):
        scores = pd.DataFrame(columns=[*data.PAIR_KEYS, "score"])
        policy, _ = decide.tune(scores, {"S1-a": set()}, ["S1-a"], {"method": "legacy_v1"})
        self.assertEqual(decide.apply(scores, ["S1-a"], policy), {"S1-a": set()})
        pairs = pd.DataFrame(columns=[*data.PAIR_KEYS, "rank"])
        frame, _ = evaluate.evaluate({"S1-a": "France"}, {"S1-a": set()}, pairs, {"S1-a": set()})
        metrics = evaluate.summarize(frame)
        self.assertIsNone(metrics["all"]["matching_decisions"]["precision"])
        self.assertIsNone(metrics["all"]["retrieval"]["recall"])
        json.dumps(metrics, allow_nan=False)

    def test_cache_identity_and_pair_validation(self):
        base = {"inputs": "v1", "retrieval": "v1", "preprocessing": "v1", "pool": "full", "seed": 42}
        key = retrieve.cache_key(["b", "a"], base)
        self.assertEqual(key, retrieve.cache_key(["a", "b"], base))
        for name in base:
            self.assertNotEqual(key, retrieve.cache_key(["a", "b"], {**base, name: "changed"}))
        self.assertNotEqual(key, retrieve.cache_key(["a"], base))
        pairs = pd.DataFrame([("S1-a", "S2-a"), ("S1-a", "S2-a")], columns=data.PAIR_KEYS)
        with self.assertRaises(ValueError):
            data.validate_pairs(pairs, ["S1-a"])

    def test_cache_reuse_zero_candidates_and_tamper_detection(self):
        queries = pd.DataFrame({"entity_id": ["S1-a", "S1-b"]})
        frame = pd.DataFrame([{**{c: "" for c in retrieve.LEGACY_COLUMNS},
                               "source1_entity_id": "S1-a", "candidate_entity_id": "S2-a",
                               "rank": 1}])
        tokens = pd.DataFrame(columns=["source1_entity_id", "route", "tokens"])
        with tempfile.TemporaryDirectory() as temporary:
            with patch("er.retrieve.retrieve", return_value=(frame, tokens)) as implementation:
                path, _ = retrieve.cached(queries, {}, temporary, {"version": 1})
                same, metadata = retrieve.cached(queries, {}, temporary, {"version": 1})
                self.assertEqual(path, same)
                self.assertTrue(metadata["cache_reused"])
                implementation.assert_called_once()
            self.assertEqual(len(pd.read_csv(path / "queries.tsv", sep="\t")), 2)
            self.assertNotIn("label", pd.read_csv(path / "pairs.tsv.gz", sep="\t").columns)
            (path / "tokens.tsv").write_text("modified")
            with self.assertRaises(ValueError):
                retrieve.cached(queries, {}, temporary, {"version": 1})

    def test_freeze_is_exclusive_and_final_rejects_tampered_state(self):
        from er import run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "run"
            source.mkdir()
            splits = root / "splits"
            splits.mkdir()
            for name in ("matcher.pkl", "policy.json", "preprocessing.json"):
                (source / name).write_text("dummy trusted fixture")
            data.write_json(source / "manifest.json", {
                "status": "development feedback", "code": {"test": "v1"}, "configuration": {}})
            with patch("er.run.code_identity", return_value={"test": "v1"}), patch(
                "er.run.load_split", return_value=(splits, {})
            ):
                run.freeze(source)
                with self.assertRaises(FileExistsError):
                    run.freeze(source)
                (source / "matcher.pkl").write_text("tampered")
                with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                    run.final(source)
                (splits / "final_access_mixed.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "Holdout already accessed"):
                    run.freeze(source)

    def test_final_streams_complete_queries_and_keeps_empty_predictions(self):
        from er import run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, splits, cache = root / "run", root / "splits", root / "cache"
            for path in (source, splits, cache):
                path.mkdir()
            with sqlite3.connect(splits / "membership.sqlite3") as db:
                db.execute("CREATE TABLE queries(id,country,truth,split,transfer_eligible)")
                db.executemany("INSERT INTO queries VALUES (?,?,?,?,?)", [
                    ("S1-empty", "US", "", "holdout", 1),
                    ("S1-match", "US", "S2-true", "holdout", 1)])
            data.write_json(splits / "manifest.json", {})
            with (source / "matcher.pkl").open("wb") as stream:
                pickle.dump({}, stream)
            data.write_json(source / "preprocessing.json", {"method": "legacy_v1"})
            data.write_json(source / "policy.json", {"method": "legacy_v1", "threshold": .5})
            profile = {"train_countries": ["US"], "tune_countries": ["US"], "evaluate_countries": ["US"]}
            cfg = {"settings": {"mixed": profile}}
            data.write_json(source / "manifest.json", {
                "status": "development feedback", "code": {"fixture": "v1"}, "configuration": cfg,
                "setting": "mixed", "split_manifest": data.fingerprint(splits / "manifest.json")})
            pd.DataFrame(columns=["source1_entity_id", "route", "tokens"]).to_csv(cache / "tokens.tsv", sep="\t", index=False)
            queries = pd.DataFrame({"entity_id": ["S1-empty", "S1-match"]})
            candidates = pd.DataFrame([("S1-match", "S2-true", 1), ("S1-match", "S2-false", 2)],
                                      columns=[*data.PAIR_KEYS, "rank"])

            def score(frame, records, state, tokens):
                result = frame[data.PAIR_KEYS].copy()
                result["score"] = result.candidate_entity_id.eq("S2-true").astype(float)
                return result, {"seconds": 0, "pairs": len(result)}

            from contextlib import ExitStack
            with ExitStack() as stack:
                for target, value in {
                    "er.run.code_identity": {"fixture": "v1"},
                    "er.run.load_split": (splits, {"preserved_inputs": {}}),
                    "er.run.verify_inputs": None, "er.data.output_root": root,
                    "er.data.query_records": queries, "er.preprocess.transform": queries,
                    "er.retrieve.build_index": {"signature": {}},
                    "er.retrieve.cached": (cache, {"cache_id": "fixture"}),
                    "er.retrieve.read_candidates": iter([candidates.iloc[:1], candidates.iloc[1:]]),
                }.items():
                    stack.enter_context(patch(target, return_value=value))
                stack.enter_context(patch("er.match.score", side_effect=score))
                tune = stack.enter_context(patch("er.decide.tune", side_effect=AssertionError("Final must not tune")))
                run.freeze(source)
                run.final(source)
                tune.assert_not_called()
                with self.assertRaises(FileExistsError):
                    run.final(source)
            result_dir = next((root / "runs").iterdir())
            metrics = json.loads((result_dir / "metrics.json").read_text())
            self.assertEqual(metrics["all"]["queries"], 2)
            self.assertEqual(metrics["all"]["end_to_end"]["macro_f0_5"], 1)
            predictions = pd.read_csv(result_dir / "predictions.tsv", sep="\t", keep_default_na=False)
            self.assertEqual(predictions.set_index("source1_entity_id").loc["S1-empty", "matched_entity_ids"], "")

    def test_splits_reproducibility_grouping_and_label_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, truth = root / "source.tsv", root / "truth.tsv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["entity_id", "country"])
                writer.writerows((f"S1-{i}", "India" if i % 2 == 0 else "US") for i in range(400))
            with truth.open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(["source1_entity_id", "matched_entity_ids"])
                writer.writerows((f"S1-{i}", "S2-shared" if i in (0, 1) else (f"S2-{i}" if i % 3 else "")) for i in range(400))
            historical = [
                {"entity_id": "S1-0", "group_id": "S1-0", "split": "train"},
                {"entity_id": "S1-1", "group_id": "S1-1", "split": "validation"},
                {"entity_id": "S1-2", "group_id": "S1-2", "split": "test"},
            ]
            cfg = {"split_version": "test", "seed": 42, "train_fraction": .8,
                   "validation_fraction": .15, "holdout_fraction": .05}
            before = data.fingerprint(source), data.fingerprint(truth)
            for name in ("one", "two"):
                data.prepare_splits(root / name, source, truth, cfg, historical, {"S1-3"})
            self.assertEqual((root / "one/membership.tsv").read_bytes(), (root / "two/membership.tsv").read_bytes())
            self.assertEqual(before, (data.fingerprint(source), data.fingerprint(truth)))
            members = pd.read_csv(root / "one/membership.tsv", sep="\t").set_index("source1_entity_id")
            self.assertEqual(members.loc["S1-0", "group_id"], members.loc["S1-1", "group_id"])
            self.assertEqual(members.loc["S1-0", "split"], "validation")
            self.assertEqual(members.loc["S1-2", "split"], "validation")
            self.assertNotEqual(members.loc["S1-3", "split"], "holdout")
            self.assertEqual(members.loc["S1-0", "transfer_eligible"], 0)
            profile = {"train_countries": ["India"], "tune_countries": ["India"], "evaluate_countries": ["US"]}
            store = data.SplitStore(root / "one", profile)
            fitting, tuning, evaluation = (store.queries(role) for role in ("fit", "tune", "evaluate"))
            self.assertTrue(fitting and tuning and evaluation)
            self.assertNotIn("S1-0", tuning)
            self.assertEqual(set(store.labels(fitting, "fit")), set(fitting))
            with self.assertRaises(ValueError):
                store.labels(evaluation, "tune")
            with self.assertRaises(ValueError):
                store.labels(tuning, "fit")
            final = data.SplitStore(root / "one", profile, final=True)
            heldout = final.queries("evaluate")
            self.assertTrue(heldout)
            with self.assertRaises(ValueError):
                store.labels(heldout, "evaluate")
            with self.assertRaises(ValueError):
                final.labels(heldout, "fit")
            self.assertEqual(set(final.labels(heldout, "evaluate")), set(heldout))
            store.close()
            final.close()



class BlockingMetricTests(unittest.TestCase):
    """Frozen blocking-v2 hand-calculated fixtures; exclude from blocker edits."""
    def fixture(self):
        queries = {"S1-a": "India", "S1-b": "US", "S1-c": "India", "S1-d": "US"}
        truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": {"S2-b"}, "S1-c": set(), "S1-d": {"S3-d"}}
        pairs = pd.DataFrame([
            ("S1-a", "S2-x", 1, 1., float("nan"), float("nan")),
            ("S1-a", "S2-a", 2, float("nan"), float("nan"), 1.),
            ("S1-c", "S2-x", 1, 1., float("nan"), float("nan")),
            ("S1-d", "S2-x", 1, 1., float("nan"), float("nan")),
            ("S1-d", "S3-d", 2, float("nan"), 1., 1.),
        ], columns=[*data.PAIR_KEYS, "rank", "name_rank", "address_rank", "dense_rank"])
        return queries, truth, pairs

    def call(self, queries, truth, pairs, **kwargs):
        return evaluate.blocking(queries, truth, pairs, completed_queries=list(queries),
                                 known_ids={"S2-a", "S3-a", "S2-b", "S3-d", "S2-x"}, budget=3,
                                 route_budgets={"name": 1, "address": 1, "dense": 1}, ks=(1, 2), **kwargs)

    def test_hand_calculated_contract(self):
        queries, truth, pairs = self.fixture()
        metrics, per, misses = self.call(queries, truth, pairs)
        group = metrics["groups"]["all"]
        self.assertAlmostEqual(group["ceiling"], (5 / 6 + 0 + 1 + 1) / 4)
        self.assertAlmostEqual(group["matched_ceiling"], (5 / 6 + 0 + 1) / 3)
        self.assertEqual(group["micro_recall"], .5)
        self.assertEqual(group["macro_recall"], .5)
        self.assertAlmostEqual(group["complete_recall"], 1 / 3)
        self.assertAlmostEqual(group["any_match_coverage"], 2 / 3)
        self.assertEqual(group["no_match_prevalence"], .25)
        self.assertEqual(group["candidate_cost"]["zero_rate"], .25)
        self.assertEqual(group["prefixes"]["1"]["micro_recall"], 0)
        self.assertEqual(group["prefixes"]["2"]["micro_recall"], .5)
        self.assertEqual(group["marginal"]["dense"]["unique_hits"], 1)
        self.assertEqual(group["marginal"]["dense"]["recall_percentage_points"], 25)
        self.assertEqual(group["sources"]["S2"]["true_links"], 2)
        self.assertEqual(len(misses), 2)
        oracle = {q: truth[q] & set(pairs.loc[pairs.source1_entity_id.eq(q), "candidate_entity_id"]) for q in queries}
        self.assertAlmostEqual(group["ceiling"], evaluate.macro_f05(queries, truth, oracle))
        for row in per.itertuples():
            expected = 1 if not row.true_links else 1.25 * row.hits / (.25 * row.true_links + row.hits)
            self.assertAlmostEqual(row.ceiling, expected)
        # Unequal truth counts demonstrate that macro and micro are distinct.
        truth["S1-a"].add("S2-b")
        changed = self.call(queries, truth, pairs)[0]["groups"]["all"]
        self.assertEqual(changed["micro_recall"], 2 / 5)
        self.assertAlmostEqual(changed["macro_recall"], (1 / 3 + 0 + 1) / 3)

    def test_undefined_and_no_match_oracle(self):
        q, t, p = self.fixture()
        metrics, _, _ = self.call({"S1-c": "India"}, {"S1-c": set()}, p[p.source1_entity_id.eq("S1-c")])
        row = metrics["groups"]["all"]
        self.assertEqual(row["ceiling"], 1)
        self.assertIsNone(row["micro_recall"])
        self.assertIsNone(row["macro_recall"])
        self.assertIsNone(row["complete_recall"])
        self.assertEqual(row["true_links"], 0)
        self.assertEqual(row["matched_queries"], 0)
        with self.assertRaises(ValueError):
            self.call({}, {}, p.iloc[:0])

    def test_invalid_artifacts_are_not_silently_repaired(self):
        q, t, p = self.fixture()
        variants = [pd.concat([p, p.iloc[:1]], ignore_index=True),
                    p.assign(rank=0), p.assign(rank=1.5), p.assign(rank=float("inf")),
                    p.assign(candidate_entity_id="S2-unknown"), p.assign(source1_entity_id="S1-unknown"),
                    p.assign(label=0), p.assign(name_rank=4)]
        for broken in variants:
            with self.subTest(columns=list(broken)), self.assertRaises(ValueError):
                self.call(q, t, broken)
        with self.assertRaises(ValueError):
            evaluate.blocking(q, t, p, completed_queries=list(q)[:-1], known_ids=set(), budget=3)
        with self.assertRaises(ValueError):
            evaluate.blocking(q, t, p, completed_queries=list(q), known_ids=lambda ids: True, budget=1)

    def test_missing_fields_cannot_overwrite_authoritative_metrics(self):
        q, t, p = self.fixture()
        safe = {i: {"missing_name": False, "missing_address": True} for i in q}
        baseline, before, _ = self.call(q, t, p)
        metrics, after, _ = self.call(q, t, p, missing_fields=safe)
        pd.testing.assert_frame_equal(before, after[before.columns])
        self.assertEqual(metrics["groups"]["all"], baseline["groups"]["all"])
        for key, value in (("hits", 999), ("true_links", 0), ("ceiling", 1),
                           ("country", "US"), ("source1_entity_id", "S1-other"),
                           ("hits_at_1", 99), ("candidates", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.call(q, t, p, missing_fields={**safe, "S1-a": {**safe["S1-a"], key: value}})
        for bad_value in ("False", None, 1):
            with self.subTest(value=bad_value), self.assertRaises(ValueError):
                self.call(q, t, p, missing_fields={**safe, "S1-a": {"missing_name": bad_value, "missing_address": False}})
        for incomplete in ({}, {"S1-a": safe["S1-a"]}, {**safe, "S1-unknown": safe["S1-a"]},
                           {**safe, "S1-a": {"missing_name": False}}):
            with self.assertRaises(ValueError):
                self.call(q, t, p, missing_fields=incomplete)


class BlockingRetrievalTests(unittest.TestCase):
    def config(self):
        import copy
        result = copy.deepcopy(data.config()["blocking"])
        result.update(shard_rows=2, query_batch_size=2)
        return result

    def records(self):
        return pd.DataFrame([
            (1, "S2-a", "S2", "India", "Alpha Shop", "1 Main Road"),
            (2, "S2-b", "S2", "US", "Alpha Shop", "1 Main Road"),
            (3, "S3-c", "S3", "India", "Beta Cafe", "2 Park Street"),
            (4, "S3-d", "S3", "US", "", ""),
        ], columns=["rowid", "entity_id", "source", "country", "business_name", "business_address"])

    def test_missing_text_and_serialization(self):
        from er import preprocess
        frame = self.records()
        frame.loc[3, "business_name"] = float("nan")
        result = preprocess.blocking_records(frame)
        self.assertEqual(result.dense_text.iloc[3], "")
        self.assertTrue(result.missing_name.iloc[3])
        self.assertEqual(result.dense_text.iloc[0], "Name: Alpha Shop Address: 1 Main Road")
        self.assertEqual(result.business_name.iloc[0], frame.business_name.iloc[0])

    def test_sharded_tfidf_matches_brute_force_global_cosine(self):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from er import preprocess
        cfg, raw = self.config(), self.records()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = retrieve.lexical_build(directory, [raw.iloc[:2], raw.iloc[2:]], cfg, {"test": "v1"})
            queries = preprocess.blocking_records(pd.DataFrame([
                ("S1-q", "India", "Alpha Beta", "Main Park"), ("S1-empty", "US", "", "")
            ], columns=["entity_id", "country", "business_name", "business_address"]))
            hits = retrieve.lexical_search(directory, queries, {"name": 4, "address": 4}, cfg, manifest)
            records = preprocess.blocking_records(raw)
            for route in ("name", "address"):
                vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, dtype=np.float32)
                matrix = vectorizer.fit_transform(records[f"lexical_{route}"])
                expected = (vectorizer.transform(queries[f"lexical_{route}"]) @ matrix.T).toarray()[0]
                expected_hits = retrieve.top_hits(expected, raw.entity_id.to_numpy(), 4)
                self.assertEqual([i for i, _ in hits["S1-q"][route]], [i for i, _ in expected_hits])
                np.testing.assert_allclose([s for _, s in hits["S1-q"][route]], [s for _, s in expected_hits], rtol=1e-6)
                self.assertEqual(hits["S1-empty"][route], [])
            self.assertEqual(manifest["pool_records"], 4)
            self.assertEqual(retrieve.lexical_build(directory, [], cfg, {"test": "v1"}), manifest)
            # Cache content hashes reject damaged row-to-ID mappings.
            (directory / "shard-00000/records.tsv.gz").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                retrieve.verified_artifact(directory)

    def test_union_deduplicates_ids_preserves_identical_text_records(self):
        rows = retrieve.fuse_routes("S1-q", {"name": [("S2-a", .9), ("S2-b", .9)],
                                            "address": [("S2-b", .8)], "dense": [("S3-c", .7)]},
                                   {"name": 2, "address": 1, "dense": 1})
        self.assertEqual([r["candidate_entity_id"] for r in rows], ["S2-b", "S2-a", "S3-c"])
        self.assertAlmostEqual(rows[0]["fusion_score"], 1 / 62 + 1 / 61)
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        with self.assertRaises(ValueError):
            retrieve.fuse_routes("q", {"name": [("S2-a", .9), ("S2-a", .8)]}, {"name": 2, "address": 0, "dense": 0})

    def test_memory_budget_cannot_exceed_user_limit(self):
        cfg = self.config()
        retrieve.ResourceWatch(cfg)
        cfg["max_memory_gib"] = 17
        with self.assertRaises(ValueError):
            retrieve.ResourceWatch(cfg)

    def test_dense_shards_exact_diagnostic_and_resume(self):
        import numpy as np
        from er import preprocess
        cfg, raw = self.config(), self.records()
        class Encoder:
            max_seq_length = 32
            def tokenizer(self, texts, **kwargs):
                return {"input_ids": [[1] * len(t.split()) for t in texts]}
            def encode(self, texts, **kwargs):
                values = np.zeros((len(texts), 1024), dtype=np.float32)
                for row, text in enumerate(texts):
                    values[row, 0 if "Alpha" in text else 1] = 1
                return values
        with tempfile.TemporaryDirectory() as temp:
            lexical_dir, dense_dir = Path(temp) / "lexical", Path(temp) / "dense"
            lexical = retrieve.lexical_build(lexical_dir, [raw.iloc[:2], raw.iloc[2:]], cfg, {"test": 1})
            encoder = Encoder()
            manifest = retrieve.dense_build(dense_dir, lexical_dir, lexical, cfg, encoder, {"test": 1}, pilot=True)
            self.assertEqual(manifest["runtime"]["encoded_records"], 3)
            self.assertEqual(manifest["runtime"]["unusable_records"], 1)
            self.assertEqual(manifest["pool_records"], 4)
            vectors = encoder.encode(["Alpha", "Beta"])
            result = retrieve.dense_search(dense_dir, vectors, ["S1-a", "S1-b"], 3, manifest)
            self.assertEqual(len(result["S1-a"]), 3)
            diagnostic = retrieve.dense_fidelity(dense_dir, manifest, vectors, cfg, k=3)
            self.assertEqual(diagnostic["ann_exact_recall_at_k"], 1)
            self.assertEqual(retrieve.dense_build(dense_dir, lexical_dir, lexical, cfg, encoder, {"test": 1}, pilot=True), manifest)

    def test_candidate_cache_completion_and_matcher_schema(self):
        from er import match
        cfg, raw = self.config(), self.records()
        cfg["budget"] = 4
        cfg["allocations"]["lexical"] = {"name": 2, "address": 2, "dense": 0}
        queries = pd.DataFrame([("S1-q", "India", "Alpha Shop", "1 Main Road"), ("S1-empty", "US", "", "")],
                               columns=["entity_id", "country", "business_name", "business_address"])
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            db_path = temp / "pool.sqlite3"
            with sqlite3.connect(db_path) as db:
                raw.to_sql("records", db, index=False)
            def connect():
                return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            lexical_dir = temp / "lexical"
            retrieve.lexical_build(lexical_dir, [raw], cfg, {"test": 1})
            cache = temp / "candidates"
            with patch("er.data.pool_connection", connect):
                manifest, runtime = retrieve.blocking_candidates(cache, queries, lexical_dir, None, cfg, "lexical", {"test": 1})
                reused, second = retrieve.blocking_candidates(cache, queries, lexical_dir, None, cfg, "lexical", {"test": 1})
            self.assertEqual(manifest, reused)
            self.assertTrue(second["cache_reused"])
            self.assertEqual(set(manifest["completed_queries"]), set(queries.entity_id))
            pairs = pd.read_csv(cache / "pairs.tsv.gz", sep="\t")
            self.assertEqual(set(pairs.candidate_entity_id), {"S2-a", "S2-b"})
            self.assertNotIn("label", pairs)
            tokens = pd.read_csv(cache / "tokens.tsv", sep="\t")
            features = match.features(pairs, queries, tokens, {"method": "legacy_v1"})
            self.assertEqual(len(features), 2)
            with patch("er.data.pool_connection", connect), self.assertRaises(ValueError):
                retrieve.blocking_candidates(cache, queries.iloc[:1], lexical_dir, None, cfg, "lexical", {"test": 1})
            # A failed shard cannot be read as an empty-candidate success.
            (cache / "batch-000000/pairs.tsv.gz").unlink()
            with self.assertRaises(FileNotFoundError):
                retrieve.verified_artifact(cache)

    def test_blocking_run_evaluates_every_query_without_fitting(self):
        from er import run
        cfg, raw = self.config(), self.records()
        cfg["budget"] = 4
        cfg["allocations"]["lexical"] = {"name": 2, "address": 2, "dense": 0}
        queries = pd.DataFrame([("S1-q", "India", "Alpha Shop", "1 Main Road"), ("S1-empty", "US", "", "")],
                               columns=["entity_id", "country", "business_name", "business_address"])
        countries = queries.set_index("entity_id").country.to_dict()
        truth = {"S1-q": {"S2-a", "S3-c"}, "S1-empty": set()}
        class Store:
            def queries(self, purpose, ids):
                return {q: countries[q] for q in ids}
            def labels(self, ids, purpose):
                return {q: truth[q] for q in ids}
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            db_path = temp / "pool.sqlite3"
            with sqlite3.connect(db_path) as db:
                raw.to_sql("records", db, index=False)
            def connect():
                return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            lexical_dir, cache, run_dir = temp / "lexical", temp / "cache", temp / "run"
            run_dir.mkdir()
            retrieve.lexical_build(lexical_dir, [raw], cfg, {"test": 1})
            with patch("er.data.pool_connection", connect):
                retrieve.blocking_candidates(cache, queries, lexical_dir, None, cfg, "lexical", {"test": 1})
            configuration = data.config()
            configuration.update(artifact_root=str(temp), blocking=cfg)
            data.write_json(temp / "manifest.json", {"split": "fixture"})
            manifest = {"configuration": configuration, "setting": "mixed", "purpose": "evaluate", "mode": "lexical",
                        "queries": countries, "scope": "synthetic fixture", "runtime": {}, "evaluator": {"version": "fixture"},
                        "candidate_cache": str(cache), "candidate_manifest": data.fingerprint(cache / "manifest.json"),
                        "split_manifest": data.fingerprint(temp / "manifest.json"), "query_list_hash": data.digest(sorted(countries))}
            data.write_json(run_dir / "manifest.json", manifest)
            with patch("er.run.blocking_evaluator", return_value={"version": "fixture"}), \
                 patch("er.run.load_split", return_value=(temp, {})), patch("er.run.verify_inputs"), \
                 patch("er.data.pool_connection", connect), \
                 patch("er.match.fit", side_effect=AssertionError("Blocking must never fit a matcher")), \
                 patch("er.decide.tune", side_effect=AssertionError("Blocking must never tune decisions")):
                run.blocking_evaluate(run_dir, final_store=Store())
            metrics = json.loads((run_dir / "blocking_metrics.json").read_text())
            self.assertAlmostEqual(metrics["groups"]["all"]["ceiling"], (5 / 6 + 1) / 2)
            self.assertEqual(metrics["groups"]["all"]["queries"], 2)
            self.assertEqual(metrics["groups"]["missing_name=True"]["queries"], 1)
            misses = pd.read_csv(run_dir / "missed_matches.tsv.gz", sep="\t")
            self.assertEqual(misses.candidate_entity_id.tolist(), ["S3-c"])

    def test_blocking_freeze_requires_tuning_and_unopened_holdout(self):
        from er import run
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            cfg = data.config()
            cfg["artifact_root"] = str(temp)
            manifest = {"configuration": cfg, "status": "development feedback", "selection_eligible": False,
                        "code": {}, "candidate_cache": str(temp / "cache")}
            for name in ("manifest.json", "blocking_metrics.json", "comparisons.json", "per_query.tsv"):
                (temp / name).write_text("fixture")
            with patch("er.run.blocking_evaluator", return_value={}), patch("er.run.code_identity", return_value={}), \
                 patch("er.run.load_split", return_value=(temp, {})), patch("er.retrieve.verified_artifact", return_value={}):
                with self.assertRaises(ValueError):
                    run.blocking_freeze(temp, manifest)
                manifest["selection_eligible"] = True
                (temp / "final_access_mixed.json").write_text("{}")
                with self.assertRaises(ValueError):
                    run.blocking_freeze(temp, manifest)
                (temp / "final_access_mixed.json").unlink()
                run.blocking_freeze(temp, manifest)
                self.assertTrue((temp / "frozen.json").exists())
                with self.assertRaises(FileExistsError):
                    run.blocking_freeze(temp, manifest)


if __name__ == "__main__":
    unittest.main()
