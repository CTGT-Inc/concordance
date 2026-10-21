"""Tests for compare_runs.py. Run from the repo root:

    python -m unittest discover -s compare_calls/tests
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import compare_runs as cr  # noqa: E402

EXAMPLES = HERE.parent / "examples"


def call(seq: int, stage: str, user: str, answer: str, system: str = "Extract rules.") -> dict:
    return {
        "sequence": seq,
        "stage": stage,
        "request": {
            "model": "toy-model",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 1.0,
        },
        "response": {"content": answer, "finish_reason": "stop"},
    }


class RunBuilder:
    """Writes run directories into a temporary root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, run_id: str, calls: list[dict], stages: dict[str, object] | None = None) -> str:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        with (run_dir / "calls.jsonl").open("w", encoding="utf-8") as handle:
            for item in calls:
                handle.write(json.dumps({"run_id": run_id, **item}) + "\n")
        if stages:
            (run_dir / "stages").mkdir()
            for name, artifact in stages.items():
                (run_dir / "stages" / f"{name}.json").write_text(json.dumps(artifact), encoding="utf-8")
        return str(run_dir)

    def write_raw(self, run_id: str, text: str) -> str:
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "calls.jsonl").write_text(text, encoding="utf-8")
        return str(run_dir)


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.runs = RunBuilder(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()


class IdenticalRequestGrouping(TempCase):
    def test_same_request_in_every_run_is_one_repeated_hash(self) -> None:
        dirs = [
            self.runs.write(r, [call(1, "interpret", "Lights off at 18:00.", '["lights off"]')])
            for r in ("run_1", "run_2", "run_3")
        ]
        rep = cr.analyze(dirs)["repeated_requests"]
        self.assertEqual(rep["distinct_requests"], 1)
        self.assertEqual(rep["repeated_across_runs"], 1)
        self.assertEqual(rep["repeated_in_all_runs"], 1)
        self.assertEqual(rep["repeated_with_changed_output"], 0)
        self.assertEqual(rep["seen_in_one_run_only"], 0)

    def test_key_order_and_extra_record_keys_do_not_change_identity(self) -> None:
        a = call(1, "s", "x", "y")
        b = call(1, "s", "x", "y")
        b["request"] = dict(reversed(list(b["request"].items())))
        b["captured_at"] = "ignored"
        dirs = [self.runs.write("run_1", [a]), self.runs.write("run_2", [b])]
        self.assertEqual(cr.analyze(dirs)["repeated_requests"]["repeated_across_runs"], 1)

    def test_request_in_two_of_three_runs(self) -> None:
        dirs = [
            self.runs.write("run_1", [call(1, "s", "shared", "a")]),
            self.runs.write("run_2", [call(1, "s", "shared", "a")]),
            self.runs.write("run_3", [call(1, "s", "other", "a")]),
        ]
        rep = cr.analyze(dirs)["repeated_requests"]
        self.assertEqual(rep["repeated_across_runs"], 1)
        self.assertEqual(rep["repeated_in_all_runs"], 0)
        self.assertEqual(rep["seen_in_one_run_only"], 1)


class ChangedOutput(TempCase):
    def test_same_request_different_response_is_reported(self) -> None:
        dirs = [
            self.runs.write("run_1", [call(1, "s", "same", "answer A")]),
            self.runs.write("run_2", [call(1, "s", "same", "answer B")]),
            self.runs.write("run_3", [call(1, "s", "same", "answer A")]),
        ]
        rep = cr.analyze(dirs)["repeated_requests"]
        self.assertEqual(rep["repeated_with_changed_output"], 1)
        changed = rep["changed"][0]
        self.assertEqual(changed["distinct_responses"], 2)
        self.assertEqual([o["call_id"] for o in changed["occurrences"]], ["run_1-1", "run_2-1", "run_3-1"])


class NearIdentical(TempCase):
    def _pair(self, user_a: str, user_b: str) -> list[dict]:
        dirs = [
            self.runs.write("run_1", [call(1, "interpret", user_a, "r")]),
            self.runs.write("run_2", [call(1, "interpret", user_b, "r")]),
        ]
        return cr.analyze(dirs, max_diff_lines=50)["near_identical"]

    def test_one_line_changed(self) -> None:
        body = "\n".join(f"Rule line {i}." for i in range(30))
        near = self._pair(f"Section id: aaaa\n{body}", f"Section id: bbbb\n{body}")
        self.assertEqual(len(near), 1)
        self.assertEqual(near[0]["differing_lines"], 1)
        self.assertEqual(near[0]["matched_by"], "position")
        self.assertTrue(near[0]["response_identical"])
        changes = [l for l in near[0]["diff"] if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        self.assertEqual(changes, ["-Section id: aaaa", "+Section id: bbbb"])

    def test_two_lines_changed_and_one_removed(self) -> None:
        lines = [f"Line {i}." for i in range(20)]
        other = list(lines)
        other[3] = "Line 3 changed."
        other[12] = "Line 12 changed."
        del other[18]
        near = self._pair("\n".join(lines), "\n".join(other))
        self.assertEqual(near[0]["differing_lines"], 3)

    def test_diff_is_truncated_at_max_lines(self) -> None:
        a = "\n".join(f"id-{i} a" for i in range(10)) + "\n" + "\n".join(f"same {i}" for i in range(40))
        b = "\n".join(f"id-{i} b" for i in range(10)) + "\n" + "\n".join(f"same {i}" for i in range(40))
        dirs = [
            self.runs.write("run_1", [call(1, "s", a, "r")]),
            self.runs.write("run_2", [call(1, "s", b, "r")]),
        ]
        near = cr.analyze(dirs, max_diff_lines=4)["near_identical"][0]
        changes = [l for l in near["diff"] if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        self.assertEqual(len(changes), 4)
        self.assertTrue(near["diff_truncated"])
        self.assertEqual(near["differing_lines"], 10)

    def test_reordered_calls_pair_by_similarity(self) -> None:
        texts = [
            "\n".join([f"Section id: {sid}", f"Title: {title}", *[f"{title} rule {i}" for i in range(8)]])
            for sid, title in (("x1", "Kitchen"), ("x2", "Parking"))
        ]
        moved = [t.replace("x1", "y1").replace("x2", "y2") for t in texts]
        dirs = [
            self.runs.write("run_1", [call(1, "s", texts[0], "a"), call(2, "s", texts[1], "b")]),
            self.runs.write("run_2", [call(1, "s", moved[1], "b"), call(2, "s", moved[0], "a")]),
        ]
        near = cr.analyze(dirs)["near_identical"]
        pairs = {(n["baseline_call_id"], n["call_id"]): n for n in near}
        self.assertEqual(set(pairs), {("run_1-1", "run_2-2"), ("run_1-2", "run_2-1")})
        for item in near:
            self.assertEqual(item["matched_by"], "similarity")
            self.assertEqual(item["differing_lines"], 1)

    def test_reordered_calls_with_shared_template_are_not_mispaired(self) -> None:
        template = [f"Instruction line {i}." for i in range(20)]

        def prompt(sid: str, topic: str) -> str:
            return "\n".join([*template, f"Section id: {sid}", *[f"{topic} rule {i}" for i in range(4)]])

        dirs = [
            self.runs.write("run_1", [call(1, "s", prompt("x1", "Kitchen"), "a"), call(2, "s", prompt("x2", "Parking"), "b")]),
            self.runs.write("run_2", [call(1, "s", prompt("y2", "Parking"), "b"), call(2, "s", prompt("y1", "Kitchen"), "a")]),
        ]
        near = cr.analyze(dirs)["near_identical"]
        self.assertEqual(
            {(n["baseline_call_id"], n["call_id"]) for n in near}, {("run_1-1", "run_2-2"), ("run_1-2", "run_2-1")}
        )
        self.assertTrue(all(n["differing_lines"] == 1 for n in near))

    def test_identical_requests_are_not_near_identical(self) -> None:
        dirs = [self.runs.write(r, [call(1, "s", "same", "r")]) for r in ("run_1", "run_2")]
        result = cr.analyze(dirs)
        self.assertEqual(result["near_identical"], [])
        self.assertEqual(result["unpaired"], [])

    def test_unrelated_requests_are_unpaired(self) -> None:
        dirs = [
            self.runs.write("run_1", [call(1, "s", "\n".join(f"alpha {i}" for i in range(12)), "r")]),
            self.runs.write("run_2", [call(1, "s", "\n".join(f"omega {i}" for i in range(12)), "r")]),
        ]
        result = cr.analyze(dirs)
        self.assertEqual(result["near_identical"], [])
        self.assertEqual(result["unpaired"][0]["baseline_only"], ["run_1-1"])
        self.assertEqual(result["unpaired"][0]["run_only"], ["run_2-1"])


class FirstDivergentStage(TempCase):
    def _calls(self) -> list[dict]:
        # Stage names are chosen so alphabetical order differs from pipeline order.
        return [call(1, "zeta", "z", "z"), call(2, "beta", "b", "b"), call(3, "alpha", "a", "a")]

    def test_first_divergent_follows_call_sequence(self) -> None:
        same = {"zeta": {"v": 1}, "beta": {"v": 1}, "alpha": {"v": 1}}
        differ = {"zeta": {"v": 1}, "beta": {"v": 2}, "alpha": {"v": 2}}
        dirs = [
            self.runs.write("run_1", self._calls(), same),
            self.runs.write("run_2", self._calls(), same),
            self.runs.write("run_3", self._calls(), differ),
        ]
        stages = cr.analyze(dirs)["stages"]
        self.assertEqual(stages["order"], ["zeta", "beta", "alpha"])
        self.assertEqual(stages["first_divergent"], "beta")
        self.assertEqual(stages["diff"]["run_a"], "run_1")
        self.assertEqual(stages["diff"]["run_b"], "run_3")
        self.assertIn('-  "v": 1', stages["diff"]["lines"])
        self.assertIn('+  "v": 2', stages["diff"]["lines"])

    def test_no_divergence(self) -> None:
        same = {"zeta": [1], "beta": [2], "alpha": [3]}
        dirs = [self.runs.write(r, self._calls(), same) for r in ("run_1", "run_2")]
        stages = cr.analyze(dirs)["stages"]
        self.assertIsNone(stages["first_divergent"])
        self.assertIsNone(stages["diff"])

    def test_missing_artifact_counts_as_divergence(self) -> None:
        dirs = [
            self.runs.write("run_1", self._calls(), {"zeta": 1, "beta": 1}),
            self.runs.write("run_2", self._calls(), {"zeta": 1}),
        ]
        stages = cr.analyze(dirs)["stages"]
        self.assertEqual(stages["first_divergent"], "beta")
        self.assertIsNone(stages["hashes"]["beta"]["run_2"])

    def test_stage_order_override(self) -> None:
        dirs = [
            self.runs.write("run_1", self._calls(), {"zeta": 1, "beta": 1, "alpha": 1}),
            self.runs.write("run_2", self._calls(), {"zeta": 2, "beta": 1, "alpha": 2}),
        ]
        stages = cr.analyze(dirs, stage_order_override=["alpha"])["stages"]
        self.assertEqual(stages["order"], ["alpha", "zeta", "beta"])
        self.assertEqual(stages["first_divergent"], "alpha")


class NormalizeRegex(TempCase):
    def _dirs(self) -> list[str]:
        return [
            self.runs.write(
                "run_1",
                [call(1, "s", "Section id: 3f9a2c71d0be4e58\nText.", "rule for 3f9a2c71d0be4e58")],
                {"commit": {"3f9a2c71d0be4e58": ["rule"]}},
            ),
            self.runs.write(
                "run_2",
                [call(1, "s", "Section id: c04b7e19a8f25d36\nText.", "rule for c04b7e19a8f25d36")],
                {"commit": {"c04b7e19a8f25d36": ["rule"]}},
            ),
        ]

    def test_without_normalisation_ids_differ(self) -> None:
        result = cr.analyze(self._dirs(), final="commit")
        self.assertEqual(result["repeated_requests"]["repeated_across_runs"], 0)
        self.assertEqual(len(result["near_identical"]), 1)
        self.assertFalse(result["final"]["all_agree"])
        self.assertIsNone(result["repeated_requests"]["without_normalization"])

    def test_masked_ids_make_requests_responses_and_artifacts_identical(self) -> None:
        result = cr.analyze(self._dirs(), final="commit", normalize=[r"[0-9a-f]{16}"])
        rep = result["repeated_requests"]
        self.assertEqual(rep["repeated_across_runs"], 1)
        self.assertEqual(rep["repeated_with_changed_output"], 0)
        self.assertEqual(rep["without_normalization"]["repeated_across_runs"], 0)
        self.assertEqual(result["near_identical"], [])
        self.assertTrue(result["final"]["all_agree"])

    def test_colliding_normalised_keys_are_kept(self) -> None:
        normalizer = cr.Normalizer([r"\d+"])
        self.assertEqual(normalizer.value({"a1": 1, "a2": 2}), {"a<normalized>": 1, "a<normalized>#2": 2})

    def test_invalid_pattern_is_an_input_error(self) -> None:
        with self.assertRaises(cr.InputError) as ctx:
            cr.analyze(self._dirs(), normalize=["("])
        self.assertIn("--normalize-regex", str(ctx.exception))


class FinalStage(TempCase):
    def test_all_runs_agree(self) -> None:
        dirs = [self.runs.write(r, [call(1, "commit", "c", "c")], {"commit": {"rules": [1, 2]}}) for r in ("run_1", "run_2")]
        final = cr.analyze(dirs, final="commit")["final"]
        self.assertTrue(final["all_agree"])
        self.assertEqual(final["distinct"], 1)

    def test_runs_disagree(self) -> None:
        dirs = [
            self.runs.write("run_1", [call(1, "commit", "c", "c")], {"commit": {"rules": [1, 2]}}),
            self.runs.write("run_2", [call(1, "commit", "c", "c")], {"commit": {"rules": [1]}}),
            self.runs.write("run_3", [call(1, "commit", "c", "c")], {"commit": {"rules": [1, 2]}}),
        ]
        final = cr.analyze(dirs, final="commit")["final"]
        self.assertFalse(final["all_agree"])
        self.assertEqual(final["distinct"], 2)
        self.assertEqual(final["hashes"]["run_1"], final["hashes"]["run_3"])

    def test_artifact_key_order_does_not_matter(self) -> None:
        dirs = [
            self.runs.write("run_1", [call(1, "s", "c", "c")], {"commit": {"a": 1, "b": 2}}),
            self.runs.write("run_2", [call(1, "s", "c", "c")], {"commit": {"b": 2, "a": 1}}),
        ]
        self.assertTrue(cr.analyze(dirs, final="commit")["final"]["all_agree"])

    def test_missing_in_one_run_is_disagreement(self) -> None:
        dirs = [
            self.runs.write("run_1", [call(1, "s", "c", "c")], {"commit": {}}),
            self.runs.write("run_2", [call(1, "s", "c", "c")], {"other": {}}),
        ]
        final = cr.analyze(dirs, final="commit")["final"]
        self.assertFalse(final["all_agree"])

    def test_unknown_final_stage_is_an_error(self) -> None:
        dirs = [self.runs.write(r, [call(1, "s", "c", "c")], {"commit": {}}) for r in ("run_1", "run_2")]
        with self.assertRaises(cr.InputError) as ctx:
            cr.analyze(dirs, final="comit")
        self.assertIn("stages found: commit", str(ctx.exception))

    def test_not_requested(self) -> None:
        dirs = [self.runs.write(r, [call(1, "s", "c", "c")]) for r in ("run_1", "run_2")]
        self.assertIsNone(cr.analyze(dirs)["final"])


class MalformedInput(TempCase):
    def _error(self, text: str) -> str:
        bad = self.runs.write_raw("run_1", text)
        good = self.runs.write("run_2", [call(1, "s", "c", "c")])
        with self.assertRaises(cr.InputError) as ctx:
            cr.analyze([bad, good])
        return str(ctx.exception)

    def test_missing_key_names_line_and_key(self) -> None:
        ok = json.dumps({"run_id": "run_1", **call(1, "s", "c", "c")})
        broken = {"run_id": "run_1", **call(2, "s", "c", "c")}
        del broken["stage"]
        message = self._error(ok + "\n" + json.dumps(broken) + "\n")
        self.assertIn("calls.jsonl:2:", message)
        self.assertIn("'stage'", message)

    def test_invalid_json_names_line(self) -> None:
        message = self._error('{"run_id": "run_1",\n')
        self.assertIn("calls.jsonl:1: invalid JSON", message)

    def test_wrong_type(self) -> None:
        record = {"run_id": "run_1", **call(1, "s", "c", "c")}
        record["sequence"] = "1"
        self.assertIn("key 'sequence' must be an integer", self._error(json.dumps(record)))

    def test_missing_response_content(self) -> None:
        record = {"run_id": "run_1", **call(1, "s", "c", "c")}
        record["response"] = {"finish_reason": "stop"}
        self.assertIn("'response.content'", self._error(json.dumps(record)))

    def test_duplicate_sequence(self) -> None:
        line = json.dumps({"run_id": "run_1", **call(1, "s", "c", "c")})
        self.assertIn("already used on line 1", self._error(line + "\n" + line + "\n"))

    def test_duplicate_run_ids_across_directories(self) -> None:
        a = self.runs.write("run_1", [call(1, "s", "c", "c")])
        b_dir = self.runs.root / "copy"
        b_dir.mkdir()
        (b_dir / "calls.jsonl").write_text(Path(a, "calls.jsonl").read_text())
        with self.assertRaises(cr.InputError) as ctx:
            cr.analyze([a, str(b_dir)])
        self.assertIn("distinct run_id", str(ctx.exception))

    def test_cli_reports_error_and_exit_code(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = cr.main([str(self.runs.root / "nope"), str(self.runs.root / "nope2")])
        self.assertEqual(code, 2)
        self.assertIn("error:", stderr.getvalue())

    def test_call_id_default(self) -> None:
        dirs = [self.runs.write(r, [call(7, "s", "c", "c")]) for r in ("run_1", "run_2")]
        rep = cr.analyze(dirs)
        self.assertEqual(rep["repeated_requests"]["repeated_across_runs"], 1)
        dirs2 = [self.runs.write(r, [call(7, "s", f"c {r}", "c")]) for r in ("run_a", "run_b")]
        near = cr.analyze(dirs2)["near_identical"]
        self.assertEqual(near[0]["baseline_call_id"], "run_a-7")


class Examples(unittest.TestCase):
    def _runs(self, name: str) -> list[str]:
        return [str(EXAMPLES / name / f"run_{i}") for i in (1, 2, 3)]

    def test_random_ids(self) -> None:
        result = cr.analyze(self._runs("random_ids"), final="commit")
        self.assertEqual(result["stages"]["first_divergent"], "interpret")
        self.assertFalse(result["final"]["all_agree"])
        interpret = [n for n in result["near_identical"] if n["stage"] == "interpret"]
        self.assertEqual(len(interpret), 6)
        self.assertTrue(all(n["differing_lines"] == 1 for n in interpret))

    def test_aliased_ids(self) -> None:
        result = cr.analyze(self._runs("aliased_ids"), final="commit")
        self.assertEqual(result["near_identical"], [])
        self.assertIsNone(result["stages"]["first_divergent"])
        self.assertTrue(result["final"]["all_agree"])
        rep = result["repeated_requests"]
        self.assertEqual(rep["repeated_in_all_runs"], rep["distinct_requests"])
        self.assertEqual(rep["repeated_with_changed_output"], 0)

    def test_output_is_deterministic_and_independent_of_argument_order(self) -> None:
        runs = self._runs("random_ids")
        first = cr.render_markdown(cr.analyze(runs, final="commit"))
        second = cr.render_markdown(cr.analyze(list(reversed(runs)), final="commit"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
