"""fri_forecaster: the shared pi harness and the ForecastBench driver, against a
fake `pi` (no network, no spend)."""
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest

from fri_forecaster import forecastbench as fb
from fri_forecaster import harness

FAKE_PI = textwrap.dedent("""\
    #!{py}
    import json, os, sys
    out = os.environ["FORECASTER_OUT"]
    open(os.path.join(out, "stdin.txt"), "w").write(sys.stdin.read())   # the prompt, as pi would read it
    spec = json.load(open(os.path.join(out, "spec.json")))
    keys = spec["allowed_keys"]
    field = spec["key_fields"][0]
    if spec["tools"].get("submit_cells"):
        with open(os.path.join(out, "pieces.jsonl"), "w") as f:
            f.write(json.dumps({{"forecasts": [{{field: keys[0], "probability": 0.1}}, {{field: "bogus", "probability": 0.9}}]}}) + "\\n")
    p = float(os.environ.get("FAKE_P", "0.3"))
    final = [{{field: k, "probability": p}} for k in keys[(0 if os.environ.get("FAKE_ALL") else 1):]]
    json.dump({{"answer": {{"forecasts": final, "rationale": "because", "key_sources": ["u"]}}}}, open(os.path.join(out, "final.json"), "w"))
    ev = [{{"type": "tool_execution_start", "toolCallId": "c1", "toolName": "web_search", "args": {{"query": "q"}}}},
          {{"type": "tool_execution_end", "toolCallId": "c1", "toolName": "web_search", "result": {{"details": {{"results": [1, 2]}}}}}},
          {{"type": "tool_execution_start", "toolCallId": "c2", "toolName": "read_page", "args": {{"url": "u"}}}},
          {{"type": "tool_execution_end", "toolCallId": "c2", "toolName": "read_page", "result": {{"details": {{"text": "", "error": "page read failed"}}}}}},
          {{"type": "tool_execution_end", "toolName": "submit_forecast", "args": {{}}, "result": {{}}}},
          {{"type": "message_end", "message": {{"role": "assistant", "usage": {{"input": 10, "output": 5, "cacheRead": 2, "cost": {{"total": 0.01}}}}}}}},
          {{"type": "message_end", "message": {{"role": "assistant", "usage": {{"input": 1, "output": 1, "cost": {{"total": 0.02}}}}}}}}]
    for e in ev:
        print(json.dumps(e))
    """)


def fake_pi(d):
    p = os.path.join(d, "pi")
    open(p, "w").write(FAKE_PI.format(py=sys.executable))
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return p


SCHEMA = {"type": "object", "properties": {"k": {"type": "string"}, "probability": {"type": "number"}}}
DESC = {"web_search": "s", "read_page": "r", "submit_cells": "c", "submit_forecast": "f"}


class HarnessTest(unittest.TestCase):
    def test_run_merges_pieces_and_final(self):
        with tempfile.TemporaryDirectory() as d:
            res = harness.run(model="m/x", system="S", prompt="P", workdir=os.path.join(d, "w"),
                              cell_schema=SCHEMA, key_fields=["k"], allowed_keys=["a", "b", "c"],
                              tool_descriptions=DESC, pi=fake_pi(d))
            self.assertEqual(res["rc"], 0)
            self.assertEqual(open(os.path.join(res["workdir"], "stdin.txt")).read(), "P")   # the prompt, on stdin
            self.assertEqual(res["cells"]["a"]["probability"], 0.1)   # from a piece
            self.assertEqual(res["cells"]["c"]["probability"], 0.3)   # from the final
            self.assertNotIn("bogus", res["cells"])                   # pieces filtered to allowed keys
            self.assertEqual(res["delivery"], {"cells_calls": 1, "cells_in_pieces": 1, "cells_in_final": 2})
            self.assertEqual(len(res["evidence"]), 2)                 # submit tools are not evidence
            self.assertEqual(res["evidence"][0]["args"], {"query": "q"})   # args joined from the start event
            self.assertEqual(res["evidence"][1]["error"], "page read failed")
            self.assertEqual((res["usage"]["research_ok"], res["usage"]["research_failed"]), (1, 1))
            self.assertTrue(res["workdir"].endswith("attempt-1"))
            u = res["usage"]
            self.assertEqual((u["turns"], u["input_tokens"], u["output_tokens"], u["cached_tokens"]), (2, 11, 6, 2))
            self.assertAlmostEqual(u["cost_usd"], 0.03)
            self.assertEqual(res["answer"]["rationale"], "because")

    def test_retry_gets_a_fresh_attempt(self):
        """A second call never reads the first call's final.json."""
        with tempfile.TemporaryDirectory() as d:
            kw = dict(model="m/x", system="S", prompt="P", workdir=os.path.join(d, "w"), cell_schema=SCHEMA,
                      key_fields=["k"], allowed_keys=["a", "b"], tool_descriptions=DESC)
            harness.run(pi=fake_pi(d), **kw)
            broken = os.path.join(d, "broken")
            open(broken, "w").write("#!/bin/sh\nexit 1\n")
            os.chmod(broken, 0o755)
            res = harness.run(pi=broken, **kw)
            self.assertTrue(res["workdir"].endswith("attempt-2"))
            self.assertEqual(res["cells"], {})
            self.assertEqual(res["answer"], {})

    def test_final_cells_outside_allowed_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"answer": {"forecasts": [{"k": "a", "probability": 0.2}, {"k": "zzz", "probability": 0.9}]}},
                      open(os.path.join(d, "final.json"), "w"))
            cells, _, _ = harness.collect(d, ["k"], ["a"])
            self.assertEqual(list(cells), ["a"])

    def test_spec_offers_only_requested_tools(self):
        with tempfile.TemporaryDirectory() as d:
            spec = harness.write_spec(d, cell_schema=SCHEMA, key_fields=["k"], allowed_keys=["a"],
                                      tool_descriptions={"submit_forecast": "f"}, research_tools=(),
                                      cells_tool=False)
            self.assertEqual(list(spec["tools"]), ["submit_forecast"])
            self.assertEqual(spec["research_tools"], [])

    def test_command_loads_extension_without_builtins(self):
        cmd = harness.pi_command("m/x", "S", thinking="max", pi="pi")
        for flag in ("--no-builtin-tools", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-context-files"):
            self.assertIn(flag, cmd)
        self.assertEqual(harness.pi_command("m/x", "S", pi="/x/cli.js")[:2], ["node", "/x/cli.js"])
        self.assertEqual(cmd[cmd.index("-e") + 1], harness.EXT)
        self.assertTrue(os.path.exists(harness.EXT))
        # No positional message: the prompt goes on stdin (harness.PROMPT_ON_STDIN).
        self.assertEqual(cmd[-2:], ["--system-prompt", "S"])
        self.assertTrue(harness.PROMPT_ON_STDIN)


QSET = {"forecast_due_date": "2026-10-11", "question_set": "2026-10-11-llm.json", "questions": [
    {"id": "m1", "source": "metaculus", "question": "Will X happen?", "resolution_dates": "N/A",
     "background": "N/A", "freeze_datetime_value": "0.4", "freeze_datetime": "2026-10-01T00:00:00+00:00",
     "freeze_datetime_value_explanation": "The market value.", "resolution_criteria": "rc", "url": "u"},
    {"id": "SPY", "source": "yfinance", "question": "Will SPY be higher on {resolution_date} than on {forecast_due_date}?",
     "resolution_dates": ["2026-10-18", "2026-11-10"], "resolution_criteria": "rc", "url": "u"},
    {"id": ["a", "b"], "source": "acled", "question": "combo", "resolution_dates": ["2026-10-18"]},
]}


class ToolCliTest(unittest.TestCase):
    def test_retries_on_429_only(self):
        from fri_forecaster import tool_cli
        out = iter([{"error": "web search failed: HTTP Error 429: Too Many Requests"}, {"results": []}])
        waits = []
        self.assertEqual(tool_cli.call(lambda **a: next(out), {}, sleep=waits.append), {"results": []})
        self.assertEqual(len(waits), 1)
        waits.clear()
        self.assertEqual(tool_cli.call(lambda **a: {"error": "page read failed: 404"}, {}, sleep=waits.append),
                         {"error": "page read failed: 404"})
        self.assertEqual(waits, [])


class ForecastBenchTest(unittest.TestCase):
    def test_combination_questions_dropped(self):
        self.assertEqual([q["id"] for q in fb.questions_of(QSET)], ["m1", "SPY"])

    def test_sample_splits_market_and_dataset(self):
        qs = [{"source": "manifold", "id": f"m{i}"} for i in range(10)] + \
             [{"source": "fred", "id": f"d{i}", "resolution_dates": ["2026-10-04"]} for i in range(10)]
        got = fb.sample_of(qs, 6)
        self.assertEqual([q["id"] for q in got], ["m0", "m3", "m6", "d0", "d3", "d6"])
        self.assertEqual(len(fb.sample_of(qs[:2] + qs[10:], 6)), 6)  # few markets: dataset fills in

    def test_prompt(self):
        m, ds = fb.questions_of(QSET)
        pm = fb.build_prompt(m, "2026-10-11")
        self.assertIn("Value on 2026-10-01: 0.4", pm)
        self.assertIn('resolution_date "final"', pm)
        self.assertNotIn("Background", pm)                 # N/A fields omitted
        pd = fb.build_prompt(ds, "2026-10-11")
        self.assertIn("higher on the resolution date than on 2026-10-11", pd)
        self.assertIn("2026-10-18, 2026-11-10", pd)

    def test_prompt_keeps_a_zero_value(self):
        q = dict(QSET["questions"][0], freeze_datetime_value=0)
        self.assertIn("Value on 2026-10-01: 0 ", fb.build_prompt(q, "2026-10-11"))

    def _round(self, d):
        qpath = os.path.join(d, "qs.json")
        json.dump(QSET, open(qpath, "w"))
        return qpath, os.path.join(d, "round")

    def test_run_and_build(self):
        with tempfile.TemporaryDirectory() as d:
            qpath, root = self._round(d)
            pi = fake_pi(d)
            os.environ["FAKE_ALL"] = "1"
            try:
                os.environ["FAKE_P"] = "0.2"
                fb.main(["run", "--root", root, "--question-set", qpath, "--model", "a/one", "--pi", pi])
                os.environ["FAKE_P"] = "0.6"
                fb.main(["run", "--root", root, "--question-set", qpath, "--model", "b/two", "--pi", pi])
            finally:
                os.environ.pop("FAKE_ALL"); os.environ.pop("FAKE_P")
            rc = fb.main(["build", "--root", root, "--member", "a/one", "--member", "b/two",
                          "--name", "ensemble", "--model-org", "FRI", "--org", "Example Team", "--set-number", "3"])
            self.assertEqual(rc, 0)
            fs = json.load(open(os.path.join(root, "2026-10-11.example-team.3.json")))
            by = {(r["id"], r["resolution_date"]): r["forecast"] for r in fs["forecasts"]}
            self.assertEqual(set(by), {("m1", None), ("SPY", "2026-10-18"), ("SPY", "2026-11-10")})
            for v in by.values():
                self.assertAlmostEqual(v, 0.4)              # the mean of 0.2 and 0.6
            self.assertEqual(fb.validate(fs, QSET), [])

    def test_build_fails_below_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            qpath, root = self._round(d)
            fb.main(["run", "--root", root, "--question-set", qpath, "--model", "a/one", "--pi", fake_pi(d)])
            # the fake leaves the first key unanswered: neither question complete
            rc = fb.main(["build", "--root", root, "--member", "a/one", "--name", "x", "--model-org", "X", "--org", "Example Team", "--set-number", "1"])
            self.assertEqual(rc, 1)
            fs = json.load(open(os.path.join(root, "2026-10-11.example-team.1.json")))
            self.assertEqual([(r["id"], r["resolution_date"]) for r in fs["forecasts"]], [("SPY", "2026-11-10")])

    def test_resume_refuses_changed_settings(self):
        with tempfile.TemporaryDirectory() as d:
            qpath, root = self._round(d)
            fb.main(["run", "--root", root, "--question-set", qpath, "--model", "a/one", "--pi", fake_pi(d)])
            with self.assertRaises(SystemExit):
                fb.main(["run", "--root", root, "--question-set", qpath, "--model", "a/one",
                         "--min-research", "5", "--pi", fake_pi(d)])

    def test_member_spec(self):
        with tempfile.TemporaryDirectory() as d:
            qpath, root = self._round(d)
            fb.main(["run", "--root", root, "--question-set", qpath, "--model", "a/one", "--no-research",
                     "--thinking", "low", "--pi", fake_pi(d)])
            self.assertEqual(fb.main(["build", "--root", root, "--member", "a/one@low,noresearch", "--name", "x",
                                      "--model-org", "X", "--org", "Example Team", "--set-number", "1"]), 1)  # found; short coverage
            with self.assertRaises(SystemExit):
                fb.main(["build", "--root", root, "--member", "a/one", "--name", "x", "--model-org", "X", "--org", "Example Team", "--set-number", "1"])

    def test_torn_results_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.jsonl")
            open(p, "w").write(json.dumps({"id": "a", "source": "s", "probabilities": {"x": 0.1}, "wanted": ["x"]}) + "\n{\"id\": \"b")
            self.assertEqual(list(fb.read_results(p)), [("s", "a")])

if __name__ == "__main__":
    unittest.main()


EXAMPLE_REGISTRY = os.path.join(os.path.dirname(__file__), "..", "examples", "names.example.json")


class NamesTest(unittest.TestCase):
    def setUp(self):
        from fri_forecaster import names
        self.names, self.reg = names, names.load(EXAMPLE_REGISTRY)

    def test_registered_names_match_the_code(self):
        for ch in self.reg["conditions"]:
            name = self.names.current(self.reg, ch)
            self.assertEqual(name, name.lower())
            self.assertTrue(name.startswith(ch + "-"))

    def test_drift_is_refused_and_points_to_the_next_place(self):
        import copy
        reg = copy.deepcopy(self.reg)
        reg["names"]["solo-oak"]["members"][0]["options"]["thinking"] = "medium"
        with self.assertRaises(SystemExit) as e:
            self.names.current(reg, "solo")
        self.assertIn("solo-elm", str(e.exception))

    def test_member_specs(self):
        self.assertEqual(self.names.member_specs(self.reg, "solo"), ["anthropic/claude-opus-5-5@high,research"])
        self.assertTrue(all(s.endswith(",noresearch") for s in self.names.member_specs(self.reg, "unaided")))

    def test_places_unique(self):
        self.assertEqual(len(self.reg["places"]), len(set(self.reg["places"])))
