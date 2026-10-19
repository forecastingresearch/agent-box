"""pi_tools.ts, through the real pi: a scripted model on a local OpenAI-compatible
server makes the tool calls, and the tests check what pi_tools.ts answered and
wrote. No network, no spend. Needs node and the pinned pi (`npm ci`); skipped
without them."""
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fri_forecaster import harness

SCHEMA = {"type": "object", "required": ["k", "probability"],
          "properties": {"k": {"type": "string"}, "probability": {"type": "number"}}}
DESC = {"web_search": "s", "read_page": "r", "submit_cells": "c", "submit_forecast": "f"}

# Research without the network: web_search answers, read_page fails. A
# debrief validator runs as the real tool_cli runs it.
FAKE_TOOL_CLI = textwrap.dedent("""\
    import json, sys
    sys.path.insert(0, {repo!r})
    tool = sys.argv[1]
    if tool == "validate":
        from fri_forecaster.tool_cli import validate
        print(json.dumps(validate(**json.load(sys.stdin))))
    else:
        print(json.dumps({{"results": [{{"url": "u", "title": "t"}}]}} if tool == "web_search"
                         else {{"error": "page read failed: 404"}}))
    """).format(repo=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
VALIDATOR = "def check(value, need):\n    return [f'{k}: missing' for k in need if k not in (value or {})]\n"


REFUSAL = object()


class ScriptedModel:
    """An OpenAI chat-completions endpoint that plays a script: one tool call per
    turn, then a closing text; REFUSAL ends a turn the way a provider's refusal
    does (pi reads it as an error). It keeps every request, so a test can read the
    tool result pi_tools.ts returned for each call."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model.requests.append(body)
                turn = model.script[len(model.requests) - 1] if len(model.requests) <= len(model.script) else "done"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in model.chunks(turn, len(model.requests)):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    @staticmethod
    def chunks(turn, n):
        base = {"id": f"c{n}", "object": "chat.completion.chunk", "created": 0, "model": "scripted"}
        if turn is REFUSAL:
            delta, finish = {"role": "assistant", "content": ""}, "content_filter"
        elif isinstance(turn, str):
            delta, finish = {"role": "assistant", "content": turn}, "stop"
        else:
            name, args = turn
            delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{n}", "type": "function",
                                                           "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        yield dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
        yield dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}])
        yield dict(base, choices=[], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    def tools(self, n):
        """The tools declared in request n (1-based)."""
        return {t["function"]["name"] for t in self.requests[n - 1].get("tools") or []}

    def last_user(self, n):
        """The text of the last user message in request n (1-based)."""
        msg = [m for m in self.requests[n - 1]["messages"] if m.get("role") == "user"][-1]
        content = msg["content"]
        return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)

    def tool_result(self, n):
        """The tool result the model saw at the start of request n (1-based):
        the answer to the call it made in turn n-1."""
        msg = [m for m in self.requests[n - 1]["messages"] if m.get("role") == "tool"][-1]
        content = msg["content"]
        return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)

    def close(self):
        self.server.shutdown()


# Skipped without node and pi, except in CI, where a skip would hide a broken install.
NEEDS_PI = unittest.skipUnless((harness.DEFAULT_PI and shutil.which("node")) or os.environ.get("CI"),
                               "needs node and pi (npm ci)")


class PiCase(unittest.TestCase):
    """A temp dir, the fake research tools, and run_pi: one harness.run
    through the real pi against a ScriptedModel."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        cli = os.path.join(d, "tool_cli.py")
        open(cli, "w").write(FAKE_TOOL_CLI)
        self.real_cli, harness.TOOL_CLI = harness.TOOL_CLI, cli

    def tearDown(self):
        harness.TOOL_CLI = self.real_cli
        self.tmp.cleanup()

    def run_pi(self, script, **kw):
        model = ScriptedModel(script)
        self.addCleanup(model.close)
        agent = os.path.join(self.tmp.name, "agent")
        os.makedirs(agent, exist_ok=True)
        json.dump({"providers": {"fake": {"baseUrl": model.url, "api": "openai-completions", "apiKey": "x",
                                          "models": [{"id": "scripted"}]}}},
                  open(os.path.join(agent, "models.json"), "w"))
        args = dict(model="fake/scripted", system="S", prompt="P", workdir=os.path.join(self.tmp.name, "w"),
                    cell_schema=SCHEMA, key_fields=["k"], allowed_keys=["a", "b"], tool_descriptions=DESC,
                    min_research=2, thinking="off", timeout=120,
                    env={"PI_CODING_AGENT_DIR": agent, "PI_OFFLINE": "1", "PI_TELEMETRY": "0"})
        args.update(kw)
        return harness.run(**args), model


@NEEDS_PI
class PiToolsTest(PiCase):
    """The default loop: the ForecastBench driver's."""

    def test_research_floor_pieces_and_final(self):
        res, model = self.run_pi([
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.1}], "rationale": "early"}),  # 1
            ("read_page", {"url": "u"}),                                                                # 2
            ("web_search", {"query": "q1"}),                                                            # 3
            ("web_search", {"query": "q2"}),                                                            # 4
            ("submit_cells", {"forecasts": [{"k": "a", "probability": 0.2}, {"k": "zzz", "probability": 0.9}]}),  # 5
            ("submit_forecast", {"forecasts": [{"k": "zzz", "probability": 0.5}], "rationale": "r"}),   # 6
            ("submit_forecast", {"forecasts": [{"k": "b", "probability": 0.3}], "rationale": "r",
                                 "key_sources": ["s"]}),                                                # 7
            ("submit_forecast", {"forecasts": [], "rationale": "again"}),                               # 8
            "done",                                                                                     # 9
        ])
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(len(model.requests), 9)

        # 1: submitting before any research is refused
        self.assertIn("submit_forecast is not offered yet: 2 more successful research call(s)", model.tool_result(2))
        # 2: a failed read does not count toward the floor
        self.assertIn("page read failed", model.tool_result(3))
        # 5: only wanted keys are accepted from a piece
        piece = json.loads(model.tool_result(6))
        self.assertEqual((piece["accepted"], piece["received"], piece["missing_first"]), (1, 1, ["b"]))
        # 6: unknown and missing keys get one warning, and nothing is written
        warn = json.loads(model.tool_result(7))
        self.assertEqual((warn["unknown_keys"], warn["missing_keys"]), (["zzz"], ["b"]))
        # 7: accepted; 8: a second final answer is refused
        self.assertTrue(json.loads(model.tool_result(8).split(" Done")[0])["ok"])
        self.assertIn("already called", model.tool_result(9))

        final = json.load(open(os.path.join(res["workdir"], "final.json")))
        self.assertEqual((final["research"], final["research_failed"]), (2, 1))
        self.assertEqual(final["answer"]["key_sources"], ["s"])
        self.assertEqual([json.loads(l)["accepted"] for l in open(os.path.join(res["workdir"], "pieces.jsonl"))], [1])
        self.assertEqual({k: c["probability"] for k, c in res["cells"].items()}, {"a": 0.2, "b": 0.3})
        self.assertEqual((res["usage"]["research_ok"], res["usage"]["research_failed"]), (2, 1))

    def test_no_research_offers_only_submit_forecast(self):
        res, model = self.run_pi([
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.4}, {"k": "b", "probability": 0.6}],
                                 "rationale": "r"}),
            "done",
        ], research_tools=(), min_research=0, cells_tool=False, tool_descriptions={"submit_forecast": "f"})
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        offered = {t["function"]["name"] for t in model.requests[0]["tools"]}
        self.assertEqual(offered, {"submit_forecast"})      # no research tools, and none of pi's built-ins
        self.assertEqual({k: c["probability"] for k, c in res["cells"].items()}, {"a": 0.4, "b": 0.6})
        self.assertEqual(model.last_user(1), "P")           # the prompt, from stdin, as the first user message

    def test_a_prompt_past_the_argv_limit_reaches_the_model(self):
        """The prompt goes on stdin: Linux hands one argument at most 128 KiB,
        and AIRO's joint prompt is past that (2026-10-19)."""
        prompt = "Forecast.\n" + "".join(f"--- question {i} ---\nWill it happen?\n" for i in range(6000))
        self.assertGreater(len(prompt.encode()), 131072)
        res, model = self.run_pi([
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.4}, {"k": "b", "probability": 0.6}],
                                 "rationale": "r"}),
            "done",
        ], prompt=prompt, research_tools=(), min_research=0, cells_tool=False,
           tool_descriptions={"submit_forecast": "f"})
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(model.last_user(1), prompt.strip())   # verbatim but for pi's trim
        self.assertEqual(open(os.path.join(res["workdir"], "prompt.txt")).read(), prompt)


SEARCH_WITH_CAUSE = {"type": "object", "required": ["query", "cause"],
                     "properties": {"query": {"type": "string"}, "recent_days": {"type": "integer"},
                                    "cause": {"type": "string", "enum": ["ai", "bio"]}}}


@NEEDS_PI
class LoopOptionsTest(PiCase):
    """The loop options AIRO's runner sets (pi_tools.ts header)."""

    def test_withheld_until_every_call_and_group_counts(self):
        res, model = self.run_pi([
            ("read_page", {"url": "u"}),                                          # fails, still counts
            ("web_search", {"query": "q", "cause": "ai", "recent_days": 7}),
            ("web_search", {"query": "q", "cause": "bio"}),                       # no recent_days: not coverage
            ("web_search", {"query": "q", "cause": "bio", "recent_days": 30}),
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.1}], "rationale": "r"}),
            "done",
        ], min_research=4, floor_counts="returned", coverage=["ai", "bio"], withhold_final=True,
            warn_missing_keys=False, tool_parameters={"web_search": SEARCH_WITH_CAUSE})
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        for n in (1, 2, 3, 4):            # 4 returned (one failed) and both groups covered after call 4
            self.assertNotIn("submit_forecast", model.tools(n))
            self.assertNotIn("submit_cells", model.tools(n))
        self.assertTrue({"submit_forecast", "submit_cells"} <= model.tools(5))
        sent = next(t for t in model.requests[0]["tools"] if t["function"]["name"] == "web_search")
        self.assertEqual(sent["function"]["parameters"]["properties"]["cause"]["enum"], ["ai", "bio"])
        final = json.load(open(os.path.join(res["workdir"], "final.json")))
        self.assertEqual((final["research"], final["research_failed"], final["forced"]), (3, 1, False))
        self.assertEqual({k: c["probability"] for k, c in res["cells"].items()}, {"a": 0.1})  # no warning for b

    def test_turn_limit_offers_only_submission(self):
        res, model = self.run_pi([
            ("web_search", {"query": "q1"}),
            ("web_search", {"query": "q2"}),
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.2}], "rationale": "forced"}),
            "done",
        ], min_research=5, withhold_final=True, warn_missing_keys=False, max_turns=3)
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(model.tools(3), {"submit_forecast"})
        self.assertTrue(model.last_user(3).startswith("Now call submit_forecast with your final answer."))
        self.assertTrue(res["forced"])

    def test_turn_limit_ends_a_call_that_never_submits(self):
        res, model = self.run_pi(["thinking", "still thinking", "no", "no", "no", "no"],
                                 min_research=1, withhold_final=True, max_turns=3)
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(len(model.requests), 5)          # the limit is max_turns + 2
        self.assertIn("Keep researching with the tools", model.last_user(2))
        self.assertEqual(model.tools(5), {"submit_forecast"})
        self.assertFalse(os.path.exists(os.path.join(res["workdir"], "final.json")))

    def test_debrief_merges_and_retries(self):
        validator = os.path.join(self.tmp.name, "validator.py")
        open(validator, "w").write(VALIDATOR)
        debrief = {"tool": {"name": "submit_debrief", "description": "d",
                            "parameters": {"type": "object", "properties": {"a": {"type": "string"},
                                                                            "b": {"type": "string"}}}},
                   "prompt": "Debrief now.", "retries": 1,
                   "validator": {"file": validator, "function": "check", "kwargs": {"need": ["a", "b"]}}}
        res, model = self.run_pi([
            ("web_search", {"query": "q1"}),
            ("web_search", {"query": "q2"}),
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.1}, {"k": "b", "probability": 0.2}],
                                 "rationale": "r"}),
            ("submit_debrief", {"a": "why a"}),
            ("submit_debrief", {"b": "why b"}),
            "done",
        ], debrief=debrief)
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertNotIn("submit_debrief", model.tools(1))
        self.assertEqual(model.tools(4), {"submit_debrief"})
        self.assertEqual(model.last_user(4), "Debrief now.")
        self.assertIn("b: missing", model.tool_result(5))
        self.assertIn("fixing: b: missing", model.last_user(5))
        self.assertEqual(res["debrief"], {"result": {"a": "why a", "b": "why b"}, "complete": True,
                                          "attempts": 2, "problems": [], "error": None})

    def test_debrief_asks_again_after_a_refusal(self):
        # A refused turn ends pi's run unless something asks again.
        debrief = {"tool": {"name": "submit_debrief", "description": "d",
                            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}},
                   "prompt": "Debrief now.", "retries": 1}
        res, model = self.run_pi([
            ("web_search", {"query": "q1"}),
            ("web_search", {"query": "q2"}),
            ("submit_forecast", {"forecasts": [{"k": "a", "probability": 0.1}, {"k": "b", "probability": 0.2}],
                                 "rationale": "r"}),
            REFUSAL,
            ("submit_debrief", {"a": "why a"}),
            "done",
        ], debrief=debrief)
        self.assertEqual(res["rc"], 0, open(os.path.join(res["workdir"], "stderr.txt")).read())
        self.assertEqual(model.last_user(5), "Now call submit_debrief with the debrief.")
        self.assertEqual(model.tools(5), {"submit_debrief"})
        self.assertEqual(res["debrief"], {"result": {"a": "why a"}, "complete": True,
                                          "attempts": 2, "problems": [], "error": None})


if __name__ == "__main__":
    unittest.main()
