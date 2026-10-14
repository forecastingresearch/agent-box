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

# Research without the network: web_search answers, read_page fails.
FAKE_TOOL_CLI = textwrap.dedent("""\
    import json, sys
    tool = sys.argv[1]
    print(json.dumps({"results": [{"url": "u", "title": "t"}]} if tool == "web_search"
                     else {"error": "page read failed: 404"}))
    """)


class ScriptedModel:
    """An OpenAI chat-completions endpoint that plays a script: one tool call per
    turn, then a closing text. It keeps every request, so a test can read the
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
        if isinstance(turn, str):
            delta, finish = {"role": "assistant", "content": turn}, "stop"
        else:
            name, args = turn
            delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"call_{n}", "type": "function",
                                                           "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        yield dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
        yield dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": finish}])
        yield dict(base, choices=[], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})

    def tool_result(self, n):
        """The tool result the model saw at the start of request n (1-based):
        the answer to the call it made in turn n-1."""
        msg = [m for m in self.requests[n - 1]["messages"] if m.get("role") == "tool"][-1]
        content = msg["content"]
        return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)

    def close(self):
        self.server.shutdown()


# Skipped without node and pi, except in CI, where a skip would hide a broken install.
@unittest.skipUnless((harness.DEFAULT_PI and shutil.which("node")) or os.environ.get("CI"), "needs node and pi (npm ci)")
class PiToolsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
