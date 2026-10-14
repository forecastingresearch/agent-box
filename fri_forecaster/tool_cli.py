#!/usr/bin/env python3
"""The research tools as a one-shot CLI for the pi extension (pi_tools.ts):
`tool_cli.py web_search '{"query": ...}'` prints the JSON fri_forecaster.tools
returns. One implementation for every harness, so comparisons are of loops,
not search stacks. Keys come from the environment the harness passes down.

`tool_cli.py validate` (arguments on stdin: {file, function, kwargs, value})
runs a caller's debrief validator, `function(value, **kwargs)` from the
Python file `file`, and prints {"problems": [...]}."""
import importlib.util
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fri_forecaster import tools  # noqa: E402
from fri_forecaster.keys import load_keys  # noqa: E402

FN = {"web_search": tools.web_search, "read_page": tools.read_page}
# Tavily answers 429 when many agents search at once (2026-10-06 practice
# run: 32 concurrent calls, ~1,700 failures). Back off and retry here, not in
# tools.py, which must stay identical to AIRO's copy.
BACKOFF = [2, 5, 10, 20, 40]


def rate_limited(res):
    return isinstance(res, dict) and "429" in str(res.get("error") or "")


def call(fn, args, sleep=time.sleep):
    res = fn(**args)
    for wait in BACKOFF:
        if not rate_limited(res):
            break
        sleep(wait * random.uniform(0.75, 1.25))
        res = fn(**args)
    return res


def validate(file, function, value, kwargs=None):
    spec = importlib.util.spec_from_file_location("forecaster_validator", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"problems": [str(p) for p in (getattr(module, function)(value, **(kwargs or {})) or [])]}


def main():
    if sys.argv[1] == "validate":
        print(json.dumps(validate(**json.load(sys.stdin))))
        return
    load_keys()
    tool, args = sys.argv[1], json.loads(sys.argv[2] if len(sys.argv) > 2 else "{}")
    if tool not in FN:
        sys.exit(f"unknown tool {tool!r}")
    print(json.dumps(call(FN[tool], args)))


if __name__ == "__main__":
    main()
