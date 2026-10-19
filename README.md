# agent-box

Agentic forecasting jobs for FRI. Today this is the FRI Forecaster package
(`fri_forecaster`); the job queue it is growing into is specced in issue #1.

[pi](https://github.com/earendil-works/pi) drives the model; this package supplies the
research tools (Tavily `web_search` / `read_page`), the submit tools, a research floor,
and the bookkeeping (evidence, usage, merged answer). Factored out of AIRO on 2026-10-05;
AIRO and the ForecastBench driver are its users.

How it fits with FRI's other code: **fri-utils is model identity** (immutable
`model_run_key`, the warehouse's `dim_model_run`); **this is the loop**. Single-call
baselines stay on fri-utils; agentic configurations run here.

## How it works

One forecasting call is a folder. The caller writes the inbox, pi works the question
with the tools it is given, and the answers land in the same folder, which is the outbox.

```
 caller: harness.run(model, system, prompt, output schema, tools)
   │
   ▼
 ┌─ inbox: <workdir>/attempt-N/ ──────────────────────────────────┐
 │ system.txt     the system prompt                               │
 │ prompt.txt     the question                                    │
 │ spec.json      the tools offered, the research floor, the      │
 │                shape of one answer, the answers wanted         │
 └───────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
 ┌─ pi 0.99.2: the agent loop (model calls, retries, context) ────┐
 │ no built-in tools; its only tools come from pi_tools.ts:       │
 │   web_search, read_page ──> tool_cli.py ──> Tavily             │
 │   submit_forecast (and optionally submit_cells), locked        │
 │   until the research floor is met                              │
 └───────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
 ┌─ outbox: the same folder ──────────────────────────────────────┐
 │ final.json     the answer (submit_forecast)                    │
 │ pieces.jsonl   answers sent along the way (submit_cells)       │
 │ events.jsonl   pi's event stream: every tool call, tokens, $   │
 └───────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
 result: cells, answer, evidence (the research calls), usage
```

Every attempt gets a fresh `attempt-N` folder, so a retry never reads an older answer.

The loop has options, all off by default (the ForecastBench driver's loop); AIRO's
runner turns them on to match its own instrument. `harness.run(..., **options)`:

| option | what it does |
|---|---|
| `floor_counts="returned"` | failed research calls count toward the research floor too |
| `coverage=[keys]` | submission waits for a recent-window `web_search` tagged with each key |
| `withhold_final=True` | the submit tools are not offered at all until the floor and coverage are met |
| `warn_missing_keys=False` | the final answer is accepted as sent, with no warning about missing cells |
| `max_turns=N` | from turn N only `submit_forecast` is offered and asked for, for up to two more turns; a reply in prose gets a nudge |
| `debrief={...}` | after the answer, one more turn for a structured debrief, validated, with retries, saved to `debrief.json` |
| `tool_parameters={...}` | replace a tool's parameter schema, e.g. to require a `cause` on `web_search` |
pi runs isolated: no built-in tools, no discovered extensions or skills, no
AGENTS.md/CLAUDE.md, so the inbox is the whole prompt.

Today each caller loops over calls itself: the ForecastBench driver makes one call per
question per model ([fri_forecaster/README.md](fri_forecaster/README.md)). Issue #1
puts a shared queue in front, **planned, not built yet**:

```
 requesters ──> inbox/<job>.json ──> workers ──> outbox/<job>/result.json ──> requesters
                                     each runs one call as above
```

## What goes in and out

A ForecastBench-style question with two resolution dates. Values are illustrative.

**In.** `prompt.txt`:

```
Today is 2026-10-11.

Question (yfinance): Will SPY's market close price be higher on the resolution date than on 2026-10-11?

Resolution criteria: Resolves to the market close price on the resolution date, from Yahoo Finance.
URL: https://finance.yahoo.com/quote/SPY
Value on 2026-10-09: 671.2 (SPY's market close on 2026-10-09.)

Give one forecast for EACH of these resolution dates (the probability the question resolves Yes when evaluated on that date): 2026-10-18, 2026-11-10
```

The harness hands pi the prompt on stdin, never as an argument (Linux allows one at most
128 KiB, and AIRO's joint prompt is past that); pi reads it as its first message, trimmed
of surrounding whitespace. `system.txt` begins *"You are a careful, calibrated forecaster…"*. `spec.json` says which
tools pi gets, how much research comes first, and what an answer looks like:

```json
{
 "research_tools": ["web_search", "read_page"],
 "min_research": 2,
 "tools": {
  "web_search": {"description": "Search the web. Returns titles, URLs, snippets and dates. …"},
  "read_page": {"description": "Read the text of one web page …"},
  "submit_forecast": {"description": "Submit your final forecasts: one {resolution_date, probability} per resolution date …"}
 },
 "cell_schema": {"type": "object", "required": ["resolution_date", "probability"],
                 "properties": {"resolution_date": {"type": "string"},
                                "probability": {"type": "number", "minimum": 0, "maximum": 1}}},
 "key_fields": ["resolution_date"],
 "allowed_keys": ["2026-10-18", "2026-11-10"]
}
```

**Out.** `final.json`, written when the model calls `submit_forecast`:

```json
{
 "t": 1791729120000,
 "research": 3,
 "research_failed": 0,
 "answer": {
  "forecasts": [
   {"resolution_date": "2026-10-18", "probability": 0.56},
   {"resolution_date": "2026-11-10", "probability": 0.6}
  ],
  "rationale": "SPY closes higher over a week a little over half the time, and over a month about 60%; nothing scheduled before 2026-11-10 looks large enough to move that.",
  "key_sources": ["https://finance.yahoo.com/quote/SPY"]
 }
}
```

The result `harness.run` returns, folded from the outbox:

```json
{
 "cells": {
  "2026-10-18": {"resolution_date": "2026-10-18", "probability": 0.56},
  "2026-11-10": {"resolution_date": "2026-11-10", "probability": 0.6}
 },
 "answer": {"forecasts": ["…as in final.json…"], "rationale": "…", "key_sources": ["…"]},
 "evidence": [
  {"tool": "web_search", "args": {"query": "SPY weekly returns base rate"}, "result": {"results": ["…"]}, "error": null},
  {"tool": "read_page", "args": {"url": "https://finance.yahoo.com/quote/SPY"}, "result": {"text": "…"}, "error": null},
  {"tool": "web_search", "args": {"query": "market calendar October November 2026", "recent_days": 30}, "result": {"results": ["…"]}, "error": null}
 ],
 "usage": {"turns": 6, "input_tokens": 48210, "output_tokens": 2950, "cached_tokens": 31800,
           "cost_usd": 0.31, "research_ok": 3, "research_failed": 0, "harness": "pi", "seconds": 74},
 "delivery": {"cells_calls": 0, "cells_in_pieces": 0, "cells_in_final": 2},
 "rc": 0
}
```

Only the answers asked for (`allowed_keys`) are kept; anything else the model sends is
dropped.

## Requirements

- Python ≥ 3.11, stdlib only.
- Node and pi **0.99.2**, pinned in `package.json`: `npm ci` in the checkout puts it where
  the harness looks (`harness.REPO_PI`). Override with `FORECASTER_PI` (a `pi` executable
  or its `cli.js`). pi owns model auth.
- `TAVILY_API_KEY` for research, and the providers' API keys, from the environment or an
  env file (`keys.py`).

```sh
git clone https://github.com/forecastingresearch/agent-box && cd agent-box
npm ci && python -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

## Use

```python
from fri_forecaster import harness
res = harness.run(model="openai-codex/gpt-6-astra", system=SYS, prompt=P, workdir=W,
                  cell_schema={...}, key_fields=["resolution_date"], allowed_keys=[...],
                  tool_descriptions={...}, min_research=2)
res["cells"], res["answer"], res["evidence"], res["usage"]
```

ForecastBench, from picking models by ECI to the forecast sets you upload:
[fri_forecaster/README.md](fri_forecaster/README.md).

## Test

```sh
python -m pytest -q        # no network, no spend
```

The harness and driver tests use a fake pi. `tests/test_pi_tools.py` runs the real pi
(`npm ci` first; skipped without it) against a scripted model on a local server, so
`pi_tools.ts` is tested as pi loads it: the research floor, pieces, the final answer.
CI runs both on every push and pull request.
