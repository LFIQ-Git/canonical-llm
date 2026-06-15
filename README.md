# canonical-llm

**Canonical BRICK LLM router (Layer 0).** Single source of truth for model
routing across the BRICK family.

`llm.ts` here is the master copy. Each app keeps a verbatim copy at its own
`lib/llm.ts`, synced from this file — the same copy-from-canonical pattern as
`canonical-app-family-menu/`. **Never edit a per-app copy.** Edit this file,
then re-sync every consumer.

## What it provides

- `chat()` / `chatWithRetry()` — single round-trip model calls, return text.
- `chatDetailed()` / `chatDetailedWithRetry()` — same call, richer return
  (`{ text, usage, raw, provider }`) for callers that need token counts, the
  raw provider response, or the provider that actually answered after failover.
- Provider routing: OCP proxy (subscription, $0/call) primary → Anthropic API
  fallback, with **automatic failover** and a circuit breaker.
- **Model tiers** — `fast` / `balanced` / `deep` via `MODEL_TIERS`; callers
  pass `tier` instead of pinning a model ID.
- **Observability** — one `llm.call` structured log line per call.

## Consumers (sync targets)

- `04-mosser.apps/leasing` (command.leasing) — `lib/llm.ts`
- `02-brick.command.collect` — `lib/llm.ts`
- `02-brick.command.repair` — `lib/llm.ts` (when it gains an AI surface)
- `02-brick.intel` — `lib/llm.ts`
- `02-brick.keystone` — `lib/llm.ts`
- `02-brick.runner` — `lib/llm.ts`

Each consumer needs `openai` and `@anthropic-ai/sdk` in its dependencies.
No consumer may import those SDKs outside its `lib/llm.ts`.

## CI guard — `check-llm-imports.mjs`

`check-llm-imports.mjs` here is the master copy of the CI guard that enforces
the rule above. Each app keeps a verbatim copy at `scripts/check-llm-imports.mjs`
(synced from this file) and runs it in CI via `npm run check:llm-imports` on
every push/PR. It scans `app/`/`lib/`/etc. for `openai` / `@anthropic-ai/sdk`
imports and fails the build if any offending file is outside the allowlist —
the app's Layer-0 router (`lib/llm.ts`, or `app/lib/llm.ts` in brick.intel),
plus files carrying a `Layer-1 exception:` marker comment (used by runner's 5
documented tool-use / JSON-mode / agent-loop files). Pure Node built-ins, no
deps. Edit the master here, then re-sync every consumer.

## Python twin — `brick_llm.py`

`brick_llm.py` is the Python port of `llm.ts` — a faithful behavioral twin for
the family's Python work (ETL jobs, e.g. the GDM extractor Cloud Run job).
It mirrors `llm.ts` feature-for-feature: same providers and `get_provider()`
resolution, same `MODEL_TIERS` / `DEFAULT_LLM_MODEL`, same OCP→Anthropic
automatic failover and circuit breaker (`BREAKER_THRESHOLD=3`,
`BREAKER_COOLDOWN_MS=60000`), same retry policy (3 attempts, `[500,1500,4000]`
ms backoff), and the same one-line `llm.call` structured log per call.

API: `chat()` / `chat_with_retry()` return text; `chat_detailed()` /
`chat_detailed_with_retry()` return a `ChatResult` (`text`, `usage`, `raw`,
`provider`). When `llm.ts` changes, change `brick_llm.py` with it.

- **Deps:** `openai` and `anthropic` Python SDKs — see `requirements.txt`
  (`pip install -r requirements.txt`). Both are imported lazily inside the
  call functions, so the pure helpers and the unit tests run without them.
- **Tests:** `test_brick_llm.py` — pure-function unit tests (no network).
  Run with `pytest`, or as a plain script: `python3 test_brick_llm.py`.
- **Env contract:** identical to `llm.ts` (see below).

## Env contract

`OCP_BASE_URL`, `OCP_API_KEY` — OCP proxy. `OCP_CF_ACCESS_CLIENT_ID`,
`OCP_CF_ACCESS_CLIENT_SECRET` — Cloudflare Access service token for the OCP
gateway (without it Access returns an SSO login page, not JSON).
`ANTHROPIC_API_KEY` — fallback.
`LLM_PROVIDER=anthropic` forces the direct provider. `LLM_MODEL_FAST` /
`LLM_MODEL_BALANCED` / `LLM_MODEL_DEEP` override tier model IDs.

Architecture + rollout tracker: `02-brick.intel/docs/llm-architecture.md`.

## Layer-0 (this package) vs Layer-1 (per-app direct SDK)

This router is **text in → text out**. It abstracts provider + tier + fallback
for the common case (`chat`, `chatDetailed`, `chatWithRetry`). It does **not**
model:

- multi-turn tool-use loops
- structured-output via Anthropic tool schemas
- document/PDF content blocks
- Anthropic beta APIs (`client.beta.*`: Files API, code_execution, …)

Apps that need those reach for the vendor SDK directly. That's a **Layer-1
exception**, not drift. Mark such files with a comment:

```ts
// Layer-1 exception: <feature> — the text-only canonical router cannot host
// this. Direct SDK import is intentional.
```

Even Layer-1 sites should import `MODEL_TIERS` from this package so a global
tier bump still reaches them. The hardcoded model strings are the drift to
avoid — the direct SDK call itself is fine.

### Current Layer-1 sites (BRICK family, 2026-06-11)

- `02-brick.runner/lib/agent/loop.ts` — multi-turn tool-use agent loop
- `02-brick.runner/lib/lease-abstract-extractor.ts` — tool-use structured output with PDF document blocks
- `02-brick.runner/app/api/excel-audit/route.ts` — beta Files API + code_execution
- `02-brick.runner/lib/fs-review-openai.ts` — OpenAI `gpt-4o-mini` structured outputs (different provider, deliberate)

Closing these would require either (a) extending this package with a
`chatWithTools` API that translates between OpenAI and Anthropic tool-use
shapes, or (b) OCP exposing an Anthropic-shape passthrough endpoint. Tracked
separately.
