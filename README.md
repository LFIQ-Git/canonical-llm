# canonical-llm

**Canonical BRICK LLM router (Layer 0).** Single source of truth for model
routing across the BRICK family, distributed as the npm package
`@lfiq/canonical-llm` from this repo, pinned by commit SHA in each consumer's
`package.json`. Each consumer's `lib/llm.ts` is a one-line re-export shim
(`export * from "@lfiq/canonical-llm"`). **To change router behavior: edit this
repo, push, then bump the SHA pin in every consumer.** Never edit a per-app
shim.

## What it provides

- `chat()` / `chatWithRetry()` — single round-trip model calls, return text.
- `chatDetailed()` / `chatDetailedWithRetry()` — same call, richer return
  (`{ text, usage, raw, provider }`) for callers that need token counts, the
  raw provider response, or the provider that actually answered after failover.
- Provider chain with per-leg circuit breakers and automatic failover:
  **OCP (Win-PC, subscription, $0/call) → ocp-fallback (hosted, paid) →
  Anthropic direct SDK**. Forcing `LLM_PROVIDER=anthropic` runs leg 3 only.
- **Prompt caching** — the Anthropic leg marks the system block
  `cache_control: ephemeral`; repeated system prefixes over the model's
  cacheable minimum bill at ~10% of input price.
- **Model tiers** — `fast` / `balanced` / `deep` via `MODEL_TIERS`; callers
  pass `tier` instead of pinning a model ID.
- **Observability** — one `llm.call` structured log line per call. Since
  v0.3.0 the Anthropic leg also logs `cacheWrite` / `cacheRead` (prompt-cache
  token counts), and `ChatUsage` carries `cache_creation_input_tokens` /
  `cache_read_input_tokens`.

## Consumers (SHA pins to bump on release)

| App | Shim | Pin location |
|---|---|---|
| `02-brick.intel` | `app/lib/llm.ts` | own `package.json` + `package-lock.json` |
| `02-brick.keystone` | `lib/llm.ts` | own `package.json` + `package-lock.json` |
| `02-brick.command` `apps/collect` | `lib/llm.ts` | workspace `package.json`; lock entries live in **command's root `package-lock.json`** |
| `02-brick.command` `apps/leasing` | `lib/llm.ts` | same — command root lockfile |
| `02-brick.command` `apps/web` | `lib/llm.ts` | same — command root lockfile |

After bumping the three command workspace pins, run
`npm install --package-lock-only` at the **command repo root** (not in the
workspace dirs) so the single root lockfile re-resolves.

No consumer may import `openai` / `@anthropic-ai/sdk` outside its shim.

## CI guard — `check-llm-imports.mjs`

The master copy of the CI guard lives in the vault at
`02-brick.apps/02-brick.hub/canonical-llm/check-llm-imports.mjs` (it is not
part of this package). Each app keeps a verbatim copy at
`scripts/check-llm-imports.mjs` and runs it in CI on every push/PR. It fails
the build on any `openai` / `@anthropic-ai/sdk` import outside the app's
Layer-0 shim, except files carrying a `Layer-1 exception:` marker comment.

## Python twin — `brick_llm.py`

`brick_llm.py` (in this repo) is the Python port of `llm.ts` for the family's
Python work — same providers, tiers, retry policy, circuit breaker, cache
counters, and `llm.call` log shape. It currently lacks the ocp-fallback leg
(OCP → Anthropic only). Tests: `python3 test_brick_llm.py` — pure functions,
no network, no SDKs needed. When `llm.ts` changes, change `brick_llm.py` with
it.

## Vault mirror

`02-brick.apps/02-brick.hub/canonical-llm/` in the vault holds a reference
mirror of `llm.ts` / `brick_llm.py` / `test_brick_llm.py` plus the CI-guard
master. The mirror is for browsing and for the guard — **this repo is the
source of truth**; re-sync the mirror after each release.

## Env contract

`OCP_BASE_URL`, `OCP_API_KEY` — Win-PC OCP proxy;
`OCP_CF_ACCESS_CLIENT_ID` / `OCP_CF_ACCESS_CLIENT_SECRET` — Cloudflare Access
service token sent to the OCP gateway. `OCP_FALLBACK_BASE_URL`,
`OCP_FALLBACK_API_KEY` — hosted fallback leg. `ANTHROPIC_API_KEY` — direct
leg. `LLM_PROVIDER=anthropic` forces the direct leg. `LLM_MODEL_FAST` /
`LLM_MODEL_BALANCED` / `LLM_MODEL_DEEP` override tier model IDs;
`EXTRACTION_MODEL` overrides the default model. `LLM_APP_NAME` /
`LLM_USER_AGENT` tag outbound requests.

Architecture + rollout tracker: `02-brick.intel/docs/llm-architecture.md`.
