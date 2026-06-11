/**
 * llm.ts — CANONICAL BRICK LLM router (Layer 0).
 *
 * This is the single source of truth. Each app keeps a verbatim copy at
 * `lib/llm.ts`, synced from here — same copy-from-canonical pattern as
 * `canonical-app-family-menu/`. Do NOT edit per-app copies; edit this file
 * and re-sync. Architecture + rollout tracker: 02-brick.intel/docs/llm-architecture.md.
 *
 * Providers (resolved + failed over automatically):
 *   - "ocp"       — Win-PC OCP proxy, subscription-backed, $0/call. Primary
 *                   when OCP_BASE_URL is set and LLM_PROVIDER !== "anthropic".
 *   - "anthropic" — Anthropic SDK directly with ANTHROPIC_API_KEY. Fallback,
 *                   and forced when LLM_PROVIDER=anthropic.
 *
 * Four things every app gets for free here:
 *   1. Automatic failover — if OCP errors transiently, the call retries on
 *      Anthropic (when ANTHROPIC_API_KEY exists). A circuit breaker skips a
 *      flapping proxy for a cooldown window.
 *   2. Model tiers — ask for "fast" | "balanced" | "deep" instead of pinning
 *      a model ID in every caller. Bump the map here, once.
 *   3. Observability — every call emits a structured `llm.call` log line
 *      (provider, model, tier, latency, ok) for spend/usage rollups.
 *   4. Two return shapes — `chat()` returns just the text; `chatDetailed()`
 *      returns `{ text, usage, raw, provider }` for callers that need token
 *      counts, the raw provider response, or the provider that actually
 *      answered (post-failover).
 *
 * No app may import `openai` / `@anthropic-ai/sdk` outside this file.
 */
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";

export type Provider = "ocp" | "anthropic";
export type ModelTier = "fast" | "balanced" | "deep";

/** Tier → model ID. One place to bump models family-wide. Keep IDs inside
 *  the OCP proxy allowlist. */
export const MODEL_TIERS: Record<ModelTier, string> = {
  fast: process.env.LLM_MODEL_FAST ?? "claude-haiku-4-5",
  balanced: process.env.LLM_MODEL_BALANCED ?? "claude-sonnet-4-6",
  deep: process.env.LLM_MODEL_DEEP ?? "claude-sonnet-4-6",
};

/** Default model when neither `model` nor `tier` is given. */
export const DEFAULT_LLM_MODEL = process.env.EXTRACTION_MODEL ?? MODEL_TIERS.balanced;

export function getProvider(): Provider {
  if ((process.env.LLM_PROVIDER ?? "").toLowerCase() === "anthropic") return "anthropic";
  if (process.env.OCP_BASE_URL) return "ocp";
  return "anthropic";
}

let _oai: OpenAI | null = null;
let _anthropic: Anthropic | null = null;

function getOpenAI(): OpenAI {
  if (_oai) return _oai;
  const apiKey = process.env.OCP_API_KEY ?? "ocp-no-key";
  const baseURL = process.env.OCP_BASE_URL;
  if (!baseURL) {
    throw new Error(
      "OCP_BASE_URL not set — cannot route to OCP proxy. Set LLM_PROVIDER=anthropic to fall back to direct Anthropic API.",
    );
  }
  // OCP's gateway 403s the OpenAI SDK's default `User-Agent: OpenAI/JS …`
  // header ("Your request was blocked"). Set a non-OpenAI UA so the
  // subscription proxy accepts the request. Callers can override via the
  // LLM_USER_AGENT env if they want their own identity in the logs.
  const userAgent = process.env.LLM_USER_AGENT ?? "brick-canonical-llm/0.1";
  _oai = new OpenAI({ apiKey, baseURL, defaultHeaders: { "User-Agent": userAgent } });
  return _oai;
}

function getAnthropic(): Anthropic {
  if (_anthropic) return _anthropic;
  const apiKey = process.env.ANTHROPIC_API_KEY ?? "";
  if (!apiKey) {
    throw new Error(
      "ANTHROPIC_API_KEY not set. Set OCP_BASE_URL to use the subscription proxy instead.",
    );
  }
  _anthropic = new Anthropic({ apiKey });
  return _anthropic;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatArgs {
  /** System / instruction text. Cache hint applied on the Anthropic path. */
  system: string;
  /** Conversation messages — usually one user message; multi-turn supported. */
  messages: ChatMessage[];
  /** Explicit model ID. Wins over `tier`. */
  model?: string;
  /** Capability tier — resolved via MODEL_TIERS when `model` is absent. */
  tier?: ModelTier;
  /** Defaults to 8192. */
  maxTokens?: number;
  /** Sampling temperature — passed through to the provider when set. */
  temperature?: number;
}

/** Token counts, normalized across providers ("" when the provider omits them). */
export interface ChatUsage {
  input_tokens?: number;
  output_tokens?: number;
}

/** Rich result — text plus the metadata callers occasionally need. */
export interface ChatResult {
  text: string;
  usage: ChatUsage;
  /** The unmodified provider response object. */
  raw: unknown;
  /** The provider that actually produced this answer (post-failover). */
  provider: Provider;
}

// ── observability ───────────────────────────────────────────────────────
function logCall(rec: {
  provider: Provider;
  model: string;
  tier?: ModelTier;
  latencyMs: number;
  ok: boolean;
  failedOver?: boolean;
  err?: string;
}): void {
  // One structured line per call — cheap to grep / ship to a log drain later.
  try {
    process.stdout.write(`llm.call ${JSON.stringify({ ...rec, ts: new Date().toISOString() })}\n`);
  } catch {
    /* logging must never throw */
  }
}

// ── circuit breaker — skip a flapping OCP proxy for a cooldown ───────────
const BREAKER_THRESHOLD = 3;
const BREAKER_COOLDOWN_MS = 60_000;
let _ocpFailures = 0;
let _ocpOpenedAt = 0;

function ocpBreakerOpen(): boolean {
  if (_ocpFailures < BREAKER_THRESHOLD) return false;
  if (Date.now() - _ocpOpenedAt > BREAKER_COOLDOWN_MS) {
    _ocpFailures = 0; // cooldown elapsed — half-open, allow a probe
    return false;
  }
  return true;
}
function noteOcpResult(ok: boolean): void {
  if (ok) {
    _ocpFailures = 0;
  } else {
    _ocpFailures += 1;
    if (_ocpFailures >= BREAKER_THRESHOLD) _ocpOpenedAt = Date.now();
  }
}

function resolveModel(args: ChatArgs): { model: string; tier?: ModelTier } {
  if (args.model) return { model: args.model, tier: args.tier };
  if (args.tier) return { model: MODEL_TIERS[args.tier], tier: args.tier };
  return { model: DEFAULT_LLM_MODEL };
}

/** One provider call's normalized output (text + usage + raw response). */
interface ProviderOutput {
  text: string;
  usage: ChatUsage;
  raw: unknown;
}

async function callOcp(
  system: string,
  messages: ChatMessage[],
  model: string,
  maxTokens: number,
  temperature?: number,
): Promise<ProviderOutput> {
  const oai = getOpenAI();
  const r = await oai.chat.completions.create({
    model,
    max_tokens: maxTokens,
    ...(temperature !== undefined ? { temperature } : {}),
    messages: [
      { role: "system", content: system },
      ...messages.map((m) => ({ role: m.role, content: m.content })),
    ],
  });
  return {
    text: r.choices[0]?.message?.content ?? "",
    usage: {
      input_tokens: r.usage?.prompt_tokens,
      output_tokens: r.usage?.completion_tokens,
    },
    raw: r,
  };
}

async function callAnthropic(
  system: string,
  messages: ChatMessage[],
  model: string,
  maxTokens: number,
  temperature?: number,
): Promise<ProviderOutput> {
  // System block marked cache_control: ephemeral → 5-min prompt-cache hits
  // (~90% input-token discount) when the system prefix repeats.
  const a = getAnthropic();
  const r = await a.messages.create({
    model,
    max_tokens: maxTokens,
    ...(temperature !== undefined ? { temperature } : {}),
    system: [{ type: "text", text: system, cache_control: { type: "ephemeral" } }],
    messages: messages.map((m) => ({ role: m.role, content: m.content })),
  });
  const block = r.content.find((b) => b.type === "text");
  return {
    text: block && block.type === "text" ? block.text : "",
    usage: {
      input_tokens: r.usage?.input_tokens,
      output_tokens: r.usage?.output_tokens,
    },
    raw: r,
  };
}

/**
 * Core routing + failover. Returns the rich result. `chat()` and
 * `chatDetailed()` are thin wrappers over this.
 *
 * Routing: primary provider per `getProvider()`. If the primary is OCP and it
 * errors transiently — or its circuit breaker is open — the call fails over
 * to Anthropic when ANTHROPIC_API_KEY is set. Anthropic-primary does not fail
 * over to OCP (it is the more reliable leg).
 */
async function runChat(args: ChatArgs): Promise<ChatResult> {
  const { model, tier } = resolveModel(args);
  const maxTokens = args.maxTokens ?? 8192;
  const primary = getProvider();
  const anthropicAvailable = Boolean(process.env.ANTHROPIC_API_KEY);
  const started = Date.now();

  // OCP primary (unless its breaker is open) → fail over to Anthropic.
  if (primary === "ocp" && !ocpBreakerOpen()) {
    try {
      const out = await callOcp(args.system, args.messages, model, maxTokens, args.temperature);
      noteOcpResult(true);
      logCall({ provider: "ocp", model, tier, latencyMs: Date.now() - started, ok: true });
      return { ...out, provider: "ocp" };
    } catch (e) {
      noteOcpResult(false);
      const err = e instanceof Error ? e.message : String(e);
      if (!anthropicAvailable) {
        logCall({ provider: "ocp", model, tier, latencyMs: Date.now() - started, ok: false, err });
        throw e;
      }
      // fall through to Anthropic failover
      logCall({ provider: "ocp", model, tier, latencyMs: Date.now() - started, ok: false, err });
    }
  }

  const fellOver = primary === "ocp";
  const aStarted = Date.now();
  try {
    const out = await callAnthropic(args.system, args.messages, model, maxTokens, args.temperature);
    logCall({ provider: "anthropic", model, tier, latencyMs: Date.now() - aStarted, ok: true, failedOver: fellOver });
    return { ...out, provider: "anthropic" };
  } catch (e) {
    const err = e instanceof Error ? e.message : String(e);
    logCall({ provider: "anthropic", model, tier, latencyMs: Date.now() - aStarted, ok: false, failedOver: fellOver, err });
    throw e;
  }
}

/**
 * Single round-trip chat. Returns the model's text ("" if none). For token
 * counts or the raw provider response, use `chatDetailed()`.
 */
export async function chat(args: ChatArgs): Promise<string> {
  return (await runChat(args)).text;
}

/**
 * Like `chat()`, but returns the rich result — `{ text, usage, raw, provider }`.
 * Use when you need token counts, the raw response (e.g. stop_reason), or the
 * provider that actually answered after failover.
 */
export async function chatDetailed(args: ChatArgs): Promise<ChatResult> {
  return runChat(args);
}

// ── retry helpers ───────────────────────────────────────────────────────
const LLM_MAX_ATTEMPTS = 3;
const LLM_BACKOFF_MS = [500, 1500, 4000];

function isTransientLLMError(e: unknown): boolean {
  if (!e || typeof e !== "object") return false;
  const status = (e as { status?: number }).status;
  if (status === 429) return true;
  if (typeof status === "number" && status >= 500 && status < 600) return true;
  const code = (e as { code?: string }).code;
  // `ETIMEDOUT` is Node's connection-timeout code — list it explicitly; the
  // literal `TIMEOUT` does not match it as a substring. `TIMEOUT` still
  // covers SDK-specific timeout codes (e.g. `REQUEST_TIMEOUT`).
  if (code && /ETIMEDOUT|TIMEOUT|ECONNRESET|ECONNREFUSED|EAI_AGAIN/i.test(code)) return true;
  return false;
}

export async function chatWithRetry(args: ChatArgs): Promise<string> {
  return (await chatDetailedWithRetry(args)).text;
}

export async function chatDetailedWithRetry(args: ChatArgs): Promise<ChatResult> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < LLM_MAX_ATTEMPTS; attempt++) {
    try {
      return await runChat(args);
    } catch (e) {
      lastErr = e;
      if (!isTransientLLMError(e) || attempt === LLM_MAX_ATTEMPTS - 1) throw e;
      const delay = LLM_BACKOFF_MS[attempt] ?? 4000;
      await new Promise((r) => setTimeout(r, delay));
    }
  }
  throw lastErr;
}

export function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => {
      reject(new Error(`${label} timed out after ${ms}ms`));
    }, ms);
    p.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); },
    );
  });
}
