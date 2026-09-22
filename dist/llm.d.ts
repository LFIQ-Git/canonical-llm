export type Provider = "ocp" | "ocp-fallback" | "anthropic";
export type ModelTier = "fast" | "balanced" | "deep";
/** Tier → model ID. One place to bump models family-wide. Keep IDs inside
 *  the OCP proxy allowlist. */
export declare const MODEL_TIERS: Record<ModelTier, string>;
/** Default model when neither `model` nor `tier` is given. */
export declare const DEFAULT_LLM_MODEL: string;
export declare function getProvider(): Provider;
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
    /** Anthropic path only — tokens written to / read from the prompt cache. */
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
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
/**
 * Single round-trip chat. Returns the model's text ("" if none). For token
 * counts or the raw provider response, use `chatDetailed()`.
 */
export declare function chat(args: ChatArgs): Promise<string>;
/**
 * Like `chat()`, but returns the rich result — `{ text, usage, raw, provider }`.
 * Use when you need token counts, the raw response (e.g. stop_reason), or the
 * provider that actually answered after failover.
 */
export declare function chatDetailed(args: ChatArgs): Promise<ChatResult>;
export declare function chatWithRetry(args: ChatArgs): Promise<string>;
export declare function chatDetailedWithRetry(args: ChatArgs): Promise<ChatResult>;
export declare function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T>;
//# sourceMappingURL=llm.d.ts.map