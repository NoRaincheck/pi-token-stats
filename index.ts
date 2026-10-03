/**
 * Token Speed Tracker — per-turn statistics persisted to session JSONL.
 *
 * Records time-based performance metrics (TTFT, decode speed, prompt tokens)
 * for every assistant message produced during a Pi session.  Data is persisted
 * as `custom` entries so it survives resume / fork and can be queried later by
 * any tool or script that reads the session file.
 *
 * Each line in `.pi/agent/sessions/**.jsonl`:
 *
 *   {"type":"custom","customType":"token_speed_stats",
 *    "data":{"providerId":"anthropic","model":"claude-sonnet-4-5",
 *            "ttftMs":1823.4,"decodeTps":67.5,
 *            "inputTokens":8420,"outputTokens":342,"costTotal":0.027}}
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// ─── Types ───────────────────────────────────────────────────────────────────

/** A single assistant-message timing record stored in the session JSONL. */
export interface TokenSpeedStatsEntry {
  /** Provider identifier (e.g. `"anthropic"`, `"openai"`). */
  providerId: string;
  /** API variant (e.g. `"anthropic-messages"`, `"openai-completions"`). */
  api: string;
  /** Model name used for this response. */
  model: string;

  /** Wall-clock timestamps in milliseconds (Date.now()). */
  timestamp: number;                // entry creation time
  requestStartAt: number | null;    // before_provider_request
  messageStartAt: number | null;    // first chunk arrived on message_start
  lastChunkAt: number | null;       // last streamed chunk

  /** Time-to-first-token in milliseconds. */
  ttftMs?: number;
  /** Is TTFT exact or estimated? */
  ttftExact?: boolean;
  /** Decode phase duration (seconds): `(lastChunkAt − messageStartAt) / 1000`. */
  decodeSeconds?: number;

  /** Live-window tokens (outputTokens minus first-token chunk, if estimable). */
  decodeTokens?: number;

  /** Generation throughput: outputTokens / full generation wall time. */
  genTps?: number;
  /** Decode-only throughput through the live window. */
  decodeTps?: number;
  /** Effective prefill throughput: inputTokens / TTFT seconds.
   *  Includes network + queue latency — reflects end-to-end client experience,
   *  not pure server-side prefill compute time. */
  prefillTps?: number;
  /** Cache-aware effective prefill tokens (input − cache reads).
   *  Useful for estimating actual compute when a large fraction of input
   *  was served from KV-cache, making raw TTFT misleadingly small. */
  cacheAwarePrefillTokens?: number;
  /** Live-window duration in seconds (same as decodeSeconds but named for clarity). */
  liveSeconds?: number;

  /** Input tokens from provider usage object. */
  inputTokens?: number;
  /** Cache-read tokens from provider usage object. */
  cacheReadTokens?: number;
  /** Total output tokens from provider usage object. */
  outputTokens?: number;
  /** Convenience: sum of all token categories. */
  totalTokens?: number;

  /** Cost breakdown (USD). */
  costInput?: number;
  costOutput?: number;
  costCacheRead?: number;
  costTotal?: number;

  /** Tool calls that ran between this message's request and its completion. */
  intermediateToolCalls?: number;
  /** Provider stop reason (`stop`, `tool_calls`, etc.). */
  stopReason?: string;
}

/** Aggregated summary for all assistant messages in one turn. */
export interface TokenSpeedTurnSummary {
  timestamp: number;
  messageCount: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCacheReadTokens: number;
  totalTokens: number;
  avgTTFTMs: number;
  minTTFTMs?: number;
  maxTTFTMs?: number;
  totalGenerationSeconds?: number;
  providerId: string;
  models: string[];
}

// ─── Constants ───────────────────────────────────────────────────────────────

const ENTRY_TYPE = "token_speed_stats";
const SUMMARY_ENTRY_TYPE = `${ENTRY_TYPE}_turn_summary`;
const CALIBRATION_KEY = "chars-per-token-calibration";

function calibrationPath(): string {
  const dir = process.env.PI_TOKEN_SPEED_STATS_DIR || join(homedir(), ".pi", "agent");
  return join(dir, "token-speed-stats-state.json");
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function isFiniteNumber(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function safeDivide(numerator: number, denominator: number): number | undefined {
  if (denominator <= 0) return undefined;
  const result = numerator / denominator;
  return Number.isFinite(result) && result >= 0 ? round(result, 2) : undefined;
}

function round(value: number, decimals = 1): number {
  return Math.round(value * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

// ─── Calibration (persisted chars/token ratio per model) ─────────────────────

interface CalibrationState {
  version: number;
  models: Record<string, CalibrationBucket>;
}

interface CalibrationBucket {
  charsPerToken: number;
  samples: number;
}

let calibrationState: CalibrationState | null = loadCalibration();

function loadCalibration(): CalibrationState {
  const path = calibrationPath();
  if (!existsSync(path)) return { version: 1, models: {} };
  try {
    const raw = JSON.parse(readFileSync(path, "utf8")) as Partial<CalibrationState>;
    return { version: 1, models: (raw.models ?? {}) } satisfies CalibrationState;
  } catch {
    return { version: 1, models: {} };
  }
}

function saveCalibration(state: CalibrationState): void {
  try {
    const path = calibrationPath();
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, JSON.stringify(state, null, 2));
  } catch { /* best-effort */ }
}

function getCalibrationBucket(state: CalibrationState, modelKey: string): CalibrationBucket {
  if (!state.models[modelKey]) {
    state.models[modelKey] = { charsPerToken: 3.5, samples: 0 };
  }
  return state.models[modelKey];
}

function updateCalibration(
  state: CalibrationState, modelKey: string, alpha: number,
  chars: number, tokens: number,
): void {
  if (tokens <= 0) return;
  const observed = chars / tokens;
  if (!Number.isFinite(observed) || observed < 1 || observed > 16) return;
  const bucket = getCalibrationBucket(state, modelKey);
  bucket.charsPerToken = bucket.samples === 0
    ? round(observed)
    : round(bucket.charsPerToken * (1 - alpha) + observed * alpha);
  bucket.samples++;
}

function charsToTokens(state: CalibrationState, modelKey: string, chars: number): number {
  const bucket = getCalibrationBucket(state, modelKey);
  return Math.max(0.5, chars / Math.max(1, bucket.charsPerToken));
}

// ─── Per-request tracking state ──────────────────────────────────────────────

let pendingProviderId: string | null = null;
let pendingApi: string | null = null;
let pendingModel: string | null = null;

interface StreamingMessage {
  requestStartAt: number | null;
  firstChunkAt: number;
  lastChunkAt: number;
  streamedCharsText: number;
  streamedCharsThinking: number;
  streamedCharsToolCall: number;
  exactOutputTokensOnStream: number;
}

let activeMessage: StreamingMessage | null = null;
let intermediateToolCallsForCurrentResponse = 0;

function modelKey(providerId: string, api: string, model: string): string {
  return `${providerId}/${api}/${model}`;
}

// ─── Main factory ────────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // Per-turn accumulator for aggregation at turn_end.
  const turnStatsCollector: TokenSpeedStatsEntry[] = [];

  /** Build and persist stats from an assistant message. */
  function buildAndSaveStats(
    providerId: string, api: string, model: string,
    inputTokens?: number, outputTokens?: number, cacheReadTokens?: number,
  ): TokenSpeedStatsEntry | null {
    if (!activeMessage || !calibrationState) return null;

    const m = activeMessage;
    const mk = modelKey(providerId, api, model);

    // TTFT.
    let ttftMs: number | undefined;
    if (m.requestStartAt !== null) {
      ttftMs = round(m.firstChunkAt - m.requestStartAt, 1);
    }

    // Decode timing.
    const decodeSeconds = round((m.lastChunkAt - m.firstChunkAt) / 1000, 3);
    const liveSeconds = decodeSeconds;

    // Live-window tokens (exclude first chunk).
    let decodeTokens: number | undefined;
    if (outputTokens !== undefined && outputTokens > 0) {
      const totalChars = m.streamedCharsText + m.streamedCharsThinking + m.streamedCharsToolCall;
      const approxFirstChunk = Math.ceil(
        (totalChars > 0 ? totalChars : 32) / getCalibrationBucket(calibrationState, mk).charsPerToken
      );
      decodeTokens = Math.max(0, outputTokens - Math.min(outputTokens, approxFirstChunk));
    }

    // Throughput rates.
    const genTps = safeDivide(outputTokens ?? 0, m.requestStartAt !== null ? (m.lastChunkAt - m.requestStartAt) / 1000 : undefined);
    const decodeTps = decodeTokens !== undefined && liveSeconds > 0
      ? safeDivide(decodeTokens, liveSeconds)
      : undefined;

    // Prefill throughput — effective client-side estimate.
    let prefillTps: number | undefined;
    let cacheAwarePrefillTokens: number | undefined;
    if (isFiniteNumber(ttftMs) && ttftMs > 0 && inputTokens !== undefined && inputTokens > 0) {
      const ttftSeconds = ttftMs / 1000;
      prefillTps = round(inputTokens / ttftSeconds, 2);
      // Cache-aware: subtract cached reads to estimate actual compute work.
      if (cacheReadTokens !== undefined && cacheReadTokens > 0) {
        const aware = inputTokens - cacheReadTokens;
        if (aware > 0) {
          cacheAwarePrefillTokens = aware;
        }
      }
    }

    // Calibration update.
    if (outputTokens !== undefined && outputTokens > 0) {
      const totalChars = m.streamedCharsText + m.streamedCharsThinking + m.streamedCharsToolCall;
      if (totalChars > 0) {
        updateCalibration(calibrationState, mk, 0.3, totalChars, outputTokens);
      }
    }

    const entry: TokenSpeedStatsEntry = {
      providerId, api, model, timestamp: Date.now(),
      requestStartAt: m.requestStartAt,
      messageStartAt: m.firstChunkAt > 0 ? m.firstChunkAt : null,
      lastChunkAt: m.lastChunkAt > 0 ? m.lastChunkAt : null,
      ttftMs, decodeSeconds, decodeTokens,
      genTps, decodeTps, prefillTps, cacheAwarePrefillTokens, liveSeconds,
      inputTokens, outputTokens, cacheReadTokens,
      totalTokens: (inputTokens ?? 0) + (outputTokens ?? 0) + (cacheReadTokens ?? 0),
      intermediateToolCalls: intermediateToolCallsForCurrentResponse || undefined,
    };

    pi.appendEntry<TokenSpeedStatsEntry>(ENTRY_TYPE, entry);
    return entry;
  }

  /** Reset everything at the start of each turn. */
  pi.on("turn_start", () => {
    activeMessage = null;
    pendingProviderId = null;
    pendingApi = null;
    pendingModel = null;
    intermediateToolCallsForCurrentResponse = 0;
  });

  pi.on("session_shutdown", () => {
    if (calibrationState) saveCalibration(calibrationState);
  });

  pi.on("model_select", () => {
    activeMessage = null;
  });

  /** Count tool calls between assistant messages. */
  pi.on("tool_call", () => {
    intermediateToolCallsForCurrentResponse++;
  });

  // ── Provider request lifecycle ───────────────────────────────────────

  let pendingRequestAt: number | null = null;

  pi.on("before_provider_request", () => {
    pendingRequestAt = Date.now();
  });

  // ── Assistant message streaming ──────────────────────────────────────

  pi.on("message_start", (event) => {
    if (event.message.role !== "assistant") return;

    const providerId = event.message.provider ?? pendingProviderId;
    const api = event.message.api ?? pendingApi;
    const model = event.message.model ?? pendingModel;

    if (!pendingProviderId) {
      pendingProviderId = providerId;
      pendingApi = api;
      pendingModel = model;
    }

    activeMessage = {
      requestStartAt: pendingRequestAt,
      firstChunkAt: Date.now(),
      lastChunkAt: 0,
      streamedCharsText: 0,
      streamedCharsThinking: 0,
      streamedCharsToolCall: 0,
      exactOutputTokensOnStream: 0,
    };
  });

  pi.on("message_update", (event) => {
    if (!activeMessage || event.message.role !== "assistant") return;

    activeMessage.lastChunkAt = Date.now();

    // Count characters by delta type.
    const rawEvent = event as Record<string, unknown>;
    const deltaType = rawEvent.streamEvent?.type as string | undefined;
    const delta = event.delta as string | undefined;
    if (typeof delta === "string") {
      switch (deltaType) {
        case "thinking_delta": activeMessage.streamedCharsThinking += delta.length; break;
        case "text_delta":     activeMessage.streamedCharsText += delta.length; break;
        case "tool_call_delta":
        case "tool_calls_delta":
          activeMessage.streamedCharsToolCall += delta.length; break;
      }
    }

    // Exact output tokens from partial usage (Gemini-style).
    const partialOutput = rawEvent.partial?.usage?.output as number | undefined;
    if (isFiniteNumber(partialOutput) && partialOutput > activeMessage.exactOutputTokensOnStream) {
      activeMessage.exactOutputTokensOnStream = Math.round(partialOutput);
    }
  });

  pi.on("message_end", (event, ctx) => {
    if (!activeMessage || event.message.role !== "assistant") return;

    const providerId = pendingProviderId ?? "";
    const api = pendingApi ?? "";
    const model = pendingModel ?? "";

    // Get usage data from the finalized message.
    const u = (event.message.usage as Record<string, unknown>) || {};
    const inputTokens = isFiniteNumber(u.input) ? Math.round(Number(u.input)) : undefined;
    const outputTokens = isFiniteNumber(u.output) ? Math.round(Number(u.output)) : undefined;
    const cacheReadTokens = isFiniteNumber(u.cacheRead) ? Math.round(Number(u.cacheRead)) : undefined;

    // Cost breakdown.
    const cost = (u.cost as Record<string, unknown>) || {};
    const getCost = (key: string): number | undefined => {
      if (!cost[key]) return undefined;
      const n = Number(cost[key]);
      return isFiniteNumber(n) ? round(n, 6) : undefined;
    };

    // Build and persist the stats entry.
    const stats = buildAndSaveStats(
      providerId, api, model,
      inputTokens, outputTokens, cacheReadTokens,
    );

    if (stats && event.message.stopReason) {
      stats.stopReason = event.message.stopReason;
    }

    // Push into the turn accumulator.
    if (stats) {
      turnStatsCollector.push(stats);
    }

    // Reset per-message state.
    activeMessage = null;
    intermediateToolCallsForCurrentResponse = 0;
  });

  /** Emit an aggregate turn summary from all accumulated stats. */
  pi.on("turn_end", () => {
    if (turnStatsCollector.length === 0) return;

    const totalInputTokens = turnStatsCollector.reduce((s, t) => s + (t.inputTokens ?? 0), 0);
    const totalOutputTokens = turnStatsCollector.reduce((s, t) => s + (t.outputTokens ?? 0), 0);
    const totalCacheReadTokens = turnStatsCollector.reduce((s, t) => s + (t.cacheReadTokens ?? 0), 0);

    const ttftValues = turnStatsCollector.map(t => t.ttftMs).filter((v): v is number => v !== undefined && v > 0);
    const avgTTFT = ttftValues.length > 0 ? round(ttftValues.reduce((a, b) => a + b, 0) / ttftValues.length) : 0;

    const firstRequestStart = Math.min(...turnStatsCollector.map(t => t.requestStartAt ?? Infinity));
    const lastMessageEnd = Math.max(...turnStatsCollector.filter(t => t.lastChunkAt).map(t => t.lastChunkAt!));
    const totalGenSeconds = (lastMessageEnd > 0 && firstRequestStart !== Infinity)
      ? round((lastMessageEnd - firstRequestStart) / 1000, 2)
      : undefined;

    const providerId = turnStatsCollector[0]?.providerId ?? "";
    const models = [...new Set(turnStatsCollector.map(t => t.model))];

    pi.appendEntry<TokenSpeedTurnSummary>(SUMMARY_ENTRY_TYPE, {
      timestamp: Date.now(),
      messageCount: turnStatsCollector.length,
      totalInputTokens,
      totalOutputTokens,
      totalCacheReadTokens,
      totalTokens: totalInputTokens + totalOutputTokens + totalCacheReadTokens,
      avgTTFTMs: avgTTFT,
      minTTFTMs: ttftValues.length > 0 ? Math.min(...ttftValues) : undefined,
      maxTTFTMs: ttftValues.length > 0 ? Math.max(...ttftValues) : undefined,
      totalGenerationSeconds: totalGenSeconds,
      providerId,
      models,
    });

    // Clear the accumulator for next turn.
    turnStatsCollector.length = 0;
  });
}
