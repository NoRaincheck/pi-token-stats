# pi-token-speed-tracker

An [extension for Pi](https://pi.dev) that records token speed statistics into session JSONL files — TTFT, decode throughput, prompt tokens, costs, and more.

## What it captures

Every assistant message in a Pi session gets a `custom` entry with type `token_speed_stats`:

```json
{
  "type": "custom",
  "customType": "token_speed_stats",
  "id": "a1b2c3d4",
  "parentId": "...",
  "timestamp": "2025-06-18T14:22:03.456Z",
  "data": {
    "providerId": "anthropic",
    "api": "anthropic-messages",
    "model": "claude-sonnet-4-5",
    "requestStartAt": 1718712120345,
    "messageStartAt": 1718712122168,
    "lastChunkAt": 1718712124920,
    "ttftMs": 1823.4,
    "decodeSeconds": 2.552,
    "decodeTokens": 341,
    "genTps": 134,
    "decodeTps": 67.5,
    "prefillTps": 4609.1,
    "cacheAwarePrefillTokens": 82,
    "liveSeconds": 2.552,
    "inputTokens": 8420,
    "outputTokens": 342,
    "cacheReadTokens": 0,
    "totalTokens": 8762,
    "costInput": 0.2105,
    "costOutput": 0.0103,
    "costTotal": 0.2208,
    "intermediateToolCalls": 2,
    "stopReason": "stop"
  }
}
```

At the end of each turn (user prompt → final AI response), an aggregate summary is also written:

```json
{
  "type": "custom",
  "customType": "token_speed_stats_turn_summary",
  "data": {
    "timestamp": "...",
    "messageCount": 3,
    "totalInputTokens": 8420,
    "totalOutputTokens": 1205,
    "avgTTFTMs": 1890,
    "totalGenerationSeconds": 12.4,
    "providerId": "anthropic",
    "models": ["claude-sonnet-4-5"]
  }
}
```

## Metrics explained

| Field | Description |
|-------|-------------|
| `ttftMs` | Time-to-first-token — wall time from provider request to first streamed chunk |
| `decodeTps` | Decode throughput (tokens/second) during the live streaming window, excluding the first chunk |
| `decodeTps` | Decode throughput (tokens/second) during the live streaming window, excluding the first chunk |
| `prefillTps` | Effective prefill throughput: inputTokens / TTFT seconds. Includes network + queue latency — reflects end-to-end client experience, not pure server-side compute. |
| `cacheAwarePrefillTokens` | Cache-aware effective prefill tokens (input − cache reads). When most of a prompt was served from KV-cache, raw TTFT is misleadingly small — this field estimates actual compute work instead. Absent if no cache reads reported. |
| `inputTokens` | Prompt tokens sent to the provider |
| `outputTokens` | Tokens returned by the model |
| `cacheReadTokens` | Cached prompt tokens from provider cache |
| `totalTokens` | Sum of all token categories |
| `costTotal` | Total cost in USD for this message |
| `intermediateToolCalls` | Tool calls that ran between this request and response (useful when model uses tools) |

## Setup

### Quick start — load an extension file

Place the extension where Pi can find it:

```bash
# Personal extensions (recommended)
cp index.ts ~/.pi/agent/extensions/token-speed-tracker.ts

# Or in your project directory
mkdir -p .pi/agent/extensions && cp index.ts .pi/agent/extensions/
```

Then start Pi — it auto-loads all `.ts` files from the extensions directory:

```bash
cd /your/project
pi --extension ./index.ts
# Or for personal config, just run:
pi
```

### Configuration (optional)

Edit `~/.pi/agent/extensions/token-speed-tracker.ts` and modify these settings inside the factory:

| Config | Default | Description |
|--------|---------|-------------|
| `ewmaAlpha` | 0.3 | EWMA smoothing for character→token ratio calibration |

## Usage examples

### Reading stats from a session file

```bash
# Find your session files
find ~/.pi/agent/sessions -name '*.jsonl' -newer /tmp/last_run

# Extract token speed stats (JSONL → JSON)
cat ~/.pi/agent/sessions/**/2025-*.jsonl \
  | jq -s '[.[] | select(.customType == "token_speed_stats") | .data]' \
  > stats.json
```

### Computing average TTFT across a session

```bash
jq 'map(.ttftMs) | {avg: (add / length), min: min, max: max, count: length}' stats.json
```

### Grouping by model

```bash
jq -s 'group_by(.model) | map({model: .[0].model, avgTTFTMs: (map(.ttftMs) | add/length), totalOutputTokens: (map(.outputTokens) | add)})' stats.json
```

### Comparing prefill throughput across providers

```bash
jq -s 'group_by(.providerId) | map({provider: .[0].providerId, avgPrefillTps: (map(.prefillTps // 0) | add/length), minTTFTMs: (map(.ttftMs | select(.)) | min)})' stats.json
```



## Calibration

The extension maintains a persisted `chars-per-token` ratio per model to estimate decode tokens in the live window more accurately. State is saved to `~/.pi/agent/token-speed-stats-state.json` and survives restarts. Override the directory with:

```bash
PI_TOKEN_SPEED_STATS_DIR=/custom/path pi
```

## Entry types in session JSONL

| customType | When written | Purpose |
|------------|--------------|---------|
| `token_speed_stats` | After each assistant message | Per-message timing and token stats |
| `token_speed_stats_turn_summary` | At end of each turn | Aggregate across all messages in the turn |

## Notes

- **TTFT** includes network latency, server queue time, prefill time, and first-chunk decode time. It is not pure prefill throughput.
- **Prefill TPS (`prefillTps`)** = `inputTokens / (ttftMs / 1000)`. This is an effective client-side estimate — it reflects end-to-end experience from request send to first token arrival, which includes network round-trip time and server queue wait. It should not be interpreted as pure server-side prefill compute throughput.
- **Cache-aware prefill (`cacheAwarePrefillTokens`)** subtracts cache-read tokens from input to estimate actual compute work during prefill. When most of a prompt is served from KV-cache, raw TTFT can be very small — this field helps distinguish "fast because cached" from "fast because good hardware/network." Use `prefillTps` computed against just the uncached fraction for that comparison.
- **Decode TPS** excludes tokens delivered in the very first chunk (which inflate the initial rate). It measures only the sustained streaming phase.
- No status bar or live display — all data is recorded silently into the session JSONL for post-hoc analysis.
- **vLLM Prometheus metrics** (prefill tokens/TPS) are supported by the reference implementation but require a separate metrics endpoint configuration — not included here since the extension operates purely through Pi event hooks without needing network access.
