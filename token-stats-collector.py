#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rich>=13.0",
# ]
# ///
"""pi-token-stats: aggregate and display session token statistics.

Walks ~/.pi/agent/sessions/**/*.jsonl, extracts ``token_speed_stats`` entries,
bins by totalTokens (Short <4K, Medium 4–16K, Long 16–64K, XLarge 64K+)
and cache level (cold / warm), then prints a summary table with median,
quartiles, IQR, and mean grouped by (cacheLevel, contextBin).

Usage:

    uv run --script token-stats-collector.py
    uv run --script token-stats-collector.py --sessions /custom/path
    uv run --script token-stats-collector.py --json  # raw JSON output
    uv run --script token-stats-collector.py --filter-model claude  # CLI substring
    uv run --script token-stats-collector.py  # interactive prompt
    uv run --script token-stats-collector.py --no-plot  # (deprecated, now default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

# The prefillTps fix (commit efac9aa) excluded cache reads from the calculation.
# All data prior to this commit has incorrect prefill TPS values and must be
# excluded from aggregation.  This date is the commit's author date in UTC.
PREFILL_FIX_DATE = datetime(2026, 10, 17, 2, 49, 3, tzinfo=timezone.utc)

# The index.ts extension has a secondary bug: `before_provider_request` fires
# at the wrong wall-clock time for subsequent requests (it fires when the
# internal request is queued, not when the HTTP request is actually sent).
# This makes measured `ttftMs` impossibly small (5–23ms) for cold entries
# with 10K+ tokens, producing prefillTps values of 1–16M tok/s.
#
# The index.ts fix (see commit in index.ts) uses a per-model calibration
# to detect and correct unreliable measured TTFT.  Until that fix is
# deployed and accepted, entries generated between the efac9aa cutoff and
# the index.ts fix are excluded.  This date is set after the index.ts fix
# commit to ensure we only include data from both fixes being live.
COLLECTION_FIX_DATE = datetime(2026, 10, 18, 0, 0, 0, tzinfo=timezone.utc)

SESSIONS_BASE = Path(
    os.environ.get(
        "PI_TOKEN_SPEED_STATS_DIR",
        str(Path.home() / ".pi" / "agent"),
    )
)
ENTRY_TYPE = "token_speed_stats"

CONTEXT_BINS: list[tuple[str, int | float, int | float]] = [
    ("short",    0,       4_000),
    ("medium", 4_000,    16_000),
    ("long",  16_000,     64_000),
    ("xlarge", 64_000,    float("inf")),
]

METRICS = ["ttftMs", "decodeTps", "genTps", "prefillTps"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def collect_unique_models(entries: list[dict]) -> list[str]:
    """Return a sorted list of unique model names from entries."""
    models: set[str] = set()
    for raw in entries:
        m = raw.get("model")
        if m:
            models.add(m)
        else:
            models.add("(unknown)")
    return sorted(models)


def parse_selection(input_str: str) -> list[int] | None:
    """Parse user input like '1,3,5' or '1-3' into a list of 1-based indices.

    Returns None for '0' or 'all'.  Raises ValueError on bad input.
    """
    s = input_str.strip().lower()
    if s == "0" or s == "all":
        return None  # signal: select everything

    indices: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and "," not in part:
            # range like '1-3'
            tokens = part.split("-")
            if len(tokens) != 2:
                raise ValueError(f"bad range: {part!r}")
            lo, hi = int(tokens[0]), int(tokens[1])
            indices.extend(range(lo, hi + 1))
        else:
            indices.append(int(part))
    return indices


def prompt_model_selection(unique_models: list[str]) -> str | list[str]:
    """Interactive prompt for model selection.  Returns 'all' or list of models."""
    print("\n  Available models:")
    print("  " + "─" * 40)
    print(f"  0.  all models")
    for i, model in enumerate(unique_models, 1):
        print(f"  {i}.  {model}")
    print("  " + "─" * 40)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        raw = input("  Select model(s) by number (comma-separated), or 0 for all: ").strip()
        if not raw:
            continue  # treat empty as re-prompt
        try:
            indices = parse_selection(raw)
            if indices is None:
                return "all"
            # validate indices are in range
            if not indices:
                continue  # empty input → re-prompt
            if any(idx < 1 or idx > len(unique_models) for idx in indices):
                print(f"  [ERROR] Index out of range (1–{len(unique_models)}). Try again.")
                continue
            models_selected = [unique_models[idx - 1] for idx in indices]
            return models_selected
        except (ValueError, IndexError):
            remaining = max_retries - attempt
            suffix = f" ({remaining} retry{'ies' if remaining != 1 else ''} left)" if remaining else ""
            print(f"  [ERROR] Invalid input.{suffix} Try again.")

    print("  [ERROR] Max retries reached. Exiting.", file=sys.stderr)
    sys.exit(1)


def context_bin(totalTokens: int) -> str:
    for label, lo, hi in CONTEXT_BINS:
        if lo <= totalTokens < hi:
            return label
    return "xlarge"


def cache_level(raw: dict) -> str:
    """Return the cache level for a session entry.

    Cache level describes whether the KV-cache was leveraged for this request:

    - **"warm"**: cacheReadTokens > 0. Some (or all) input tokens were served
      directly from the prompt cache (KV-cache hit). The server skips the
      prefill phase for cached tokens, so TTFT is shorter and total cost is
      lower. A warm request may still incur prefill work for *new* tokens
      that do not appear in the cache.

    - **"cold"**: cacheReadTokens == 0. No input tokens were cached, so the
      full prefill pass over every input token was performed. This reflects
      the worst-case latency and cost for this model.
    """
    return "warm" if (raw.get("cacheReadTokens") or 0) > 0 else "cold"


def fmt_range(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">{lo:,}"
    return f"{lo:,}–{hi:,}"


def total_tokens(raw: dict) -> int:
    t = raw.get("totalTokens")
    if t is not None:
        return int(t)
    return int(raw.get("inputTokens") or 0) + int(raw.get("outputTokens") or 0) + int(raw.get("cacheReadTokens") or 0)


def bin_label(ctx: str, lo: float, hi: float) -> str:
    return f"{ctx} [{fmt_range(lo, hi)}]"


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def _median(sorted_vals: list[float]) -> float:
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


# ─── File reading ─────────────────────────────────────────────────────────────

TIMESTAMP_FIELDS = ["date", "timestamp", "time", "createdAt", "timeMs"]


def _entry_before_cutoff(raw: dict) -> bool:
    """Return True if a session entry was recorded before either fix."""
    for field in TIMESTAMP_FIELDS:
        val = raw.get(field)
        if val is None:
            continue
        # Handle epoch milliseconds
        if isinstance(val, (int, float)) and val > 1e12:
            try:
                dt = datetime.fromtimestamp(val / 1000, tz=timezone.utc)
                return dt < COLLECTION_FIX_DATE  # use the later of the two dates
            except (OSError, ValueError, OverflowError):
                pass
        # Handle ISO-8601 / RFC-3339 strings
        if isinstance(val, str):
            val = val.strip()
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    dt = datetime.strptime(val, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt < COLLECTION_FIX_DATE  # use the later of the two dates
                except ValueError:
                    continue
    return False


def jsonl_entries(path: Path):
    """Yield parsed JSON objects from a JSONL file."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"  [WARN] malformed JSON at {path}:{lineno}",
                        file=sys.stderr,
                    )
    except (OSError, PermissionError) as e:
        print(f"  [WARN] cannot read {path}: {e}", file=sys.stderr)


def find_entries(base: Path, verbose: bool = True):
    """Yield all token_speed_stats entry data dicts from all sessions.

    Entries whose *date / timestamp* falls before the prefillTps fix
    (commit efac9aa, 2026-10-17) are silently excluded, because their
    prefill TPS values are known to be incorrect.
    """
    sessions_dir = base / "sessions"
    if not sessions_dir.is_dir():
        print(f"[WARN] sessions directory not found: {sessions_dir}",
              file=sys.stderr)
        return

    seen: set[Path] = set()
    for jsonl_file in sorted(sessions_dir.rglob("*.jsonl")):
        resolved = jsonl_file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        count = 0
        filtered_out = 0
        for entry in jsonl_entries(jsonl_file):
            if (entry.get("type") == "custom"
                    and entry.get("customType") == ENTRY_TYPE):
                raw = entry.get("data", {})
                if _entry_before_cutoff(raw):
                    filtered_out += 1
                    continue
                count += 1
                yield raw
        if count and verbose:
            extra = (f" (excluded {filtered_out} pre-fix entries)"
                     if filtered_out else "")
            print(f"  {jsonl_file.name}  ({count} entries){extra}")


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(entries: list[dict]) -> tuple[dict[str, dict], set[str]]:
    groups: dict[str, dict] = {}
    all_models: set[str] = set()

    for raw in entries:
        tt = total_tokens(raw)
        cl = cache_level(raw)
        ctx = context_bin(tt)
        model = raw.get("model")
        if model:
            all_models.add(model)

        # Find the bin bounds for labeling
        lo, hi = 0, 0
        for lbl, b_lo, b_hi in CONTEXT_BINS:
            if lbl == ctx:
                lo, hi = b_lo, b_hi
                break

        key = f"{cl}|{ctx}|{lo}|{hi}"
        if key not in groups:
            groups[key] = {
                "label": f"{cl} × {bin_label(ctx, lo, hi)}",
                "count": 0,
                "ttftMs": [],
                "decodeTps": [],
                "genTps": [],
                "prefillTps": [],
                "totalTokens": [],
                "inputTokens": [],
                "outputTokens": [],
                "models": set[str](),
            }

        g = groups[key]
        g["count"] += 1
        if model:
            g["models"].add(model)
        for m in METRICS:
            v = raw.get(m)
            if v is not None and isinstance(v, (int, float)):
                g[m].append(float(v))
        g["totalTokens"].append(tt)
        g["inputTokens"].append(int(raw.get("inputTokens") or 0))
        g["outputTokens"].append(int(raw.get("outputTokens") or 0))

    return groups, all_models


# ─── Display ──────────────────────────────────────────────────────────────────

def display(groups: dict[str, dict], all_models: set[str]) -> None:
    if not groups:
        print(
            "No token_speed_stats entries found. "
            "Run pi with the token-speed-tracker extension first."
        )
        return

    sep = "=" * 72
    thin_sep = "-" * 72

    print(f"\n{sep}")
    print("  PI TOKEN STATISTICS — Aggregated from session JSONL")
    print(f"  Sessions dir: {SESSIONS_BASE / 'sessions'}")
    print(f"  Total groups: {len(groups)}  "
          "(cold/warm × short/medium/long/xlarge)")
    if all_models:
        print(f"  Models included:  " + ", ".join(sorted(all_models)))
    else:
        print("  Models included:  (unknown)")
    print(sep)

    # Sort: cold before warm, short before xlarge
    order_cache = {"cold": 0, "warm": 1}
    order_bin = {"short": 0, "medium": 1, "long": 2, "xlarge": 3}
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (
            order_cache.get(k.split("|")[0], 99),
            order_bin.get(k.split("|")[1], 99),
        ),
    )

    for key in sorted_keys:
        g = groups[key]
        if g["count"] == 0:
            continue

        print(f"\n{thin_sep}")
        model_list = g.get("models")
        model_info = f"  Models: {', '.join(sorted(model_list))}" if model_list else "  Models: (unknown)"
        print(f"  {g['label']}     (n = {g['count']})\n{model_info}")

        # Summary table with median, Q1, Q3, IQR, plus mean for reference
        labels = {
            "ttftMs": "TTFT",
            "decodeTps": "Decode TPS",
            "genTps": "Gen TPS",
            "prefillTps": "Prefill TPS",
        }
        units = {
            "ttftMs": "ms",
            "decodeTps": "tok/s",
            "genTps": "tok/s",
            "prefillTps": "tok/s",
        }

        hdr = (f"  {'Metric':<16}"
               f" {'Median':>10}  {'Q1':>10}  {'Q3':>10}  {'IQR':>10}"
               f" {'Avg':>10}  {'Min':>12}  {'Max':>12}  {'Units':>10}")
        sep_line = (f"  {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}"
                    f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
        print(hdr)
        print(sep_line)

        for m in METRICS:
            vals = g[m]
            if not vals:
                continue
            avg = round(sum(vals) / len(vals), 1)
            mn, mx = round(min(vals), 1), round(max(vals), 1)
            s = sorted(vals)
            median = round(_median(s), 1)
            q1 = round(_percentile(s, 25), 1)
            q3 = round(_percentile(s, 75), 1)
            iqr = round(q3 - q1, 1)
            print(f"  {labels[m]:<16} {median:>10.1f}  {q1:>10.1f}  {q3:>10.1f}  {iqr:>10.1f}"
                  f"  {avg:>10.1f}  {mn:>12.1f}  {mx:>12.1f}  "
                  f"{units[m]:>10}")

    print(f"\n{sep}")
    print("  End of report.")
    print(sep)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate pi token_speed_stats from session JSONL files.",
    )
    parser.add_argument(
        "--sessions", type=Path, default=None,
        help=f"Base directory (default: {SESSIONS_BASE}).",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
        help="Output raw JSON instead of the formatted report.",
    )
    parser.add_argument(
        "--filter-model", type=str, default=None,
        help="Only include entries matching this model name (substring).",
    )
    parser.add_argument(
        "--no-plot", action="store_true", default=False,
        help="(deprecated, no longer has any effect — plots have been removed).",
    )
    args = parser.parse_args()  # accepts --no-plot for backward compat, then ignores it

    base = args.sessions or SESSIONS_BASE

    # Phase 1: collect ALL entries to discover models
    all_entries: list[dict[str, Any]] = []
    for raw in find_entries(base, verbose=not args.json):
        all_entries.append(raw)

    # Phase 2: apply model filtering
    if args.filter_model:
        # CLI flag — substring match, case-insensitive
        all_raw: list[dict[str, Any]] = []
        for raw in all_entries:
            model = (raw.get("model") or "").lower()
            if args.filter_model.lower() not in model:
                continue
            all_raw.append(raw)
    else:
        # Interactive selection (only when there are models to filter on)
        unique_models = collect_unique_models(all_entries)
        if unique_models:
            selection = prompt_model_selection(unique_models)
            if selection == "all":
                all_raw = all_entries  # no filtering
            else:
                all_raw = [
                    raw for raw in all_entries
                    if (raw.get("model") or "") in selection
                ]
        else:
            all_raw = all_entries  # no models found, proceed anyway

    if args.json:
        print(json.dumps(all_raw, indent=2, default=str))
        return

    if not all_raw:
        print("No token_speed_stats entries found. "
              "Run pi with the token-speed-tracker extension first.")
        return

    groups, all_models = aggregate(all_raw)
    display(groups, all_models)


if __name__ == "__main__":
    main()
