#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rich>=13.0",
#     "tqdm>=4.70.1",
# ]
# ///
"""pi-token-stats: aggregate and display session token statistics.

Walks ~/.pi/agent/sessions/**/*.jsonl, extracts ``token_speed_stats`` entries,
bins by totalTokens (Short <4K, Medium 4–16K, Long 16–64K, XLarge 64K+)
and cache level (cold / warm), then prints a colorized summary table with
median, average, P80, and P90 grouped by (cacheLevel, contextBin).  PrefillTps
is not reported — the measurement is unreliable due to a timing bug in the
extension.

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
from pathlib import Path
from typing import Any
from rich.console import Console
from rich.rule import Rule
from rich.table import Table, Column
from rich.text import Text
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────

# The TTFT measurement for cold entries is unreliable before the calibration
# fix (commit 7470c8c) — values were impossibly small (5–23ms) because
# before_provider_request fires too late in a session.  PrefillTps is
# entirely excluded from the report.  All other metrics (TTFT, decodeTps,
# genTps, costs, token counts) are included normally.

SESSIONS_BASE = Path(
    os.environ.get(
        "PI_TOKEN_SPEED_STATS_DIR",
        str(Path.home() / ".pi" / "agent"),
    )
)
ENTRY_TYPE = "token_speed_stats"

CONTEXT_BINS: list[tuple[str, int | float, int | float]] = [
    ("short", 0, 4_000),
    ("medium", 4_000, 16_000),
    ("long", 16_000, 64_000),
    ("xlarge", 64_000, float("inf")),
]

METRICS = ["ttftMs", "decodeTps", "genTps"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

console = Console(force_terminal=True, color_system="truecolor")


def c(*parts, **kwargs) -> None:
    """Shortcut: print with Rich Console."""
    console.print(*parts, **kwargs)


def colorize_cache_label(label: str) -> Text:
    """Colorize cache level labels."""
    t = Text(label)
    if label == "cold":
        t.stylize("bold red")
    elif label == "cool":
        t.stylize("bold yellow")
    elif label == "warm":
        t.stylize("bold green")
    return t


def colorize_context_label(label: str) -> Text:
    """Colorize context bin labels."""
    t = Text(label)
    colors = {
        "short": "cyan",
        "medium": "yellow",
        "long": "magenta",
        "xlarge": "magenta",
    }
    if label in colors:
        t.stylize(colors[label])
    return t


def colorize_metric_name(name: str) -> Text:
    """Colorize metric names."""
    t = Text(name)
    colors = {"TTFT": "bold yellow", "Decode TPS": "bold cyan", "Gen TPS": "bold green"}
    if name in colors:
        t.stylize(colors[name])
    return t


# TPS performance bands for value colorization (tok/s)
TPS_BANDS: list[tuple[int | float, str]] = [
    (0, "#ff4444"),  # Slow (red)
    (15, "#ffaa00"),  # Medium (orange)
    (30, "#00ff88"),  # Fast (green)
    (45, "#44ddff"),  # Blazing (cyan)
]


def colorize_tps_value(val: float) -> Text:
    """Colorize a TPS value based on performance bands."""
    import re as _re

    s = f"{val:.1f}"
    t = Text(s)
    for threshold, color in TPS_BANDS:
        if val <= threshold:
            t.stylize(color)
            return t
    # Fallback (val > 45)
    t.stylize("#44ddff")
    return t


TTFT_BANDS: list[tuple[int | float, str]] = [
    (45, "#ff4444"),  # Slow (red) — worst TTFT
    (30, "#ffaa00"),  # Medium (orange)
    (15, "#00ff88"),  # Fast (green)
    (0, "#44ddff"),   # Blazing (cyan) — best TTFT
]


def colorize_ttft_value(val: float) -> Text:
    """Colorize a TTFT value — lower (faster) is better.

    P99 (slowest) gets red, P50 (fastest) gets green/cyan.
    Inverse of TPS where high = good.
    Formats via ``fmt_ttft`` and applies performance-band coloring.
    """
    display = fmt_ttft(val)
    # Parse the raw ms value for threshold comparison
    try:
        # Try extracting the raw numeric value (before 's' or 'm' suffixes)
        import re as _re
        match = _re.search(r"([\d.]+)", display)
        ms_val = float(match.group(1)) if match else val
    except (ValueError, AttributeError):
        ms_val = val
    t = Text(display)
    for threshold, color in TTFT_BANDS:
        if ms_val >= threshold:
            t.stylize(color)
            return t
    # Fallback (val <= 0)
    t.stylize("#44ddff")
    return t


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
        raw = input(
            "  Select model(s) by number (comma-separated), or 0 for all: "
        ).strip()
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
                print(
                    f"  [ERROR] Index out of range (1–{len(unique_models)}). Try again."
                )
                continue
            models_selected = [unique_models[idx - 1] for idx in indices]
            return models_selected
        except (ValueError, IndexError):
            remaining = max_retries - attempt
            suffix = (
                f" ({remaining} retry{'ies' if remaining != 1 else ''} left)"
                if remaining
                else ""
            )
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

    Cache level describes whether the KV-cache was leveraged for this request,
    based on the cache hit ratio (cacheReadTokens / inputTokens):

    - **"warm"**: cacheHitRatio ≥ 50 %. A substantial portion of the prompt was
      served from the KV-cache, so the server skips most or all of the prefill
      phase. TTFT is short and total cost is lower.

    - **"cool"**: cacheHitRatio ∈ (0 %, 50 %). Only a partial fraction of the
      prompt was served from cache. TTFT is reduced but the benefit is smaller
      than for a fully warm request.

    - **"cold"**: cacheReadTokens == 0. No tokens were cached, so the full
      prefill pass over every input token was performed. This reflects the
      worst-case latency and cost for this model.
    """
    cacheRead = raw.get("cacheReadTokens") or 0
    inputTok  = raw.get("inputTokens") or 0
    cacheHitRatio = cacheRead / inputTok if inputTok > 0 else 0.0
    if cacheHitRatio == 0:
        return "cold"
    if cacheHitRatio < 0.5:
        return "cool"
    return "warm"


def fmt_range(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">{lo:,}"
    return f"{lo:,}–{hi:,}"


def total_tokens(raw: dict) -> int:
    t = raw.get("totalTokens")
    if t is not None:
        return int(t)
    return (
        int(raw.get("inputTokens") or 0)
        + int(raw.get("outputTokens") or 0)
        + int(raw.get("cacheReadTokens") or 0)
    )


def bin_label(ctx: str, lo: float, hi: float) -> str:
    return f"{ctx} [{fmt_range(lo, hi)}]"


def fmt_ttft(val: float) -> str:
    """Format a TTFT value for display.

    Below 10 000 ms: show raw milliseconds (e.g. "2 847.3").
    ≥ 10 000 ms but < 60 000 ms: show seconds with ms in parentheses
      (e.g. "14.2s  (14200ms)").
    ≥ 60 000 ms: show minutes with ms in parentheses
      (e.g. "1m 40s  (100000ms)").
    """
    if val >= 60_000:
        minutes = int(val // 60_000)
        leftover_ms = val - minutes * 60_000
        seconds_part = leftover_ms / 1000
        return f"{minutes}m {seconds_part:.0f}s  ({val:.0f}ms)"
    elif val >= 10_000:
        return f"{val / 1000:.1f}s  ({val:.0f}ms)"
    return f"{val:.1f}"


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
    """Yield all token_speed_stats entry data dicts from all sessions."""
    sessions_dir = base / "sessions"
    if not sessions_dir.is_dir():
        print(f"[WARN] sessions directory not found: {sessions_dir}", file=sys.stderr)
        return

    seen: set[Path] = set()
    jsonl_files = sorted(sessions_dir.rglob("*.jsonl"))
    for jsonl_file in tqdm(jsonl_files, desc="Scanning", unit="file"):
        resolved = jsonl_file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        count = 0
        for entry in jsonl_entries(jsonl_file):
            if entry.get("type") == "custom" and entry.get("customType") == ENTRY_TYPE:
                count += 1
                yield entry.get("data", {})
        if count and verbose:
            print(f"  {jsonl_file.name}  ({count} entries)")


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
                "totalTokens": [],
                "inputTokens": [],
                "outputTokens": [],
            }

        g = groups[key]
        g["count"] += 1
        for m in METRICS:
            v = raw.get(m)
            if v is not None and isinstance(v, (int, float)):
                g[m].append(float(v))
        g["totalTokens"].append(tt)
        g["inputTokens"].append(int(raw.get("inputTokens") or 0))
        g["outputTokens"].append(int(raw.get("outputTokens") or 0))

    # Filter out groups with fewer than 32 samples
    groups = {k: v for k, v in groups.items() if v["count"] >= 32}

    return groups, all_models


# ─── Display ──────────────────────────────────────────────────────────────────


def display(groups: dict[str, dict], all_models: set[str]) -> None:
    if not groups:
        print(
            "No token_speed_stats entries found. "
            "Run pi with the token-speed-tracker extension first."
        )
        return

    # Header as a Rich Table (rows → columns, with visible borders)
    hdr = Table(show_header=False, show_edge=True)
    hdr.add_column("", style="bold", width=20)
    hdr.add_column("", width=80)
    hdr.add_row("Sessions dir:", str(SESSIONS_BASE / "sessions"))
    hdr.add_row(
        "Total groups:", f"{len(groups)}  (cold/warm × short/medium/long/xlarge)"
    )
    hdr.add_row(
        "Models included:", ", ".join(sorted(all_models)) if all_models else "(unknown)"
    )
    c(hdr)
    c(Rule())

    # Sort: cold before cool before warm, short before xlarge
    order_cache = {"cold": 0, "cool": 1, "warm": 2}
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

        # Parse the group label to colorize cache/context parts
        parts = g["label"].split(" × ")
        cache_part = parts[0] if parts else "cold"
        context_part = parts[1] if len(parts) > 1 else "short [0–4,000]"
        cache_text = colorize_cache_label(cache_part)
        context_text = colorize_context_label(context_part)
        group_title = Text(f"  {cache_text} × {context_part}")
        count_text = Text(f"  (n = {g['count']})")
        count_text.stylize("dim")
        c(f"  {group_title}     ", count_text)
        c()  # blank line

        # Summary table with median, Q1, Q3, IQR, plus mean for reference
        labels = {
            "ttftMs": "TTFT",
            "decodeTps": "Decode TPS",
            "genTps": "Gen TPS",
        }
        units = {
            "ttftMs": "ms",
            "decodeTps": "tok/s",
            "genTps": "tok/s",
        }

        # Build a Rich Table — columns, widths, and alignment handled by Rich.
        table = Table(show_header=True, show_edge=False, expand=False)
        table.add_column("Metric", style="bold", width=16)
        table.add_column("Median", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("P80", justify="right")
        table.add_column("P90", justify="right")

        for m in METRICS:
            vals = g[m]
            if not vals:
                continue
            avg = round(sum(vals) / len(vals), 1)
            s = sorted(vals)
            median = round(_median(s), 1)
            # TPS: p90 = 10th percentile (slow), p80 = 20th percentile
            # TTFT: p90 = 90th percentile (slow), p80 = 80th percentile
            if m in ("decodeTps", "genTps"):
                p80 = round(_percentile(s, 20), 1)   # 20th pct = slow
                p90 = round(_percentile(s, 10), 1)   # 10th pct = slowest
            else:
                p80 = round(_percentile(s, 80), 1)
                p90 = round(_percentile(s, 90), 1)
            if m == "ttftMs":
                table.add_row(
                    colorize_metric_name(labels[m]),
                    colorize_ttft_value(median),
                    colorize_ttft_value(avg),
                    colorize_ttft_value(p80),
                    colorize_ttft_value(p90),
                )
            else:
                table.add_row(
                    colorize_metric_name(labels[m]),
                    colorize_tps_value(median),
                    colorize_tps_value(avg),
                    colorize_tps_value(p80),
                    colorize_tps_value(p90),
                )

        c(table)
        c()  # spacing line after each table

    c(Rule())
    footer = Text("  End of report.")
    footer.stylize("dim")
    c(footer)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate pi token_speed_stats from session JSONL files.",
    )
    parser.add_argument(
        "--sessions",
        type=Path,
        default=None,
        help=f"Base directory (default: {SESSIONS_BASE}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON instead of the formatted report.",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        help="Only include entries matching this model name (substring).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="(deprecated, no longer has any effect — plots have been removed).",
    )
    args = parser.parse_args()  # accepts --no-plot for backward compat, then ignores it

    base = args.sessions or SESSIONS_BASE

    # Phase 1: collect ALL entries (tqdm in find_entries shows real-time progress).
    entries_list = list(find_entries(base, verbose=False))
    all_entries: list[dict[str, Any]] = entries_list

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
                    raw for raw in all_entries if (raw.get("model") or "") in selection
                ]
        else:
            all_raw = all_entries  # no models found, proceed anyway

    if args.json:
        print(json.dumps(all_raw, indent=2, default=str))
        return

    groups, all_models = aggregate(all_raw)
    display(groups, all_models)


if __name__ == "__main__":
    main()
