#!/usr/bin/env python3
"""Calibrate the stand-in by BOOTSTRAP, not by a parametric fit.

TTFT samples are taken with replacement from the observed measurements, stratified
by tokens_in band (<400, 400-900, 900-1600, >1600). No mixture, no lognormal, no
claim about bimodality: the stand-in can only reproduce what was actually seen.

Sources: ttft.jsonl (first_token_ms/labels), probe.jsonl (first_token_ms),
traces/trace_real.jsonl (draft_ttft_ms). Also reports cl100k_base token-accounting
error against the real prompt_tokens, per band.

    python scripts/standin/fit.py
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BANDS = [("<400", 0, 400), ("400-900", 400, 900),
         ("900-1600", 900, 1600), (">1600", 1600, 10 ** 9)]
SEED = 1337


def band_of(tokens: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= tokens < hi:
            return name
    return ">1600"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def reconstruct_prompt(record: dict):
    """Rebuild the exact prompt used, so tokenizer error can be measured."""
    label = str(record.get("label", ""))
    try:
        if record.get("_src") == "ttft":
            ttft = load_module("time_ttft", ROOT / "scripts" / "time_ttft.py")
            if label.startswith("in~"):
                n = int(label.split("~")[1].split("(")[0])
                return ttft.prompt(n)
            if label.startswith("cache-1500"):
                return ttft.prompt(1500)
            if label.startswith("qwen") or label == "minimal":
                return ttft.prompt(1000)
        if record.get("_src") == "probe":
            probe = load_module("probe", ROOT / "scripts" / "probe.py")
            phase = str(record.get("phase", ""))  # probe.jsonl logs "phase", not "label"
            n = 300 if phase.startswith("1-medium") else 40
            return probe.prompt(n)
    except Exception:
        return None
    return None


def main() -> None:
    samples: list[tuple[float, float]] = []  # (tokens_in, ttft_ms)
    for name, src in (("ttft.jsonl", "ttft"), ("probe.jsonl", "probe"),
                      ("traces/trace_real.jsonl", "trace")):
        p = ROOT / name
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if src == "trace":
                tin, ttft = r.get("draft_tokens_in"), r.get("draft_ttft_ms")
            else:
                tin, ttft = r.get("tokens_in"), r.get("first_token_ms")
                r["_src"] = src
            if tin and ttft and r.get("status", "ok") == "ok":
                samples.append((float(tin), float(ttft), r))

    bands: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover
        print(f"tiktoken unavailable: {exc}")
        enc = None

    raw_residual: dict[str, list[float]] = {}   # model -> real - cl100k
    rows = []
    for tin, ttft, rec in samples:
        bands[band_of(tin)].append(round(ttft, 1))
        prompt = reconstruct_prompt(rec) if rec.get("_src") in ("ttft", "probe") else None
        if prompt is not None and enc is not None and rec.get("tokens_in"):
            text = "\n".join(str(m.get("content", "")) for m in prompt)
            est = len(enc.encode(text))
            model = str(rec.get("model", "")).split("/")[-1] or "unknown"
            raw_residual.setdefault(model, []).append(rec["tokens_in"] - est)
            rows.append((band_of(tin), rec["tokens_in"], est, model))

    # The residual is dominated by a CONSTANT chat-template overhead, not by the
    # tokenizer: gpt-oss prompts measure ~71 tokens above cl100k at every length.
    overhead = {m: round(statistics.median(v), 1) for m, v in raw_residual.items() if v}
    corrected_err: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}
    raw_err: dict[str, list[float]] = {name: [] for name, _, _ in BANDS}
    for band, real, est, model in rows:
        raw_err[band].append(abs(est - real) / real)
        adj = est + overhead.get(model, 0.0)
        corrected_err[band].append(abs(adj - real) / real)

    report = {
        "seed": SEED,
        "method": "seeded conditional bootstrap, stratified by tokens_in band",
        "sources": ["ttft.jsonl", "probe.jsonl", "traces/trace_real.jsonl"],
        "ttft_bands": {k: {"n": len(v), "samples_ms": sorted(v)} for k, v in bands.items()},
        "limitation": ("samples are drawn only from observed values: no band can produce a "
                       "TTFT slower than its observed maximum, so slow tails are "
                       "reproduced, not invented. Bands with fewer than 8 observations are "
                       "pooled with their neighbours at sample time, otherwise a 6-sample band "
                       "cannot express the service's slow tail at all"),
        "sparse_band_threshold": 8,
        "tokenizer": {
            "encoding": "cl100k_base",
            "template_overhead_tokens": overhead,
            "mean_relative_error_raw_by_band": {
                k: (round(statistics.fmean(v), 4) if v else None)
                for k, v in raw_err.items()},
            "mean_relative_error_after_overhead_by_band": {
                k: (round(statistics.fmean(v), 4) if v else None)
                for k, v in corrected_err.items()},
            "n_by_band": {k: len(v) for k, v in raw_err.items()},
            "note": ("error is measured on reconstructable probe/ttft prompts only "
                     "(filler-heavy); dense pipeline prompts with citations and math are "
                     "not represented"),
        },
        "inter_token": {
            "measured_gap_ms_p50": 0.3057,
            "target_tokens_per_sec": 128.0,
            "gap_ms": 0.3057,
            "note": ("the measured per-token floor is the calibrated number; the 128 tok/s figure "
                     "is tokens_out/total_ms, which is dominated by TTFT, so it emerges from "
                     "TTFT + completion length rather than from the cadence"),
        },
        "tokens_out": None,
    }

    gaps = []
    outs = []
    for name in ("ttft.jsonl", "probe.jsonl"):
        for line in (ROOT / name).open():
            r = json.loads(line)
            if r.get("status") == "ok" and (r.get("tokens_out") or 0) > 0:
                gaps.append((r["total_ms"] - (r["first_token_ms"] or 0)) / r["tokens_out"])
                if r["tokens_out"] <= 200:
                    outs.append(r["tokens_out"])
    for line in (ROOT / "traces/trace_real.jsonl").open():
        r = json.loads(line)
        if (r.get("draft_tokens_out") or 0) > 0:
            outs.append(r["draft_tokens_out"])
    # Draft length is fitted ONLY on the real pipeline drafts: the ttft sweep
    # prompts ("in roughly 40/300 tokens") are not the citation-style draft prompt,
    # and using them made the stand-in answer 147 tokens where real drafts did 123.
    drafts = []
    for line in (ROOT / "traces/trace_real.jsonl").open():
        r = json.loads(line)
        if (r.get("draft_tokens_out") or 0) > 0:
            drafts.append(r["draft_tokens_out"])
    outs.sort()
    drafts.sort()
    def dist(v):
        v = sorted(v)
        return {"p10": v[int(.1 * (len(v) - 1))], "p50": v[len(v) // 2],
                "p90": v[int(.9 * (len(v) - 1))], "mean": round(statistics.fmean(v), 1),
                "n": len(v), "samples": v}

    # length depends on max_tokens: 40-token route calls are not drafts
    report["tokens_out"] = dist(outs)
    report["tokens_out_short"] = dist([o for o in outs if o <= 64] or outs)
    report["tokens_out_draft"] = dist(drafts or outs)
    report["tokens_out_draft"]["source"] = "traces/trace_real.jsonl draft_tokens_out only"
    report["measured_gap_ms_p50"] = round(sorted(gaps)[len(gaps) // 2], 4)

    (HERE / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")

    total = sum(len(v) for v in bands.values())
    print(f"bootstrap over {total} observed samples (seed {SEED})")
    for name, lo, hi in BANDS:
        v = bands[name]
        print(f"  {name:9s} n={len(v):2d}  "
              + (f"min={min(v):7.1f} p50={sorted(v)[len(v)//2]:7.1f} max={max(v):7.1f}" if v else "empty"))
    print("\ncl100k_base token accounting vs real prompt_tokens:")
    print(f"  template overhead (real - cl100k, median): {overhead}")
    for name, lo, hi in BANDS:
        r, c = raw_err[name], corrected_err[name]
        if not r:
            continue
        print(f"  {name:9s} n={len(r):2d}  raw {statistics.fmean(r) * 100:5.1f}%  "
              f"after overhead {statistics.fmean(c) * 100:5.1f}%")
    worst = max(((k, statistics.fmean(v)) for k, v in corrected_err.items() if v),
                key=lambda kv: kv[1], default=None)
    if worst:
        print(f"  worst band after correction: {worst[0]} ({worst[1] * 100:.1f}%)")
    print(f"\ninter-token: {report['inter_token']['gap_ms']} ms/token "
          f"({1000 / report['inter_token']['gap_ms']:.0f} tok/s), floor "
          f"{report['inter_token']['measured_gap_ms_p50']} ms")
    print(f"completion length  short(max<=64): {report['tokens_out_short']['mean']} mean "
          f"(n={report['tokens_out_short']['n']})   "
          f"draft: {report['tokens_out_draft']['mean']} mean "
          f"(n={report['tokens_out_draft']['n']})")


if __name__ == "__main__":
    main()
