"""
Compute summary statistics for a pilot evaluation run.

Usage:
    python scripts/summarise_results.py <eval_dir>

Example:
    python scripts/summarise_results.py outputs/evaluations/pilot_run_20260518T135815_bdad28
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def proportion_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = proportion_ci(k, n)
    return f"{100 * k / n:.1f}% (95% CI: {100*lo:.1f}–{100*hi:.1f}%)"


def mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    m = sum(values) / len(values)
    if len(values) > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
        return f"{m:.3f} (SD = {sd:.3f})"
    return f"{m:.3f}"


def chi2_2x2(a: int, b: int, c: int, d: int) -> tuple[float, str]:
    """
    2×2 chi-squared test (no Yates correction).
    Rows = strategies (martens, CoT); cols = hallucination (yes, no).
    Returns (chi2 statistic, significance string).
    """
    n = a + b + c + d
    if n == 0:
        return (0.0, "n/a")
    e_a = (a + b) * (a + c) / n
    e_b = (a + b) * (b + d) / n
    e_c = (c + d) * (a + c) / n
    e_d = (c + d) * (b + d) / n
    cells = [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]
    chi2 = sum((o - e) ** 2 / e for o, e in cells if e > 0)
    # approximate critical values: χ²(1) = 3.841 (p<.05), 6.635 (p<.01), 10.828 (p<.001)
    if chi2 >= 10.828:
        sig = "p < .001"
    elif chi2 >= 6.635:
        sig = "p < .01"
    elif chi2 >= 3.841:
        sig = "p < .05"
    else:
        sig = "n.s."
    return (chi2, sig)


# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------

def load_evaluations(eval_dir: Path) -> list[dict]:
    path = eval_dir / "evaluations.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_robustness(eval_dir: Path) -> list[dict]:
    path = eval_dir / "robustness.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_metadata(eval_dir: Path) -> dict:
    import yaml  # type: ignore
    path = eval_dir / "eval_metadata.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

HAL_TYPES = ["sign_inversion", "rank_swap", "feature_fabrication", "omission"]
STRATEGIES = ["martens", "chain_of_thought"]


def analyse_evaluations(rows: list[dict]) -> None:
    print("=" * 70)
    print("SECTION 1 — EVALUATION OVERVIEW")
    print("=" * 70)

    total = len(rows)
    parse_errors = [r for r in rows if r.get("parse_error", "").strip()]
    valid = [r for r in rows if not r.get("parse_error", "").strip()]

    print(f"\nTotal narratives:          {total}")
    print(f"Extraction parse errors:   {len(parse_errors)} ({100*len(parse_errors)/total:.1f}%)")
    print(f"Successfully evaluated:    {len(valid)}")

    print("\n" + "-" * 70)
    print("SECTION 2 — HALLUCINATION RATES BY TYPE (valid extractions only)")
    print("-" * 70)

    header = f"{'Type':<22} {'Overall':>20} {'Martens':>20} {'Chain-of-thought':>20}"
    print("\n" + header)
    print("-" * len(header))

    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        by_strategy[r["prompt_strategy"]].append(r)

    mart = by_strategy.get("martens", [])
    cot  = by_strategy.get("chain_of_thought", [])

    for col in HAL_TYPES + ["any_hallucination"]:
        label = col.replace("_", " ").capitalize()
        k_all = sum(int(r[col]) for r in valid)
        k_m   = sum(int(r[col]) for r in mart)
        k_c   = sum(int(r[col]) for r in cot)
        print(
            f"  {label:<20} "
            f"{pct(k_all, len(valid)):>20} "
            f"{pct(k_m, len(mart)):>20} "
            f"{pct(k_c, len(cot)):>20}"
        )

    print("\n" + "-" * 70)
    print("SECTION 3 — STRATEGY COMPARISON (chi-squared tests, df = 1)")
    print("-" * 70)
    print(f"\n  {'Hallucination type':<25} {'chi2':>8} {'sig':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*10}")

    for col in HAL_TYPES + ["any_hallucination"]:
        label = col.replace("_", " ").capitalize()
        a = sum(int(r[col]) for r in mart)          # martens & hallucinated
        b = len(mart) - a                            # martens & not
        c = sum(int(r[col]) for r in cot)           # CoT & hallucinated
        d = len(cot) - c                             # CoT & not
        chi2, sig = chi2_2x2(a, b, c, d)
        print(f"  {label:<25} {chi2:>8.2f} {sig:>10}")

    print("\n" + "-" * 70)
    print("SECTION 4 — FEATURE-LEVEL HALLUCINATION PATTERNS")
    print("-" * 70)

    feature_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in valid:
        extraction_raw = r.get("extraction_json", "")
        if not extraction_raw or extraction_raw.strip() in ("{}", ""):
            continue
        try:
            ext = json.loads(extraction_raw)
        except json.JSONDecodeError:
            continue

        features = ext.get("features", {})
        for feat, info in features.items():
            feature_counts[feat]["mentions"] += 1

    # Sign inversion per feature: compare extraction sign to SHAP sign
    # (requires pairing with the robustness data — done in Section 5)
    # Here we report which features appear most frequently across narratives.
    if feature_counts:
        print(f"\n  {'Feature':<35} {'Narrative mentions':>20}")
        print(f"  {'-'*35} {'-'*20}")
        for feat, counts in sorted(feature_counts.items(), key=lambda x: -x[1]["mentions"]):
            print(f"  {feat:<35} {counts['mentions']:>20}")

    print("\n" + "-" * 70)
    print("SECTION 5 — CO-OCCURRENCE OF HALLUCINATION TYPES")
    print("-" * 70)

    combos: dict[frozenset, int] = defaultdict(int)
    for r in valid:
        active = frozenset(col for col in HAL_TYPES if int(r[col]) == 1)
        if active:
            combos[active] += 1

    combos_sorted = sorted(combos.items(), key=lambda x: -x[1])
    print(f"\n  {'Combination':<45} {'Count':>8} {'%':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8}")
    for combo, count in combos_sorted[:15]:
        label = " + ".join(sorted(c.replace("_", " ") for c in combo))
        print(f"  {label:<45} {count:>8} {100*count/len(valid):>7.1f}%")


def analyse_robustness(records: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SECTION 6 — EXTRACTION ROBUSTNESS")
    print("=" * 70)

    total = len(records)
    unreliable = [r for r in records if r["robustness"].get("extraction_unreliable")]
    low_rel    = [r for r in records if r["robustness"].get("flagged_low_reliability")]
    scoreable  = [
        r for r in records
        if not r["robustness"].get("extraction_unreliable")
        and r["robustness"].get("narrative_reliability_score") is not None
    ]

    scores = [r["robustness"]["narrative_reliability_score"] for r in scoreable]
    top_k  = [
        r["robustness"]["top_k_set_agreement"]
        for r in scoreable
        if r["robustness"].get("top_k_set_agreement") is not None
    ]

    print(f"\nNarratives with robustness check: {total}")
    print(f"Extraction-unreliable (<3 valid runs): {len(unreliable)}")
    print(f"Flagged low reliability (score < 0.8): {len(low_rel)}")
    print(f"\nNarrative reliability score: {mean_sd(scores)}")
    print(f"Top-k set agreement:         {mean_sd(top_k)}")

    # by strategy
    print("\n  By prompt strategy:")
    for strat in STRATEGIES:
        sub = [r for r in scoreable if r["prompt_strategy"] == strat]
        s_scores = [r["robustness"]["narrative_reliability_score"] for r in sub]
        s_topk   = [
            r["robustness"]["top_k_set_agreement"]
            for r in sub
            if r["robustness"].get("top_k_set_agreement") is not None
        ]
        print(f"  {strat:<20} reliability = {mean_sd(s_scores)}")
        print(f"  {'':<20} top-k agree  = {mean_sd(s_topk)}")

    # sign agreement per feature across all narratives
    sign_agg: dict[str, list[float]] = defaultdict(list)
    val_agg:  dict[str, list[float]] = defaultdict(list)
    for r in scoreable:
        pf = r["robustness"].get("per_feature", {})
        for feat, metrics in pf.items():
            sa = metrics.get("sign_agreement")
            va = metrics.get("value_agreement")
            if sa is not None:
                sign_agg[feat].append(sa)
            if va is not None:
                val_agg[feat].append(va)

    print(f"\n  {'Feature':<35} {'Mean sign agree':>17} {'N':>5}  {'Mean value agree':>17} {'N':>5}")
    print(f"  {'-'*35} {'-'*17} {'-'*5}  {'-'*17} {'-'*5}")
    all_feats = sorted(sign_agg.keys())
    for feat in all_feats:
        sa_vals = sign_agg[feat]
        va_vals = val_agg.get(feat, [])
        sa_mean = sum(sa_vals) / len(sa_vals) if sa_vals else float("nan")
        va_mean = sum(va_vals) / len(va_vals) if va_vals else float("nan")
        sa_str  = f"{sa_mean:.3f}" if sa_vals else "n/a"
        va_str  = f"{va_mean:.3f}" if va_vals else "n/a"
        print(f"  {feat:<35} {sa_str:>17} {len(sa_vals):>5}  {va_str:>17} {len(va_vals):>5}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarise_results.py <eval_dir>")
        sys.exit(1)

    eval_dir = Path(sys.argv[1])
    if not eval_dir.exists():
        print(f"Directory not found: {eval_dir}")
        sys.exit(1)

    try:
        meta = load_metadata(eval_dir)
        print(f"\nRun ID:        {meta.get('run_id', 'unknown')}")
        print(f"Dataset:       {meta['config']['datasets'][0]['name']}")
        print(f"Model:         {next(m['id'] for m in meta['config']['models'] if m.get('generation'))}")
        print(f"Instances:     {meta['config']['datasets'][0]['n_instances']}")
        print(f"n_records:     {meta['n_records']}")
        print(f"n_failed:      {meta['n_failed']}")
        print(f"n_any_halluc:  {meta['n_any_hallucination']}")
    except Exception as e:
        print(f"(Metadata load error: {e})")

    rows = load_evaluations(eval_dir)
    analyse_evaluations(rows)

    rob = load_robustness(eval_dir)
    analyse_robustness(rob)


if __name__ == "__main__":
    main()
