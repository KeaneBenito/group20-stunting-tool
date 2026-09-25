"""Reproduction check for the stunting screening tool (Section 3.13.4).

Run from the project root with the virtual environment active::

    python self_test.py

Stage A rebuilds the 31-feature input matrix from the raw held-out records and
compares it against ``test_selected.csv``, the matrix the notebook itself
supplied to the classifier.

Stage B scores those records through the same code path the application uses
and compares the resulting probabilities and tier assignments against
``section_3_11_tier_assignments.csv``.

Exits 0 on pass and 1 on failure. Save the console output as evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stunting_core as core

TEST_DATA = Path(__file__).resolve().parent / "test_data"
RAW_TEST = TEST_DATA / "test.csv"
REF_MATRIX = TEST_DATA / "test_selected.csv"
REF_TIERS = TEST_DATA / "section_3_11_tier_assignments.csv"

PROB_TOL = 1e-6  # the reference CSV stores float32-derived values
RULE = "=" * 78


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def require(path: Path) -> None:
    if not path.exists():
        print(f"\nMISSING FILE: {path}")
        print("Copy it from Drive into test_data/ and run again.")
        sys.exit(1)


def main() -> int:
    ok = True

    print(RULE)
    print("PREFLIGHT -- artefacts and manifest guard")
    print(RULE)
    for path in (RAW_TEST, REF_MATRIX, REF_TIERS):
        require(path)

    manifest = core.load_manifest()
    problems = core.check_manifest(manifest)
    if problems:
        for p in problems:
            fail(p)
        print("\nManifest guard failed. Not scoring. Re-export the bundle (EP-5).")
        return 1
    schema_problems = core.check_model_schema(manifest)
    if schema_problems:
        for p_ in schema_problems:
            fail(p_)
        print("\nModel file does not match the manifest. Not scoring. Re-export (EP-5).")
        return 1
    print(f"  PASS  booster schema verified -- 31 features, 3 classes, "
          f"xgboost {core.EXPECTED_XGBOOST_VERSION}")
    print(f"  PASS  manifest verified -- {manifest['model_name']}, "
          f"exported {manifest.get('exported_at', 'unknown')}")
    print(f"        tau_H = {manifest['tau_H']!r}")
    print(f"        tau_M = {manifest['tau_M']!r}  ({manifest.get('tau_M_source', '')})")

    model = core.load_model(manifest)

    # ---------------------------------------------------------------- Stage A
    print()
    print(RULE)
    print("STAGE A -- preprocessing parity against test_selected.csv")
    print(RULE)

    raw = pd.read_csv(RAW_TEST, low_memory=False)
    ref_matrix = pd.read_csv(REF_MATRIX, low_memory=False)
    built = core.preprocess(raw, manifest)

    feature_cols = list(manifest["feature_cols"])
    stage_a_ok = True

    if len(built.columns) != len(feature_cols):
        fail(f"column count {len(built.columns)} != {len(feature_cols)}")
        stage_a_ok = False
    if list(built.columns) != feature_cols:
        fail("column order does not match manifest feature_cols")
        stage_a_ok = False

    ref_missing = [c for c in feature_cols if c not in ref_matrix.columns]
    if ref_missing:
        fail(f"reference matrix is missing columns: {ref_missing}")
        stage_a_ok = False

    worst_col, worst_diff = None, 0.0
    if stage_a_ok:
        if len(built) != len(ref_matrix):
            fail(f"row count {len(built)} != {len(ref_matrix)}")
            stage_a_ok = False
        else:
            for col in feature_cols:
                diff = float(
                    np.nanmax(
                        np.abs(
                            built[col].astype(float).to_numpy()
                            - ref_matrix[col].astype(float).to_numpy()
                        )
                    )
                )
                if diff > worst_diff:
                    worst_col, worst_diff = col, diff
                if diff > PROB_TOL:
                    fail(f"column {col}: max abs diff {diff:.3e}")
                    stage_a_ok = False

    if stage_a_ok:
        print(f"  PASS  {len(feature_cols)}/{len(feature_cols)} columns match over "
              f"{len(built):,} records, max diff {worst_diff:.2e}"
              + (f" (worst column: {worst_col})" if worst_diff else ""))
    ok = ok and stage_a_ok

    # ---------------------------------------------------------------- Stage B
    print()
    print(RULE)
    print("STAGE B -- score and tier parity against section_3_11_tier_assignments.csv")
    print(RULE)

    ref = pd.read_csv(REF_TIERS)
    tau_h = float(manifest["tau_H"])
    tau_m = float(manifest["tau_M"])

    probs = core.predict_severe_proba(model, built, manifest)
    ref_probs = ref["p_severely_stunted"].astype(float).to_numpy()

    stage_b_ok = True
    if len(probs) != len(ref_probs):
        fail(f"row count {len(probs)} != reference {len(ref_probs)}")
        return 1

    prob_diff = np.abs(probs - ref_probs)
    max_prob_diff = float(prob_diff.max())
    if max_prob_diff > PROB_TOL:
        fail(f"max |prob diff| {max_prob_diff:.3e} exceeds {PROB_TOL:.0e}")
        stage_b_ok = False

    # Tiers are recomputed from the reference probabilities at the manifest
    # thresholds. The stored priority_tier column is only used as a cross-check,
    # because a copy of this file produced before the tau_M revision carries
    # tiers built at the earlier threshold.
    tool_tiers = core.assign_tiers(probs, tau_h, tau_m)
    ref_tiers_now = core.assign_tiers(ref_probs, tau_h, tau_m)
    disagreements = int((tool_tiers.to_numpy() != ref_tiers_now.to_numpy()).sum())
    if disagreements:
        stage_b_ok = False

    print(f"  max |prob diff|      : {max_prob_diff:.3e}")
    print(f"  tier disagreements   : {disagreements}")

    if disagreements:
        idx = np.flatnonzero(tool_tiers.to_numpy() != ref_tiers_now.to_numpy())
        print("\n  Disagreeing records (row, tool p, ref p, tool tier, ref tier, "
              "distance to nearest threshold):")
        for i in idx[:25]:
            d = min(abs(probs[i] - tau_h), abs(probs[i] - tau_m))
            print(f"    {i:6d}  {probs[i]:.8f}  {ref_probs[i]:.8f}  "
                  f"{tool_tiers.iloc[i]:<16s}  {ref_tiers_now.iloc[i]:<16s}  {d:.2e}")
        if len(idx) > 25:
            print(f"    ... and {len(idx) - 25} more")

    # Tier counts against Table 4.10 (Section 4.7.2).
    counts = tool_tiers.value_counts().to_dict()
    print("\n  Tier counts produced by the tool, against Table 4.10:")
    for tier in core.TIER_ORDER:
        got = int(counts.get(tier, 0))
        want = core.TIER_REFERENCE_COUNTS[tier]
        mark = "ok  " if got == want else "DIFF"
        if got != want:
            stage_b_ok = False
        print(f"    [{mark}] {tier:<16s} {got:>6,}   Table 4.10: {want:>6,}")

    # Informational: has the stored column drifted from the current tau_M?
    if "priority_tier" in ref.columns:
        stored_mismatch = int((ref["priority_tier"].to_numpy() != ref_tiers_now.to_numpy()).sum())
        if stored_mismatch:
            print(f"\n  NOTE  the stored priority_tier column disagrees with the current "
                  f"thresholds in {stored_mismatch:,} rows.")
            print("        This is expected if this CSV predates the tau_M revision to "
                  f"{tau_m}. It is not a tool defect;")
            print("        the probabilities, which are what Stage B tests, are unchanged. "
                  "Re-export the file from")
            print("        the notebook to remove the ambiguity.")

    ok = ok and stage_b_ok

    # ---------------------------------------------------------------- Summary
    print()
    print(RULE)
    print("SUMMARY -- paste into Section 3.13.4")
    print(RULE)
    print(f"STAGE A: {'PASS' if stage_a_ok else 'FAIL'} -- "
          f"{len(feature_cols)}/{len(feature_cols)} columns match, max diff {worst_diff:.2e}")
    print(f"STAGE B: {'PASS' if stage_b_ok else 'FAIL'} -- max |prob diff| {max_prob_diff:.2e}, "
          f"{disagreements} tier disagreements (n = {len(probs):,})")
    print(f"Model: {manifest['model_name']}   tau_H = {tau_h:.10f}   tau_M = {tau_m:.10f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
