"""Shared inference core for the Group 20 stunting screening tool (model v2).

Both ``app.py`` and ``self_test.py`` import this module, so the code path the
panel sees demonstrated is provably the same code path the reproduction check
in Section 3.13.4 validates.

Every transformation below is traced to its origin in the v2 analysis notebook
(LatestThesisModel_v2_FinalRun.ipynb):

* ENNS code 9999 treated as missing on every categorical input, then median
  (continuous) or mode (categorical) imputation and standardisation of weight
  and age -- Task C (Decision D1), parameters re-derived and exported by cell
  EP-5 v2 into ``manifest.json``;
* one-hot encoding of the twelve nominal variables -- Task D (panel revision
  R3), restricted to the forty-four columns retained by the Stage 1 filter of
  Section 3.3.3. Ten of the twelve variables keep at least one column;
  dwelling type and province keep none and are therefore not inputs;
* tier assignment -- Equations 3.15 to 3.17, Section 3.11.2.

No parameter is re-estimated from the records supplied to the tool, so a child
scored in isolation receives exactly the treatment it would have received
inside the test partition (Section 3.13.2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
ARTEFACTS = ROOT / "artefacts"
MANIFEST_PATH = ARTEFACTS / "manifest.json"
CODEBOOK_PATH = ARTEFACTS / "enns_codebook.csv"
THRESHOLDS_PATH = ARTEFACTS / "section_3_11_thresholds.json"
SURFACE_PATH = ARTEFACTS / "section_3_11_threshold_surface.csv"
REPRODUCTION_PATH = ARTEFACTS / "reproduction_check.json"

# --------------------------------------------------------------------------
# Hard-coded expectations (Section 3.13.2)
#
# The application holds its own copy of the preprocessing parameters and
# compares them against the manifest before scoring anything. Where the two
# disagree the application refuses to score, because a tool preprocessing
# differently from the notebook returns values that look correct and are not
# the values reported in Chapter 4.
#
# Source of every number below: manifest.json exported 2026-10-09 by cell
# EP-5 v2 into model_outputs/tool_bundle_v2/, cross-checked against
# claude/outputs_v2/section_3_11_thresholds_v2.json, P1_missing_values_v2.csv,
# P2_normalization_v2.csv, P4_stage1_all_candidates_v2.csv and
# J1_tier_composition_v2.csv.
# --------------------------------------------------------------------------

EXPECTED_MANIFEST_VERSION = 2
EXPECTED_MODEL_NAME = "Exp4_ADASYN_XGB"
EXPECTED_MODEL_FILE = "model_Exp4_ADASYN_XGB_v2.json"
EXPECTED_SEVERE_INDEX = 2
EXPECTED_TAU_H = 0.10098794847726822
EXPECTED_TAU_M = 0.057

#: ENNS "not stated / missing" code. Treated as missing on every categorical
#: input before imputation (v2 notebook, Task C, Decision D1).
MISSING_CODES: List[float] = [9999.0]

#: Training-partition median (weight, age) and mode (every categorical input).
#: `sex` and `ethnicity` are listed because the manifest records them, but the
#: tool never imputes them: neither had a missing value in either partition,
#: so a missing one is a data-entry error and stops the run (see preprocess).
EXPECTED_IMPUTE: Dict[str, float] = {
    "weight": 11.69999980926514,
    "age": 2.822724104,
    "sex": 1.0,
    "ethnicity": 0.0,
    "regcode": 8.0,
    "bedroom": 1.0,
    "fwealthq": 1.0,
    "roof": 6.0,
    "wall": 6.0,
    "floor": 8.0,
    "tenurhws": 1.0,
    "tenurlot": 1.0,
    "electrct": 2.0,
    "wdrinkng": 91.0,
    "drinksafe": 0.0,
    "makesafe": 0.0,
    "toilet": 1.0,
    "fuelmain": 6.0,
    "collect": 0.0,
    "burn": 0.0,
    "dump": 0.0,
    "composting": 0.0,
    "segregate": 1.0,
}
NO_IMPUTE_COLS: Tuple[str, ...] = ("sex", "ethnicity")

#: StandardScaler parameters (mean, population SD), fitted on the training
#: partition after median imputation. Unchanged from v1.
EXPECTED_SCALER: Dict[str, Tuple[float, float]] = {
    "weight": (11.633574383551077, 3.4945029302608726),
    "age": (2.7028876781178597, 1.436853151054606),
}

#: The forty-four columns in MODEL order (the Stage 1 importance order of
#: P4_stage1_all_candidates_v2.csv, NOT alphabetical). Order matters: the
#: booster maps columns by position.
EXPECTED_FEATURE_COLS: List[str] = [
    "weight", "age", "bedroom", "wall", "fwealthq", "roof", "sex", "burn",
    "segregate", "composting", "dump", "tenurlot_1.0", "tenurlot_3.0",
    "collect", "toilet_1.0", "floor_8.0", "wdrinkng_41.0", "electrct_2.0",
    "fuelmain_6.0", "tenurhws_1.0", "wdrinkng_14.0", "wdrinkng_91.0",
    "ethnicity", "toilet_2.0", "wdrinkng_11.0", "tenurhws_3.0",
    "fuelmain_2.0", "floor_2.0", "electrct_5.0", "drinksafe_0.0",
    "makesafe_0.0", "floor_3.0", "wdrinkng_12.0", "fuelmain_5.0", "floor_1.0",
    "drinksafe_1.0", "regcode_8", "makesafe_1.0", "regcode_10",
    "tenurlot_2.0", "toilet_8.0", "regcode_5", "regcode_6", "regcode_15",
]

#: Comparisons of floating-point parameters. Every value above is a verbatim
#: copy of the manifest, so the guard can be strict.
GUARD_TOLERANCE = 1e-12

#: XGBoost version recorded inside model_Exp4_ADASYN_XGB_v2.json
#: ("version": [3, 4, 1]) and printed by the v2 notebook. A booster saved by
#: one version and loaded by another can load without raising and score
#: differently, so requirements.txt pins this exact version.
EXPECTED_XGBOOST_VERSION = "3.4.1"

# --------------------------------------------------------------------------
# Encoding scheme (v2 notebook, Task D)
#
# The suffix format is NOT uniform and must not be generated by a single rule.
# `regcode` was an integer column, so pandas.get_dummies produced `regcode_8`;
# the other nine variables were float columns, so it produced `floor_8.0` and
# so on. Each pair is written out explicitly and checked against the manifest
# at start-up. A category code with no column here (for example floor = 4,
# or a region outside the five listed) is the all-zero reference level, which
# is exactly how the notebook treated it after the Stage 1 filter.
# --------------------------------------------------------------------------

ONE_HOT_SCHEME: Dict[str, List[Tuple[float, str]]] = {
    "regcode": [(5.0, "regcode_5"), (6.0, "regcode_6"), (8.0, "regcode_8"),
                (10.0, "regcode_10"), (15.0, "regcode_15")],
    "floor": [(1.0, "floor_1.0"), (2.0, "floor_2.0"), (3.0, "floor_3.0"),
              (8.0, "floor_8.0")],
    "electrct": [(2.0, "electrct_2.0"), (5.0, "electrct_5.0")],
    "tenurhws": [(1.0, "tenurhws_1.0"), (3.0, "tenurhws_3.0")],
    "tenurlot": [(1.0, "tenurlot_1.0"), (2.0, "tenurlot_2.0"), (3.0, "tenurlot_3.0")],
    "drinksafe": [(0.0, "drinksafe_0.0"), (1.0, "drinksafe_1.0")],
    "makesafe": [(0.0, "makesafe_0.0"), (1.0, "makesafe_1.0")],
    "wdrinkng": [(11.0, "wdrinkng_11.0"), (12.0, "wdrinkng_12.0"),
                 (14.0, "wdrinkng_14.0"), (41.0, "wdrinkng_41.0"),
                 (91.0, "wdrinkng_91.0")],
    "toilet": [(1.0, "toilet_1.0"), (2.0, "toilet_2.0"), (8.0, "toilet_8.0")],
    "fuelmain": [(2.0, "fuelmain_2.0"), (5.0, "fuelmain_5.0"), (6.0, "fuelmain_6.0")],
}

#: Flat set of every generated one-hot column name.
ONE_HOT_NAMES = {name for pairs in ONE_HOT_SCHEME.values() for _, name in pairs}

#: Columns passed to the model untouched, as raw integer codes.
PASS_THROUGH_COLS: List[str] = [
    "bedroom", "wall", "fwealthq", "roof", "sex", "burn", "segregate",
    "composting", "dump", "collect", "ethnicity",
]

#: Columns the booster declares as integer-typed. Source: the exported model
#: file's ``learner.feature_types``, which marks `sex` and `ethnicity` as "int",
#: the thirty-one one-hot columns as "i" (indicator) and everything else as
#: "float". XGBoost validates the schema of a named DataFrame at predict time,
#: so the dtypes are aligned here rather than left to whatever pandas inferred.
INT_COLS: List[str] = ["sex", "ethnicity"]

#: Columns standardised at inference. Height is absent by design: Section 1.6
#: Limitation 1 excludes it from the feature set to prevent target leakage, so
#: the tool neither requests nor accepts it.
SCALED_COLS: List[str] = ["weight", "age"]

#: Every raw column a caller must supply (the 23 form fields; manifest
#: ``raw_inputs``). Dwelling type and province are not inputs.
REQUIRED_RAW_COLS: List[str] = [
    "weight", "age", "sex", "ethnicity", "regcode", "bedroom", "fwealthq",
    "roof", "wall", "floor", "tenurhws", "tenurlot", "electrct", "wdrinkng",
    "drinksafe", "makesafe", "toilet", "fuelmain", "collect", "burn", "dump",
    "composting", "segregate",
]
CATEGORICAL_COLS: List[str] = [c for c in REQUIRED_RAW_COLS if c not in SCALED_COLS]

# --------------------------------------------------------------------------
# Tier definitions (Table 3.7, quoted verbatim from
# claude/outputs_v2/table_3_7_action_protocol_v2.csv) and the tier performance
# observed on the test partition (Table 4.10, Section 4.7.2; v2 values from
# J1_tier_composition_v2.csv). The manifest carries the same precision values
# and the guard compares them.
# --------------------------------------------------------------------------

HIGH, MONITOR, ONTRACK = "High Priority", "Needs Monitoring", "On Track"
TIER_ORDER: List[str] = [HIGH, MONITOR, ONTRACK]

ACTION_PROTOCOL: Dict[str, str] = {
    HIGH: (
        "Immediate nutritional assessment and intervention by a licensed "
        "health professional at the Rural Health Unit."
    ),
    MONITOR: (
        "Scheduled regular growth monitoring (monthly) by Barangay Health "
        "Workers; supplementary feeding program inclusion if applicable."
    ),
    ONTRACK: (
        "Routine surveillance at standard intervals (e.g., quarterly); general "
        "health and nutrition promotion for parents."
    ),
}

#: Section 4.7.2, Table 4.10. Held-out test partition, n = 8,632.
TIER_PRECISION_PCT: Dict[str, float] = {HIGH: 26.72, MONITOR: 3.68, ONTRACK: 0.89}
TIER_REFERENCE_COUNTS: Dict[str, int] = {HIGH: 1830, MONITOR: 760, ONTRACK: 6042}
BASE_RATE_PCT = 6.61
TEST_PARTITION_N = 8632

DISCLAIMER = (
    "Screening result only. This is not a diagnosis and does not replace "
    "in-person assessment by a health professional. On the held-out test "
    f"partition, {TIER_PRECISION_PCT[HIGH]:.2f} per cent of the children placed in "
    "the High Priority tier were in fact severely stunted, against a base rate of "
    f"{BASE_RATE_PCT:.2f} per cent. A High Priority result is therefore a trigger "
    "for assessment, not a finding of severe stunting."
)


class ToolError(RuntimeError):
    """Raised when the tool cannot score safely. Never caught to fall back."""


# --------------------------------------------------------------------------
# Loading and guarding
# --------------------------------------------------------------------------


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """Read the exported manifest, raising an actionable error if absent."""
    if not path.exists():
        raise ToolError(
            f"{path} not found. Copy model_Exp4_ADASYN_XGB_v2.json and manifest.json "
            "from Drive (dataset/model_outputs/tool_bundle_v2/) into the artefacts/ folder."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= GUARD_TOLERANCE
    except (TypeError, ValueError):
        return False


def _expected_dtype(col: str) -> str:
    if col in ONE_HOT_NAMES:
        return "bool"
    if col in INT_COLS:
        return "int64"
    return "float64"


def check_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Compare the manifest against the hard-coded expectations.

    Returns a list of human-readable discrepancies; an empty list means the
    manifest matches and scoring may proceed (Section 3.13.2).
    """
    problems: List[str] = []

    if manifest.get("manifest_version") != EXPECTED_MANIFEST_VERSION:
        problems.append(
            f"manifest_version: manifest has {manifest.get('manifest_version')!r}, "
            f"expected {EXPECTED_MANIFEST_VERSION} (the v2 one-hot encoding). "
            "A v1 manifest describes the 31-feature model and must not be used."
        )
    if manifest.get("model_name") != EXPECTED_MODEL_NAME:
        problems.append(
            f"model_name: manifest has {manifest.get('model_name')!r}, "
            f"expected {EXPECTED_MODEL_NAME!r}"
        )
    if manifest.get("model_file") != EXPECTED_MODEL_FILE:
        problems.append(
            f"model_file: manifest has {manifest.get('model_file')!r}, "
            f"expected {EXPECTED_MODEL_FILE!r}"
        )
    if manifest.get("severe_class_index") != EXPECTED_SEVERE_INDEX:
        problems.append(
            f"severe_class_index: manifest has {manifest.get('severe_class_index')!r}, "
            f"expected {EXPECTED_SEVERE_INDEX}"
        )

    got_cols = list(manifest.get("feature_cols", []))
    if got_cols != EXPECTED_FEATURE_COLS:
        problems.append(
            f"feature_cols: the manifest lists {len(got_cols)} columns; this "
            f"application was written against {len(EXPECTED_FEATURE_COLS)} "
            "(order and membership must match exactly)"
        )

    for key, expected in (("tau_H", EXPECTED_TAU_H), ("tau_M", EXPECTED_TAU_M)):
        got = manifest.get(key)
        if not _close(got, expected):
            problems.append(f"{key}: manifest has {got!r}, expected {expected!r}")

    got_missing = [float(x) for x in manifest.get("missing_codes", [])]
    if sorted(got_missing) != sorted(MISSING_CODES):
        problems.append(
            f"missing_codes: manifest has {manifest.get('missing_codes')!r}, "
            f"expected {MISSING_CODES!r}"
        )

    impute = manifest.get("impute", {})
    for col, expected_val in EXPECTED_IMPUTE.items():
        if not _close(impute.get(col), expected_val):
            problems.append(
                f"impute[{col}]: manifest has {impute.get(col)!r}, expected {expected_val!r}"
            )

    for col, (mu, sd) in EXPECTED_SCALER.items():
        got = manifest.get("scaler", {}).get(col)
        if (
            not isinstance(got, Sequence)
            or len(got) != 2
            or not _close(got[0], mu)
            or not _close(got[1], sd)
        ):
            problems.append(f"scaler[{col}]: manifest has {got!r}, expected [{mu!r}, {sd!r}]")

    # One-hot specification: same variables, same codes, same column names.
    got_onehot = manifest.get("onehot", {})
    expected_spec = {v: sorted((c, n) for c, n in pairs) for v, pairs in ONE_HOT_SCHEME.items()}
    try:
        got_spec = {
            v: sorted((float(d["code"]), str(d["column"])) for d in lst)
            for v, lst in got_onehot.items()
        }
    except (KeyError, TypeError, ValueError):
        got_spec = None
    if got_spec != expected_spec:
        problems.append(
            "onehot: the manifest's one-hot specification (variables, category "
            "codes or column names) differs from the one this application encodes"
        )

    got_dtypes = manifest.get("feature_dtypes", {})
    bad_dtypes = [
        c for c in EXPECTED_FEATURE_COLS if got_dtypes.get(c) != _expected_dtype(c)
    ]
    if bad_dtypes:
        problems.append(f"feature_dtypes: unexpected training dtype for {bad_dtypes}")

    if sorted(manifest.get("raw_inputs", [])) != sorted(REQUIRED_RAW_COLS):
        problems.append("raw_inputs: the manifest's list of raw inputs differs from the form's")

    got_prec = manifest.get("tier_precision_pct", {})
    for tier, expected_val in TIER_PRECISION_PCT.items():
        got = got_prec.get(tier)
        if got is None or abs(float(got) - expected_val) > 1e-9:
            problems.append(
                f"tier_precision_pct[{tier}]: manifest has {got!r}, expected {expected_val}"
            )
    if not _close(manifest.get("base_rate_pct"), BASE_RATE_PCT):
        problems.append(
            f"base_rate_pct: manifest has {manifest.get('base_rate_pct')!r}, "
            f"expected {BASE_RATE_PCT}"
        )

    # The encoding scheme must account for every feature column and no others.
    accounted = set(SCALED_COLS) | set(PASS_THROUGH_COLS) | ONE_HOT_NAMES
    unexplained = [c for c in EXPECTED_FEATURE_COLS if c not in accounted]
    unused = sorted(accounted - set(EXPECTED_FEATURE_COLS))
    if unexplained:
        problems.append(f"feature columns this encoder cannot produce: {unexplained}")
    if unused:
        problems.append(f"encoder produces columns the model does not use: {unused}")

    return problems


def check_model_schema(manifest: Dict[str, Any]) -> List[str]:
    """Verify the exported booster against the manifest, without loading xgboost.

    The model file is plain JSON, so its recorded feature names, feature types,
    class count and library version can be read directly. A mismatch here means
    the bundle is internally inconsistent and must be re-exported (EP-5 v2).
    """
    problems: List[str] = []
    model_path = ARTEFACTS / str(manifest.get("model_file", ""))
    if not model_path.is_file():
        return [f"{model_path} not found; copy the v2 bundle from Drive into artefacts/"]

    with open(model_path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    learner = blob.get("learner", {})

    version = ".".join(str(v) for v in blob.get("version", []))
    if version != EXPECTED_XGBOOST_VERSION:
        problems.append(
            f"model file was saved by xgboost {version}, expected "
            f"{EXPECTED_XGBOOST_VERSION}; update the pin in requirements.txt"
        )
    if list(learner.get("feature_names", [])) != list(manifest.get("feature_cols", [])):
        problems.append("booster feature_names differ from manifest feature_cols")
    params = learner.get("learner_model_param", {})
    if str(params.get("num_class")) != "3":
        problems.append(f"booster num_class is {params.get('num_class')!r}, expected 3")
    if str(params.get("num_feature")) != str(len(EXPECTED_FEATURE_COLS)):
        problems.append(
            f"booster num_feature is {params.get('num_feature')!r}, "
            f"expected {len(EXPECTED_FEATURE_COLS)}"
        )

    declared = dict(zip(learner.get("feature_names", []), learner.get("feature_types", [])))
    for col, kind in declared.items():
        if kind == "i" and col not in ONE_HOT_NAMES:
            problems.append(f"booster marks {col!r} as an indicator; the encoder does not")
        if kind != "i" and col in ONE_HOT_NAMES:
            problems.append(f"the encoder makes {col!r} an indicator; the booster does not")
        if kind == "int" and col not in INT_COLS:
            problems.append(f"booster marks {col!r} as int; INT_COLS does not list it")
    return problems


def load_model(manifest: Dict[str, Any]):
    """Load the fitted booster from XGBoost's native JSON format.

    xgboost is imported here rather than at module scope so that the
    preprocessing path can be exercised in environments where it is absent.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ToolError(
            "xgboost is not installed in this environment. Activate the virtual "
            "environment and run: pip install -r requirements.txt"
        ) from exc

    model_path = ARTEFACTS / manifest["model_file"]
    if not model_path.exists():
        raise ToolError(
            f"{model_path} not found. Copy the model file named in manifest.json "
            "from Drive into the artefacts/ folder."
        )
    model = XGBClassifier()
    model.load_model(str(model_path))
    return model


def load_codebook(path: Path = CODEBOOK_PATH) -> pd.DataFrame:
    """Load the ENNS codebook used to label the form's dropdown options."""
    if not path.exists():
        raise ToolError(
            f"{path} not found. Copy claude_ENNS_codebook_FILLED.csv into "
            "artefacts/ and rename it enns_codebook.csv."
        )
    cb = pd.read_csv(path)
    cb["code"] = pd.to_numeric(cb["code"], errors="coerce")
    return cb


def options_for(codebook: pd.DataFrame, variable: str) -> List[Tuple[float, str]]:
    """Return ``(code, label)`` pairs for one variable, ordered by code."""
    rows = codebook[codebook["variable"] == variable].sort_values("code")
    label_col = "label_TO_FILL" if "label_TO_FILL" in rows.columns else rows.columns[-1]
    return [(float(r.code), str(getattr(r, label_col))) for r in rows.itertuples()]


def validate_codes(raw: pd.DataFrame, codebook: pd.DataFrame) -> List[str]:
    """Report category codes that the ENNS codebook does not define.

    Without this check an undefined code (say floor = 99) would be scored
    silently as the all-zero reference level. Blank cells and the missing code
    9999 are allowed: they are imputed. `bedroom` is a count, not a coded
    variable, and is checked for being a whole number from 0 upward instead.
    """
    problems: List[str] = []
    for col in CATEGORICAL_COLS:
        if col not in raw.columns:
            continue
        values = pd.to_numeric(raw[col], errors="coerce")
        non_numeric = raw[col].notna() & values.isna() & (raw[col].astype(str).str.strip() != "")
        if non_numeric.any():
            rows = (np.flatnonzero(non_numeric.to_numpy()) + 1)[:10].tolist()
            problems.append(f"{col}: non-numeric value in row(s) {rows}")
        present = values.notna() & ~values.isin(MISSING_CODES)
        if col == "bedroom":
            bad = present & ((values < 0) | (values != np.floor(values)))
            what = "is not a whole number of bedrooms"
        else:
            allowed = set(codebook.loc[codebook["variable"] == col, "code"].dropna())
            if not allowed:
                continue
            bad = present & ~values.isin(allowed)
            what = "is not defined in the ENNS codebook"
        if bad.any():
            rows = (np.flatnonzero(bad.to_numpy()) + 1)[:10].tolist()
            codes = sorted(set(values[bad].tolist()))[:10]
            problems.append(f"{col}: value(s) {codes} {what} (data row(s) {rows})")
    return problems


# --------------------------------------------------------------------------
# Reference statistics shown in the interface
# --------------------------------------------------------------------------


def load_thresholds_json(path: Path = THRESHOLDS_PATH) -> Dict[str, Any]:
    """Load the Section 3.11 operating-point statistics, if present.

    These are the exact figures reported in Sections 4.7.1 and 5.2.3. They are
    displayed in preference to any value interpolated from the trade-off grid,
    which is sampled at 0.001 and therefore does not land on tau_H exactly.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_surface(path: Path = SURFACE_PATH) -> pd.DataFrame:
    """Load the sensitivity/specificity trade-off surface, if present."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_reproduction(path: Path = REPRODUCTION_PATH) -> Dict[str, Any]:
    """Load the recorded reproduction-check results (aggregate values only)."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def surface_at(surface: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Nearest grid point on the trade-off surface.

    The grid is sampled every 0.001, so the returned row is an approximation
    for any threshold that does not fall on a grid point. Callers must label it
    as such rather than reporting it as the value at the threshold itself.
    """
    if surface.empty:
        return {}
    row = surface.iloc[(surface["threshold"] - float(threshold)).abs().idxmin()]
    return {k: float(row[k]) for k in surface.columns}


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def preprocess(raw: pd.DataFrame, manifest: Dict[str, Any]) -> pd.DataFrame:
    """Turn raw ENNS-coded records into the 44-column model input matrix.

    The order of operations reproduces the v2 notebook: treat code 9999 as
    missing on the categorical inputs, impute with the training partition's
    median (continuous) or mode (categorical), standardise weight and age with
    the training partition's mean and standard deviation, one-hot encode the
    ten nominal fields that kept a column, then reindex to ``feature_cols``.

    A missing raw column raises rather than being filled with zeros, because a
    silently zero-filled column scores without error and scores wrongly.
    """
    missing = [c for c in REQUIRED_RAW_COLS if c not in raw.columns]
    if missing:
        raise ToolError(f"input is missing required columns: {missing}")

    df = pd.DataFrame(index=raw.index)
    missing_codes = [float(x) for x in manifest["missing_codes"]]

    # 1. Numeric coercion, 9999 -> missing, imputation (Task C).
    for col in REQUIRED_RAW_COLS:
        series = pd.to_numeric(raw[col], errors="coerce").astype("float64")
        if col in CATEGORICAL_COLS:
            series = series.mask(series.isin(missing_codes))
        if col not in NO_IMPUTE_COLS:
            series = series.fillna(float(manifest["impute"][col]))
        df[col] = series

    # `sex` and `ethnicity` had no missing value in either partition. A missing
    # one here is a data-entry error and must stop the run rather than be guessed.
    for col in NO_IMPUTE_COLS:
        if df[col].isna().any():
            raise ToolError(
                f"{col} is missing for {int(df[col].isna().sum())} record(s); "
                "it must be supplied."
            )

    # 2. Standardisation -- weight and age only (Task C).
    for col in SCALED_COLS:
        mean, std = manifest["scaler"][col]
        df[col] = (df[col] - float(mean)) / float(std)

    # 3. One-hot encoding (Task D). A code with no column leaves every
    #    indicator of that variable at 0 (the reference level).
    for source, pairs in ONE_HOT_SCHEME.items():
        for level, name in pairs:
            df[name] = df[source] == level

    # 4. Column order and dtypes, exactly as the booster was trained.
    matrix = df[list(manifest["feature_cols"])].copy()
    for col in matrix.columns:
        matrix[col] = matrix[col].astype(_expected_dtype(col))

    if matrix.isna().any().any():
        bad = matrix.columns[matrix.isna().any()].tolist()
        raise ToolError(f"unresolved missing values remain in: {bad}")
    return matrix


# --------------------------------------------------------------------------
# Scoring and tiering
# --------------------------------------------------------------------------


def predict_severe_proba(model, matrix: pd.DataFrame, manifest: Dict[str, Any]) -> np.ndarray:
    """Return P(Severely Stunted) -- column 2 of the class-probability array."""
    return model.predict_proba(matrix)[:, int(manifest["severe_class_index"])]


def assign_tier(p: float, tau_h: float, tau_m: float) -> str:
    """Equations 3.15 to 3.17 (Section 3.11.2).

    ``p`` is compared at full precision. Rounding it first would move children
    who sit exactly on a threshold (row 5980 of the test partition has
    P(Severely Stunted) equal to tau_H).
    """
    p = float(p)
    if p >= tau_h:
        return HIGH
    if p >= tau_m:
        return MONITOR
    return ONTRACK


def assign_tiers(probs: Sequence[float], tau_h: float, tau_m: float) -> pd.Series:
    """Vectorised form of :func:`assign_tier`."""
    arr = np.asarray(probs, dtype=float)
    return pd.Series(
        np.where(arr >= tau_h, HIGH, np.where(arr >= tau_m, MONITOR, ONTRACK)),
        index=getattr(probs, "index", None),
    )


def blank_template() -> pd.DataFrame:
    """One empty row carrying the exact headers batch mode expects."""
    return pd.DataFrame([{col: "" for col in ["child_id"] + REQUIRED_RAW_COLS}])
