"""Stunting risk screening tool -- Group 20, CSS200-3 (Section 3.13).

Hosted for demonstration. Run it from the project root with::

    streamlit run app.py

It makes no network request of any kind: no remote fonts, no images fetched
from a CDN, no telemetry, no API calls. Section 1.6 Limitation 5 and Section
3.13.1 both state that the tool was not deployed and not hosted, and network
code would make those statements false.

All inference logic lives in ``stunting_core``, which ``self_test.py`` also
imports, so the code path demonstrated to the panel is the same code path the
reproduction check in Section 3.13.4 validates.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import stunting_core as core

st.set_page_config(page_title="Stunting Risk Screening Tool", layout="wide")

# ==========================================================================
# 1. Start-up: load artefacts, run both guards (Section 3.13.2)
# ==========================================================================


@st.cache_resource(show_spinner=False)
def _boot():
    manifest = core.load_manifest()
    problems = core.check_manifest(manifest) + core.check_model_schema(manifest)
    model = core.load_model(manifest) if not problems else None
    return manifest, problems, model


@st.cache_data(show_spinner=False)
def _reference_data():
    return core.load_codebook(), core.load_thresholds_json(), core.load_surface()


try:
    MANIFEST, PROBLEMS, MODEL = _boot()
    CODEBOOK, THRESHOLDS, SURFACE = _reference_data()
except core.ToolError as exc:
    st.error(str(exc))
    st.stop()

if PROBLEMS:
    st.error(
        "The exported bundle does not match the preprocessing parameters this "
        "application was built against. Scoring has been stopped, because a "
        "tool that preprocesses differently from the notebook would return "
        "values that appear correct and are not the values reported in "
        "Chapter 4."
    )
    for problem in PROBLEMS:
        st.write(f"- {problem}")
    st.stop()

# --------------------------------------------------------------------------
# Age units
#
# The model was trained on ENNS `age` in YEARS, derived from the number of
# days between date of birth and survey date divided by 365.25 (every value in
# test.csv is a whole number of days on that divisor). The manifest's imputation
# median and scaler are therefore also in years. The form accepts age in months
# or as two dates for the user's convenience and converts to years before
# anything reaches stunting_core, so the validated preprocessing path is
# untouched.
# --------------------------------------------------------------------------

DAYS_PER_YEAR = 365.25
MONTHS_PER_YEAR = 12.0
MAX_AGE_YEARS = 5.0  # training range: 0 to 4.9993 years


def _age_from_dates(dob: dt.date, measured: dt.date) -> Tuple[Optional[float], str]:
    """Return (age in years, error message). Mirrors the ENNS derivation."""
    days = (measured - dob).days
    if days <= 0:
        return None, "Date of birth must be before the date of measurement."
    years = days / DAYS_PER_YEAR
    if years >= MAX_AGE_YEARS:
        return None, (
            f"The child is {years:.2f} years old. The model was trained on "
            "children under five and cannot score this child."
        )
    return years, ""


MANIFEST_TAU_H = float(MANIFEST["tau_H"])
MANIFEST_TAU_M = float(MANIFEST["tau_M"])

# ==========================================================================
# 2. Sidebar: thresholds as adjustable parameters (Section 3.13.3)
# ==========================================================================

with st.sidebar:
    st.header("Thresholds")
    st.caption(
        "Initialised from the exported manifest. Both are parameters rather "
        "than constants, so a revised value may be substituted without "
        "modifying the tool."
    )

    if st.button("Reset to manifest values", use_container_width=True):
        st.session_state["tau_h"] = MANIFEST_TAU_H
        st.session_state["tau_m"] = MANIFEST_TAU_M

    tau_h = st.number_input(
        "tau_H — High Priority", min_value=0.0, max_value=1.0,
        value=st.session_state.get("tau_h", MANIFEST_TAU_H),
        step=0.001, format="%.8f", key="tau_h",
    )
    tau_m = st.number_input(
        "tau_M — Needs Monitoring", min_value=0.0, max_value=1.0,
        value=st.session_state.get("tau_m", MANIFEST_TAU_M),
        step=0.001, format="%.8f", key="tau_m",
    )

    st.caption(
        "tau_H was derived by Youden's J-statistic. tau_M was derived from an "
        "operational caseload criterion. The domain expert endorsed the tiering "
        "principle without selecting a specific value, so the caseload-derived "
        f"value was retained — {MANIFEST.get('tau_M_source', '')}"
    )

    if tau_m > tau_h:
        st.warning("tau_M exceeds tau_H, which empties the Needs Monitoring tier.")
    if abs(tau_h - MANIFEST_TAU_H) > 1e-12 or abs(tau_m - MANIFEST_TAU_M) > 1e-12:
        st.info(
            "Thresholds differ from the manifest. Results below are "
            "exploratory and are not the figures reported in Chapter 4."
        )

    st.divider()
    st.success(f"Bundle verified — {MANIFEST['model_name']}")
    st.caption(
        f"Exported {MANIFEST.get('exported_at', 'unknown')} · "
        f"xgboost {core.EXPECTED_XGBOOST_VERSION} · "
        f"{len(MANIFEST['feature_cols'])} features"
    )

# ==========================================================================
# 3. Form option lists, drawn from the ENNS codebook
# ==========================================================================

DROPDOWN_FIELDS: List[Tuple[str, str]] = [
    ("regcode", "Region"),
    ("sex", "Sex"),
    ("ethnicity", "Indigenous People household"),
    ("fwealthq", "Wealth quintile"),
    ("roof", "Roof material"),
    ("wall", "Outer wall material"),
    ("floor", "Floor material"),
    ("tenurhws", "Tenure of house"),
    ("tenurlot", "Tenure of lot"),
    ("electrct", "Electricity"),
    ("wdrinkng", "Main source of drinking water"),
    ("drinksafe", "Treats drinking water to make it safe"),
    ("makesafe", "Treatment to make water safe"),
    ("toilet", "Toilet facility"),
    ("fuelmain", "Main cooking fuel"),
    ("collect", "Waste collected"),
    ("burn", "Waste burned"),
    ("dump", "Waste dumped"),
    ("composting", "Composting practised"),
    ("segregate", "Waste segregated"),
]


@st.cache_data(show_spinner=False)
def _options() -> Dict[str, List[Tuple[float, str]]]:
    out: Dict[str, List[Tuple[float, str]]] = {}
    for var, _ in DROPDOWN_FIELDS:
        opts = core.options_for(CODEBOOK, var)
        if not opts:
            raise core.ToolError(
                f"The codebook has no rows for {var!r}; the form cannot be built."
            )
        out[var] = opts
    return out


try:
    OPTIONS = _options()
except core.ToolError as exc:
    st.error(str(exc))
    st.stop()


def _default_index(var: str) -> int:
    """Pre-select the training-partition mode where the manifest supplies one."""
    mode = MANIFEST["impute"].get(var)
    if mode is None:
        return 0
    for i, (code, _) in enumerate(OPTIONS[var]):
        if abs(code - float(mode)) < 1e-9:
            return i
    return 0


def _result_block(prob: float, tier: str) -> None:
    """Render one child's result with the disclaimer required by Section 3.13.3."""
    banner = {core.HIGH: st.error, core.MONITOR: st.warning, core.ONTRACK: st.success}[tier]
    banner(f"**{tier}** — P(Severely Stunted) = {prob:.4f}")
    st.markdown(f"**Action protocol (Table 3.7).** {core.ACTION_PROTOCOL[tier]}")
    st.caption(core.DISCLAIMER)
    st.caption(
        "Observed precision of this tier on the held-out test partition "
        f"(Table 4.10): {core.TIER_PRECISION_PCT[tier]:.2f} per cent."
    )


# ==========================================================================
# 4. Pages
# ==========================================================================

st.title("Stunting Risk Screening Tool")
st.caption(
    "Group 20 · CSS200-3 · research prototype, hosted for demonstration. "
    "Screening support only — not a diagnosis."
)

tab_single, tab_batch, tab_thresh, tab_about = st.tabs(
    ["Single child", "Batch upload", "Thresholds", "About"]
)

# ------------------------------------------------------------ single child
with tab_single:
    st.subheader("Enter one child")
    st.caption(
        "Height is not an input. Section 1.6 Limitation 1 excludes it from the "
        "feature set to prevent target leakage, so the tool neither requests "
        "nor accepts it."
    )

    # The unit selector sits outside the form: widgets inside a Streamlit form
    # do not rerun until submission, so the inputs could not switch otherwise.
    age_mode = st.radio(
        "Enter age as",
        ["Age in months", "Date of birth"],
        horizontal=True,
        help="Date of birth reproduces the ENNS age derivation exactly "
             "(days since birth divided by 365.25).",
    )

    with st.form("single_child"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Child**")
            if age_mode == "Age in months":
                age_months = st.number_input(
                    "Age (months)", min_value=0.0, max_value=59.99,
                    value=24.0, step=1.0, format="%.2f",
                    help="Completed months, as recorded in growth monitoring. "
                         "Decimals are accepted.",
                )
            else:
                dob = st.date_input(
                    "Date of birth",
                    value=dt.date.today() - dt.timedelta(days=730),
                    min_value=dt.date.today() - dt.timedelta(days=365 * 7),
                    max_value=dt.date.today(),
                )
                measured = st.date_input(
                    "Date of measurement",
                    value=dt.date.today(),
                    min_value=dt.date.today() - dt.timedelta(days=365 * 7),
                    max_value=dt.date.today() + dt.timedelta(days=365),
                )
            weight = st.number_input(
                "Weight (kg)", min_value=1.0, max_value=40.0,
                value=11.7, step=0.1, format="%.1f",
            )
            bedroom = st.number_input(
                "Number of bedrooms", min_value=0, max_value=9, value=1, step=1,
            )

        values: Dict[str, float] = {}
        for i, (var, label) in enumerate(DROPDOWN_FIELDS):
            target = col_b if i % 2 == 0 else col_c
            with target:
                opts = OPTIONS[var]
                chosen = st.selectbox(
                    label,
                    options=list(range(len(opts))),
                    format_func=lambda j, o=opts: o[j][1],
                    index=_default_index(var),
                    key=f"sel_{var}",
                )
                values[var] = opts[chosen][0]

        submitted = st.form_submit_button("Score this child", type="primary")

    if submitted:
        if age_mode == "Age in months":
            age_years, age_error = age_months / MONTHS_PER_YEAR, ""
        else:
            age_years, age_error = _age_from_dates(dob, measured)

    if submitted and age_error:
        # st.stop() is avoided here because it would also halt rendering of the
        # other tabs on this rerun.
        st.error(age_error)
    elif submitted:
        st.caption(
            f"Age used for scoring: {age_years * MONTHS_PER_YEAR:.2f} months "
            f"({age_years:.4f} years)."
        )
        record = {"age": age_years, "weight": weight, "bedroom": float(bedroom), **values}
        try:
            matrix = core.preprocess(pd.DataFrame([record]), MANIFEST)
            prob = float(core.predict_severe_proba(MODEL, matrix, MANIFEST)[0])
        except core.ToolError as exc:
            st.error(str(exc))
        else:
            st.divider()
            _result_block(prob, core.assign_tier(prob, tau_h, tau_m))

# ------------------------------------------------------------ batch upload
with tab_batch:
    st.subheader("Upload a list of children")

    st.download_button(
        "Download blank template (CSV)",
        data=core.blank_template().to_csv(index=False).encode("utf-8"),
        file_name="stunting_tool_template.csv",
        mime="text/csv",
    )
    st.caption(
        "One row per child, using the ENNS numeric codes. Give age either as "
        "`age` in years, as in ENNS extracts, or as `age_months` in its place. "
        "The optional child_id column is carried through to the output and is "
        "not used for scoring."
    )

    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        try:
            raw = pd.read_csv(upload, low_memory=False)
            if "age" not in raw.columns and "age_months" in raw.columns:
                raw["age"] = (
                    pd.to_numeric(raw["age_months"], errors="coerce") / MONTHS_PER_YEAR
                )
            if "age" in raw.columns and (
                pd.to_numeric(raw["age"], errors="coerce") >= MAX_AGE_YEARS
            ).any():
                raise core.ToolError(
                    "One or more children are five years or older. The model was "
                    "trained on children under five; remove those rows. If the "
                    "age column holds months, rename it to age_months."
                )
            matrix = core.preprocess(raw, MANIFEST)
            probs = core.predict_severe_proba(MODEL, matrix, MANIFEST)
        except core.ToolError as exc:
            st.error(str(exc))
        except Exception as exc:  # malformed CSV, wrong delimiter, and so on
            st.error(f"The file could not be read: {exc}")
        else:
            st.session_state["batch"] = (raw, probs)

    if "batch" in st.session_state:
        raw, probs = st.session_state["batch"]

        # Re-tiering uses the sidebar values and does not re-score, so moving a
        # threshold during a demonstration is instant and provably changes only
        # the tier assignment, never the model's probability.
        tiers = core.assign_tiers(probs, tau_h, tau_m)

        ranked = pd.DataFrame(
            {
                "child_id": (
                    raw["child_id"] if "child_id" in raw.columns
                    else range(1, len(raw) + 1)
                ),
                "p_severely_stunted": probs,
                "priority_tier": tiers.to_numpy(),
            }
        )
        ranked["action_protocol"] = ranked["priority_tier"].map(core.ACTION_PROTOCOL)
        ranked = ranked.sort_values(
            "p_severely_stunted", ascending=False
        ).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))

        cols = st.columns(3)
        for col, tier in zip(cols, core.TIER_ORDER):
            n = int((ranked["priority_tier"] == tier).sum())
            share = 100 * n / len(ranked) if len(ranked) else 0.0
            col.metric(tier, f"{n:,}", f"{share:.2f}% of list", delta_color="off")

        st.caption(core.DISCLAIMER)
        st.dataframe(
            ranked.style.format({"p_severely_stunted": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download ranked priority list (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name="priority_list.csv",
            mime="text/csv",
        )

# -------------------------------------------------------------- thresholds
with tab_thresh:
    st.subheader("Where the thresholds come from")

    if THRESHOLDS:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"**tau_H = {THRESHOLDS['tau_H']:.10f}** — "
                f"{THRESHOLDS.get('tau_H_method', 'Youden J-statistic')}"
            )
            st.write(
                pd.DataFrame(
                    {
                        "Statistic": ["Sensitivity", "Specificity", "Precision"],
                        "At tau_H": [
                            f"{THRESHOLDS['sensitivity_at_tau_H']:.4f}",
                            f"{THRESHOLDS['specificity_at_tau_H']:.4f}",
                            f"{THRESHOLDS['precision_at_tau_H']:.4f}",
                        ],
                    }
                )
            )
        with c2:
            st.markdown(
                f"**tau_M = {THRESHOLDS['tau_M']:.10f}** — "
                f"{THRESHOLDS.get('tau_M_method', 'caseload')}"
            )
            st.write(
                pd.DataFrame(
                    {
                        "Statistic": ["Sensitivity", "Specificity", "Precision"],
                        "At tau_M": [
                            f"{THRESHOLDS['sensitivity_at_tau_M']:.4f}",
                            f"{THRESHOLDS['specificity_at_tau_M']:.4f}",
                            f"{THRESHOLDS['precision_at_tau_M']:.4f}",
                        ],
                    }
                )
            )
        st.caption(
            "These are the exact operating-point figures reported in Section "
            f"4.7.1, computed on the held-out test partition of "
            f"{core.TEST_PARTITION_N:,} children. The two action tiers together "
            f"captured {THRESHOLDS['n_severe_captured_high_plus_monitoring']} of "
            f"{THRESHOLDS['n_severe_total']} severely stunted children, or "
            f"{100 * THRESHOLDS['n_severe_captured_high_plus_monitoring'] / THRESHOLDS['n_severe_total']:.2f} "
            f"per cent, within a screening burden of "
            f"{THRESHOLDS['screening_burden_pct']:.2f} per cent."
        )
    else:
        st.info(
            "artefacts/section_3_11_thresholds.json is absent, so the exact "
            "operating-point statistics cannot be shown."
        )

    if not SURFACE.empty:
        st.divider()
        st.markdown("**Sensitivity–specificity trade-off across the probability range**")
        st.line_chart(
            SURFACE.set_index("threshold")[["sensitivity", "specificity", "precision"]],
            height=320,
        )

        at_h = core.surface_at(SURFACE, tau_h)
        at_m = core.surface_at(SURFACE, tau_m)
        st.write(
            pd.DataFrame(
                [
                    {
                        "Threshold in use": "tau_H",
                        "Value": f"{tau_h:.6f}",
                        "Nearest grid point": f"{at_h['threshold']:.3f}",
                        "Sensitivity": f"{at_h['sensitivity']:.4f}",
                        "Specificity": f"{at_h['specificity']:.4f}",
                        "Precision": f"{at_h['precision']:.4f}",
                    },
                    {
                        "Threshold in use": "tau_M",
                        "Value": f"{tau_m:.6f}",
                        "Nearest grid point": f"{at_m['threshold']:.3f}",
                        "Sensitivity": f"{at_m['sensitivity']:.4f}",
                        "Specificity": f"{at_m['specificity']:.4f}",
                        "Precision": f"{at_m['precision']:.4f}",
                    },
                ]
            )
        )
        st.caption(
            "The trade-off surface is sampled every 0.001, so the row above is "
            "the nearest grid point and not the value at the threshold itself. "
            "At tau_H the nearest grid point lies above tau_H and therefore "
            "reports a slightly lower sensitivity. The figures reported in the "
            "paper are the exact ones shown at the top of this page."
        )
    else:
        st.info(
            "artefacts/section_3_11_threshold_surface.csv is absent, so the "
            "trade-off curve cannot be drawn."
        )

# ------------------------------------------------------------------- about
with tab_about:
    st.subheader("About this tool")
    st.markdown(
        f"""
**Model.** `{MANIFEST['model_name']}` — ADASYN oversampling combined with
XGBoost, identified in Section 4.5 as the best-performing condition under the
selection criterion stated in Research Question 2. Exported from the analysis
notebook `LatestThesisModel_Sept27.ipynb` (cell EP-5; recorded in the manifest under its working name, `SuperSuperLatestColab_Thesis_PATCH8`) on
{MANIFEST.get('exported_at', 'unknown')}, trained on
{MANIFEST.get('n_train', 0):,} records. The tool performs no training of its own.

**Preprocessing.** Reproduced rather than re-estimated: training-partition
medians and modes for missing values, the one-hot scheme fitted in Section
3.3.2 restricted to the {len(MANIFEST['feature_cols'])} features retained by
the Stage 1 filter of Section 3.3.3, and the standard scaler fitted on the
training partition. At start-up the application compares its own copy of those
parameters against the manifest, and the manifest against the booster's own
recorded schema, refusing to score on any discrepancy.

**Tier performance on the held-out test partition** (Table 4.10, Section 4.7.2,
n = {core.TEST_PARTITION_N:,}; base rate {core.BASE_RATE_PCT:.2f} per cent):
"""
    )
    st.table(
        pd.DataFrame(
            {
                "Priority tier": core.TIER_ORDER,
                "Children": [
                    f"{core.TIER_REFERENCE_COUNTS[t]:,}" for t in core.TIER_ORDER
                ],
                "Tier precision": [
                    f"{core.TIER_PRECISION_PCT[t]:.2f}%" for t in core.TIER_ORDER
                ],
                "Action protocol": [core.ACTION_PROTOCOL[t] for t in core.TIER_ORDER],
            }
        )
    )
    st.markdown(
        """
**Limitations.**

- Height is not an input (Section 1.6, Limitation 1).
- The tool is a research prototype hosted for demonstration and examination
  purposes. It has not been deployed into any health service, has not been used
  on a live caseload, and has undergone no prospective validation
  (Section 1.6, Limitation 5).
- tau_M was derived from an operational caseload criterion rather than a
  clinical one. The domain expert endorsed the tiering principle but did not
  select a specific value (Section 3.11.2, Section 4.8).
- Every output is a screening result requiring in-person assessment. The
  majority of children placed in the High Priority tier are not in fact
  severely stunted (Section 4.7.2).

**Reproduction check (Section 3.13.4).** Both stages were executed. Stage A
rebuilt the input matrix from the raw held-out records and reproduced all
thirty-one features for all 8,632 records, with a maximum absolute difference of
6.22e-14. Stage B scored those records through this same code path: the largest
difference in predicted probability against the notebook's own output was
2.96e-08 in the notebook environment and 3.31e-08 on a local machine, and no
child was assigned a different priority tier in either run. The check is not
bundled with the hosted application because it requires the ENNS microdata,
which is not redistributed; it is run from the private repository.
"""
    )
