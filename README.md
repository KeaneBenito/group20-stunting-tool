# Stunting Risk Screening Tool

Research prototype accompanying the undergraduate thesis *Addressing Class Imbalance in
Childhood Stunting Classification* (Group 20, CSS200-1). It operationalises the
best-performing classifier reported in Chapter 4 — ADASYN oversampling with XGBoost, refitted
after the encoding revision (twelve nominal variables one-hot encoded; 44 feature columns) — as a
three-tier screening interface.

**This is not a medical device.** Every output is a screening result requiring in-person
assessment by a health professional. The tool has not been deployed into any health service,
has not been used on a live caseload, and has undergone no prospective validation.

## What it does

| Mode | Purpose |
|:--|:--|
| Single child | Scores one child entered by age (months or date of birth), weight, sex and household conditions |
| Batch upload | Scores a CSV of children and returns a ranked priority list |
| Thresholds | Shows the operating points and the sensitivity–specificity trade-off |
| About | Model provenance, tier performance and limitations |

Children are placed in one of three tiers by their predicted probability of severe stunting:
High Priority at or above τ_H = 0.10098795, Needs Monitoring from τ_M = 0.057 up to τ_H, and
On Track below τ_M. Both thresholds are read from `artefacts/manifest.json` and are adjustable
at runtime.

## Repository layout

```
app.py                 Streamlit interface
stunting_core.py       Inference core — guards, preprocessing, tiering
self_test.py           Two-stage reproduction check (requires microdata; not run in the cloud)
requirements.txt       Pinned dependencies
artefacts/             Exported model bundle (model_Exp4_ADASYN_XGB_v2.json, manifest.json),
                       ENNS codebook, threshold statistics and reproduction_check.json
```

## Data

**No ENNS survey records are included in this repository.** The microdata is held under the
team's data use arrangement with FNRI-DOST and is not redistributed. `test_data/` and the demo
CSVs are excluded by `.gitignore`. The exported model artefact contains fitted tree parameters
only, not the records used to fit them.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The application verifies its own preprocessing parameters against `artefacts/manifest.json`, and
the manifest against the booster's recorded schema, at start-up. It refuses to score on any
discrepancy.

## Reproduction check

With the ENNS microdata placed in `test_data/`, `python self_test.py` rebuilds the model input
matrix from the raw held-out records and compares it against the notebook's own matrix, then
scores those records and compares probabilities and tier assignments against the notebook's
output. When both stages pass it writes the aggregate results to
`artefacts/reproduction_check.json`, which the About tab displays. Recorded in the notebook
environment (cell EP-5 v2): all 44 columns matched across 8,632 records (max difference
6.22e-14); largest probability difference 2.97e-08; zero tier disagreements. The `test_data/`
files must come from the v2 run.

## Model provenance

`Exp4_ADASYN_XGB` (v2 encoding), exported 9 October 2026 by cell EP-5 v2 of
`LatestThesisModel_v2_FinalRun.ipynb`, fitted on 34,524 training records under XGBoost 3.4.1.
The application performs no training. The version used at the 9 October 2026 defense and in the
expert review (31 features) remains available under the git tag `v1-defense-2026-10-09`.
