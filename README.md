# ARISEN-Shock: Interpretable Septic Shock Early-Warning Research Pipeline

This repository contains code and outputs from an ongoing research project focused on interpretable septic shock early-warning using ICU patient data, machine learning, rule mining, and agent-based LLM reasoning.

The project began with baseline zero-shot and few-shot prompting experiments using large language models. After observing that raw LLM prompting alone was not reliable enough for clinical reasoning over continuous ICU variables, the project shifted toward an interpretable agentic framework called **ARISEN-Shock**.

**ARISEN-Shock** stands for:

**Agent-based Recursive Interpretable Sepsis Early-warning Network for Septic Shock**

The goal is to move beyond black-box prediction and build a system that can provide evidence-backed reasoning for why a patient may be at risk of sepsis or septic shock.

---

## Project Overview

Septic shock is difficult to detect early because its symptoms can overlap with many other clinical conditions. Traditional AI models may predict risk, but they often do not explain their reasoning clearly enough for clinical use.

This project explores a multi-agent framework where different components contribute evidence:

- **Agent A: Data Summarizer**  
  Converts raw ICU vitals into interpretable clinical rules using data-driven discretization.

- **Agent B: Knowledge Specialist**  
  Retrieves relevant clinical knowledge, guidelines, and literature.

- **Agent C: Differential Diagnostician**  
  Helps rule out mimicking conditions before reaching a conclusion.

My work focused primarily on the data engineering and machine learning layer that supports Agent A.

---

## My Contributions

My role focused on data engineering, preprocessing, feature analysis, and interpretable rule generation.

I contributed to:

- Preprocessing and merging two ICU datasets: PhysioNet 2019 and PHEMS
- Cleaning and balancing patient-level sepsis datasets
- Comparing clinical variable overlap between PhysioNet and PHEMS
- Applying correlation analysis and mean comparison analysis
- Applying Random Forest feature importance
- Applying Decision Tree threshold analysis
- Building a discretization pipeline for continuous ICU vitals
- Applying FP-Growth association rule mining to generate interpretable sepsis rules
- Testing baseline Gemini zero-shot and few-shot prompting experiments

Example rule format:

```text
Temp_high + O2Sat_low + Resp_high → Sepsis
```

These rules are intended to help ground Agent A with interpretable, data-driven clinical patterns.

---

## Repository Structure

```text
septic-shock-research-github/
│
├── src/
│   ├── baseline_prompting/
│   │   └── gemini_0shot_and_fewshot_with_context.py
│   │
│   ├── preprocessing/
│   │   ├── physionet_data_analysis.py
│   │   └── phems_merge_and_balance.py
│   │
│   ├── dataset_comparison/
│   │   └── variable_comparison_betweenDatasets.py
│   │
│   ├── feature_analysis/
│   │   ├── mean_correlation_analysis.py
│   │   ├── random_forest_importance.py
│   │   └── decision_tree_thresholds.py
│   │
│   ├── discretization/
│   │   └── discretization_pipeline.py
│   │
│   └── rule_mining/
│       └── fp_growth_rules.py
│
├── outputs/
│   ├── baseline_prompting/
│   ├── variable_comparison/
│   ├── physionet_analysis/
│   ├── feature_analysis/
│   ├── decision_trees/
│   ├── discretization/
│   └── association_rules/
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── experiments/
│   └── README.md
│
├── poster/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Pipeline

The main workflow is:

```text
Raw ICU datasets
→ preprocessing and cleaning
→ dataset variable comparison
→ feature analysis
→ decision tree threshold extraction
→ discretization of continuous vitals
→ FP-Growth association rule mining
→ interpretable rule outputs for Agent A
```

---

## File Paths

Scripts do **not** use hardcoded absolute paths. Every script finds the repo root
on its own using the exact same line:

```python
# Repo root: walk up until we find requirements.txt (works at any nesting depth).
ROOT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "requirements.txt").exists())

CSV_PATH = ROOT_DIR / "data" / "processed" / "physionet_balanced.csv"
```

How it works: starting from the script's own location, it walks up the folders
until it finds `requirements.txt` (the marker that sits at the repo root). That
folder becomes `ROOT_DIR`, and every data/output path is built from it.

What this means for you:

- **Just clone and run** — paths work on any machine, no editing needed.
- All data and output paths are built from `ROOT_DIR`, never typed out by hand.
- Keep data where the repo expects it: `data/` and `outputs/` under the repo root.
- **It does not matter how deep a script is nested** — there is no folder-depth
  number to keep in sync. You can move or copy a script anywhere in the repo and
  it still finds the root. (This replaced the old, fragile `parents[N]` approach.)
- If you add a new script, just copy that same `ROOT_DIR` line — it is identical
  in every file on purpose.
- If your data must live outside the repo, don't change code — symlink it in:
  ```bash
  ln -s /path/to/your/data data
  ```

> **Known issue:** `gemini_0shot_with_context.py` and `gemini_0shot_and_fewshot_with_context.py`
> still use a relative `CSV_PATH = "Dataset_balanced.csv"` instead of the `ROOT_DIR` pattern.
> They need converting once we confirm whether that file is a real separate dataset or just `physionet_balanced.csv`.

---

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python src/preprocessing/physionet_data_analysis.py
python src/preprocessing/phems_merge_and_balance.py
```

Run dataset comparison:

```bash
python src/dataset_comparison/variable_comparison_betweenDatasets.py
```

Run feature analysis:

```bash
python src/feature_analysis/mean_correlation_analysis.py
python src/feature_analysis/random_forest_importance.py
python src/feature_analysis/decision_tree_thresholds.py
```

Run discretization and rule mining:

```bash
python src/discretization/discretization_pipeline.py
python src/rule_mining/fp_growth_rules.py
```

---

## Data

The full raw and processed datasets are not included in this repository due to file size, licensing, and privacy considerations.

See `data/README.md` for the expected local dataset structure.

---

## Outputs

Generated outputs include:

- Dataset comparison tables and figures
- PhysioNet exploratory analysis figures
- Mean comparison plots
- Correlation ranking plots
- Random Forest feature importance plots
- Decision Tree visualizations
- Discretized patient-level datasets
- FP-Growth association rules for sepsis prediction

---

## Experiments

The `experiments/` folder is reserved for exploratory work that is not yet part of the finalized pipeline.

Use it for:

- New prompt tests
- Agentic model prototypes
- Threshold experiments
- Additional model comparisons
- Temporary scripts before they are cleaned and moved into `src/`

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- MLxtend
- Google GenAI SDK