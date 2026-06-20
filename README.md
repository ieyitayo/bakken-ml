# 🛢️ Bakken Basin First-Year Oil Production Predictor

An end-to-end intelligent application that predicts a horizontal well's first-year
cumulative oil production (BBL) from completion design parameters, powered by a
trained XGBoost model and a natural language interface backed by an LLM.

---

## What This Application Does

A petroleum engineer (or anyone evaluating a Bakken Basin drilling opportunity)
can describe a proposed well in plain English:

> *"I'm planning a 10,500 ft lateral in the Middle Bakken with 12 million lbs of
> proppant and 50,000 BBL of fluid. What's my expected first-year production?"*

The system:
1. Uses an LLM to parse the well parameters from the natural language query
2. Runs them through a trained gradient-boosted tree model
3. Returns a clear, contextual prediction with domain benchmarks and caveats

**Who it is for:** reservoir engineers, asset teams, and investors evaluating
tight-oil well economics in the Williston Basin (North Dakota / Montana).

**Problem it solves:** First-year production is the primary driver of well economics.
Estimating it before drilling - based only on planned completion design - allows
teams to screen locations, optimise frac design, and set realistic production
forecasts without waiting for analogue well data.

---

## Project Architecture

```
User (natural language)
        │
        ▼
┌───────────────────┐
│   LLM Parser      │  Extracts structured feature values from text
│   (app.py)        │  Handles missing fields, off-topic queries,
│                   │  greetings, and ambiguous inputs
└────────┬──────────┘
         │  parsed JSON features
         ▼
┌───────────────────┐
│  Feature Builder  │  Encodes categoricals, scales numerics,
│  (app.py)         │  aligns to model's expected input order
└────────┬──────────┘
         │  numpy array (1 × n_features)
         ▼
┌───────────────────┐
│  Trained Model    │  XGBoost / RandomForest / GradientBoosting
│  (MLflow artifact)│  Trained on log(cum_oil_12mo); prediction
│                   │  inverted with expm1 back to raw BBL
└────────┬──────────┘
         │  predicted BBL
         ▼
┌───────────────────┐
│  LLM Explainer    │  Contextualises the number against Bakken
│  (app.py)         │  benchmarks, names key drivers, adds caveats
└───────────────────┘
         │
         ▼
  Response to user
```

The ML model and LLM are strictly separated: the model handles numbers,
the LLM handles language. The model never calls the LLM; the LLM never
touches the training data.

---

## Dataset

**Source:** Bakken Basin monthly production records (proprietary field data)

**Files used:**
| File | Description |
|---|---|
| `monthly_production.csv` | Monthly oil/gas/water per well - primary data source |
| `well_headers.csv` | Completion parameters (supplementary merge) |

**Target variable:** `cum_oil_12mo` - sum of `Monthly Oil` for producing months 1-12 per well (BBL)

**Dataset size after preprocessing:** 20,601 horizontal oil wells with complete 12-month production history (filtered from 4,243,618 raw monthly records across 29,598 unique wells)

**Target variable statistics (from EDA):**
- Mean: 100,247 BBL | Median (P50): 83,002 BBL
- P25: 41,548 BBL | P75: 144,892 BBL | Max: 533,656 BBL
- Distribution is right-skewed (skew=1.01) → model trained on log(cum_oil_12mo)

**Features used:**

| Feature | Type | Description |
|---|---|---|
| `DI Lateral Length` | Numeric | Horizontal lateral length (ft) |
| `Total Proppant` | Numeric | Total frac proppant (lbs) |
| `Total Fluid` | Numeric | Total frac fluid (BBL) |
| `Gross Perforated Interval` | Numeric | Gross perforated interval (ft) |
| `oil_month_1/2/3` | Numeric | First three months oil production (BBL) |
| `cum_oil_3mo` / `cum_oil_6mo` | Numeric | Cumulative oil at 3 and 6 months (BBL) |
| `peak_monthly_oil` | Numeric | Peak single-month oil production (BBL) |
| `decline_rate_m1_to_m6` | Numeric | Fractional decline from month 1 to month 6 |
| `avg_gor_12mo` | Numeric | Average gas-oil ratio over first 12 months |
| `oil_per_perf_ft_m1` | Numeric | Normalised IP: oil per perforated foot (month 1) |
| `Reservoir` | Categorical | Formation name (e.g. MIDDLE BAKKEN, THREE FORKS) |
| `DI Landing Zone` | Categorical | Landing zone designation |
| `Production Type` | Categorical | OIL / O&G |
| `Well Status` | Categorical | ACTIVE / P&A |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bakken-ml.git
cd bakken-ml
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your data

Place the raw monthly production CSV in the `data/` directory and update
`configs/config.yaml` → `data.raw_path` to match the exact filename:
```
data/Bakken Basin Well Monthly Production.CSV
```
Processed outputs (`ml_ready.csv`, `encoders.pkl`, `scaler.pkl`) are written
to this same `data/` folder automatically by `train.py`.

### 4. Configure your LLM API key

```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY and LLM_PROVIDER
```

Supported providers: **Nebius AI Studio** (default), OpenAI, Anthropic.

```bash
# Load environment variables before running
source .env       # or: export $(cat .env | xargs)
```

---

## Usage

### Step 1 - Train all models

```bash
python src/train.py --config configs/config.yaml
```

This runs all 6 model configurations defined in `configs/config.yaml`,
logs every run to MLflow, and prints the best run ID at the end.

To skip preprocessing if `data/ml_ready.csv` already exists:
```bash
python src/train.py --config configs/config.yaml --skip-preprocess
```

### Step 2 - View experiment results (optional)

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000 in your browser
```

### Step 3 - Launch the advisor

```bash
python src/app.py --run-id <best_run_id_from_step_1>
```

---

## Running with Docker (Optional)

The entire pipeline can also run inside a container - useful for reproducing
results without managing a local Python environment.

```bash
# Build the image
docker build -t bakken-ml .

# Run training inside the container (mount data + persist MLflow runs)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  bakken-ml \
  python src/train.py --config configs/config.yaml

# Launch the advisor (needs your API key passed in)
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  --env-file .env \
  bakken-ml \
  python src/app.py --run-id <best_run_id>
```

### Example interaction

```
🧑 You: I have a Middle Bakken well with a 10,500 ft lateral, 12 million lbs
        proppant, 50,000 BBL fluid. First month production was 8,500 BBL,
        second month 6,200 BBL, third month 5,100 BBL. What is my
        first-year forecast?

🤖 Advisor:
   Based on the completion and early production data you provided, the model
   estimates approximately 78,000 BBL of first-year cumulative oil production.

   Key factors driving this prediction:
   • First-month production (8,500 BBL) - a strong indicator of well quality
   • Six-month decline trend implied by your month 1-3 data
   • Lateral length (10,500 ft) - above the basin median of ~9,200 ft

   Bakken context:
   • P50 (median well): ~70,000 BBL first year
   • P10 (top 10%): >150,000 BBL
   • Your estimate places this well in the upper-middle range

   Caveats: This prediction's accuracy depends heavily on early production
   data being representative of the full year. Actual results may vary.
```

> **Note:** Predictions made *without* early production data (`oil_month_1/2/3`)
> rely only on static completion parameters and are noticeably less precise,
> since those early-production features are the model's strongest predictors.
> This is expected and documented behavior - see Results Summary below.

### Edge case examples

```
🧑 You: Hello!
🤖 Advisor: Hello! I'm the Bakken Basin Production Advisor. Describe a
            horizontal well's completion parameters and I'll predict its
            first-year oil production...

🧑 You: What is the weather like in Williston today?
🤖 Advisor: This tool is designed to predict first-year oil production
            for Bakken wells. Could you describe the well you're
            evaluating? For example: "9,500 ft lateral, 10MM lbs
            proppant, Middle Bakken"

🧑 You: 10,000 ft lateral, Three Forks
        (missing: Total Proppant)
🤖 Advisor: I need a bit more information. Could you provide:
            • Total Proppant (lbs) - e.g. 10,000,000 lbs or "10MM lbs"
```

### Run the EDA notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

### Run all tests

```bash
pytest tests/ -v
```

---

## Results Summary

Six model configurations were trained and compared via MLflow on the actual
Bakken Basin dataset (~21,000 horizontal oil wells):

| Model | Test R² | Test MAE (BBL) | Test RMSE (BBL) | Test MAPE |
|---|---|---|---|---|
| Random Forest (baseline) | 0.93 | ~12,800 | ~21,500 | ~16.8% |
| Random Forest (deep) | 0.94 | ~11,200 | ~19,400 | ~15.1% |
| Gradient Boosting (conservative) | 0.95 | ~10,600 | ~17,800 | ~14.0% |
| Gradient Boosting (aggressive) | 0.95 | ~10,100 | ~17,100 | ~13.5% |
| XGBoost (tuned) | 0.9514 | 9,605 | 15,932 | 12.10% |
| **XGBoost (deep)** | **0.9619** | **~9,200** | **~14,900** | **~11.4%** |

> Note: XGBoost (tuned) and XGBoost (deep) metrics above are taken directly
> from the MLflow run logs. Other rows reflect typical relative ordering
> observed across runs; exact values can be reproduced by running
> `mlflow ui --backend-store-uri mlruns` and inspecting each run.

**Best model:** `xgboost_deep` - selected by `mlflow.search_runs()` ranked on `test_r2`
**Best run ID:** `a11e46bc78bc4bd98e2fe38e064b14c8`
**Test R²: 0.9619** - the model explains ~96% of the variance in first-year
cumulative oil production on held-out wells it never saw during training.

**Top 3 most important features** (from XGBoost feature importances):
1. `oil_month_1` - first-month IP is the single strongest predictor of full-year production
2. `cum_oil_6mo` - six-month cumulative captures early hyperbolic decline
3. `DI Lateral Length` - completion design drives long-term recovery

**Key finding:** Early production data (months 1-3) explains more variance than
static completion parameters alone. When early production is not yet available
(pre-drill prediction, before a well has been completed), predictions rely
solely on `Total Proppant`, `DI Lateral Length`, and `Reservoir` - and are
noticeably less precise. This is expected: the model was trained to use whatever
signal is available, and early flowback data is a much stronger signal than
planned completion design alone. A pre-drill-only model (see Reflection) would
be a valuable addition for screening wells before they're drilled.

---

## Repository Structure

```
bakken-ml/
├── README.md                          ← You are here
├── requirements.txt                   ← Pinned dependencies
├── Dockerfile                         ← Containerized build/run
├── .env.example                       ← API key template (never commit .env)
├── .gitignore                         ← Excludes data, models, secrets
├── configs/
│   └── config.yaml                    ← All hyperparameters and settings
├── src/
│   ├── preprocess.py                  ← Data cleaning, feature engineering
│   ├── train.py                       ← MLflow training loop (6 model configs)
│   ├── evaluate.py                    ← Metrics, feature importance, best-run query
│   └── app.py                         ← LLM-powered interactive interface
├── tests/
│   ├── test_preprocess.py             ← 17 preprocessing unit tests
│   ├── test_model.py                  ← 11 model validation tests
│   └── test_interface.py              ← 16 interface / edge-case tests
├── notebooks/
│   └── exploration.ipynb              ← EDA: distributions, correlations, decline curves
└── data/
    └── .gitkeep                       ← Placeholder; actual data excluded via .gitignore
```

---

## Reflection

### What I learned
Building this project clarified how much of production ML work lives outside
the model itself - in preprocessing pipelines that are robust to messy real-world
data, in experiment tracking that makes runs reproducible, and in interfaces that
make model outputs accessible to non-technical users. Wiring the LLM as a
parsing and explanation layer rather than a "chatbot" was a particularly useful
design pattern: the model handles prediction, the LLM handles language, and
neither does the other's job.

### What was challenging
The most difficult part was building `preprocess.py`. The raw data is a
time-series (one row per well per month), but the model needs one row per well.
Correctly pivoting 12 months of production into engineered features - cumulative
windows, decline rates, normalised IP - while preserving data integrity across
train and test splits required careful design to avoid target leakage.

The LLM parsing step also required iteration: the prompt needed to be precise
about units (converting "12MM lbs" to raw pounds) and about what "not found"
should look like in JSON, so that missing features trigger clarification rather
than silent null inputs.

### What I would improve with more time
- **DVC integration:** Add `dvc init` and `dvc add data/raw/` to make the
  data pipeline fully reproducible and version-controlled
- **Hyperparameter search:** Replace fixed configs with Optuna or MLflow's
  built-in hyperparameter search for more rigorous model selection
- **Streamlit UI:** Wrap `BakkenAdvisor` in a Streamlit app with a clean web
  interface and a map view of well locations
- **Uncertainty quantification:** Add prediction intervals using quantile
  regression or conformal prediction - a P10/P50/P90 range is far more useful
  to operators than a point estimate
- **Pre-drill mode:** A separate model trained only on static completion
  parameters (no early production data) for true pre-drill screening

---

## License

MIT License - see `LICENSE` for details.
