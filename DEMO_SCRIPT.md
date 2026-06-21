# 🛢️ Bakken Basin Production Advisor — Demo Script

Use this script for your demo recording and for starting the app from scratch.

---

## Starting from a Fresh Terminal

Open PowerShell terminal in VS Code (`Ctrl + `` `) and run each block in order:

### Step 1 — Activate virtual environment
```powershell
.venv\Scripts\activate
```
✅ You should see `(.venv)` at the start of your prompt.

---

### Step 2 — Set environment variables
```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"
$env:LLM_API_KEY="v1.CmQKHHN0YXRpY2tleS1lMDBmeGJyOHcwc2h6MTFjNTASIXNlcnZpY2VhY2NvdW50LWUwMG5qZmd3ZHYzYXlwOHllajIMCNaaqtEGEMzTs94BOgwI153CnAcQwJfosgFAAloDZTAw.AAAAAAAAAAHXOXHW-uQh7B-3ICoE0qnvxuTufAxw-X57GNo6jsoxiruV9MWX1OP4QJ6Ok1yrJAqZ-r48S34JPvB5wO1VISMH"
$env:LLM_PROVIDER="nebius"
$env:LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"
```

---

### Step 3 — Run the test suite (confirm 44 pass)
```powershell
pytest tests/ -v
```
✅ Expected output: `44 passed, 2 warnings`

---

### Step 4 — Launch the advisor
```powershell
python src/app.py --run-id ec6ed26878dd4d10b13deb53e83dd33a
```
✅ Wait for the banner to appear before typing queries.

---

## Demo Queries (type inside the app)

### 🟢 Normal prediction (with early production data)
```
I have a Middle Bakken well with a 10,500 ft lateral, 12 million lbs proppant,
50,000 BBL fluid. First month production was 8,500 BBL, second month 6,200 BBL,
third month 5,100 BBL. What is my first-year forecast?
```
✅ Expected: Realistic prediction ~50,000–90,000 BBL with Bakken benchmarks

---

### 🟡 Edge case 1 — Greeting
```
Hello!
```
✅ Expected: Welcome message with example query

---

### 🟡 Edge case 2 — Off-topic query
```
What is the weather like in Williston today?
```
✅ Expected: Redirects user to provide well parameters

---

### 🟡 Edge case 3 — Missing required feature
```
I have a 10,500 ft lateral in the Middle Bakken with 50,000 BBL of fluid
```
✅ Expected: Asks specifically for Total Proppant (lbs)

---

### 🔴 Exit the app
```
quit
```

---

## Re-running Training (if needed)

Only needed if you change `preprocess.py`, `config.yaml`, or want fresh MLflow runs:

```powershell
# Full pipeline (preprocessing + all 6 models)
python src/train.py --config configs/config.yaml

# Skip preprocessing if ml_ready.csv already exists
python src/train.py --config configs/config.yaml --skip-preprocess
```

Note the new Run ID printed at the end and update Step 4 above.

---

## View MLflow Experiment Results

```powershell
mlflow ui --backend-store-uri mlruns
```
Then open: http://localhost:5000

---

## Project Run ID Reference

| Run | Model | Test R² |
|---|---|---|
| `ec6ed26878dd4d10b13deb53e83dd33a` | xgboost_deep | **0.9621** ✅ Best |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `(.venv)` not showing | Run `.venv\Scripts\activate` |
| `401 Unauthorized` from LLM | API key expired — get new one from studio.nebius.com |
| `404 model not exist` | Run `$env:LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"` |
| `MLFLOW_ALLOW_FILE_STORE` error | Run `$env:MLFLOW_ALLOW_FILE_STORE="true"` before training |
| Tests failing | Make sure you replaced all updated src/ and tests/ files |
