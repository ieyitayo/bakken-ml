"""
app.py
------
LLM-powered natural language interface for the Bakken Basin
first-year oil production prediction model.

A user types a plain-English query describing a well's completion
parameters. The LLM parses the values, the trained model predicts
first-year oil production, and the LLM explains the result in
plain English with domain context.

Usage (Jupyter notebook interactive loop):
    python src/app.py --run-id <best_mlflow_run_id> \
                      --config configs/config.yaml

Or import and call directly:
    from app import BakkenAdvisor
    advisor = BakkenAdvisor(run_id="...", config_path="configs/config.yaml")
    advisor.chat("I have a 10,000 ft lateral, 10MM lbs proppant ...")
"""

from evaluate import load_model_for_inference
import argparse
import json
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import requests
import yaml

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LLM client (provider-agnostic)
# ─────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around any OpenAI-compatible chat completions endpoint.
    Reads provider settings from environment variables so API keys are
    never hardcoded.

    Supported providers (set LLM_PROVIDER env var):
      - "nebius"    → Nebius AI Studio  (default, as used in Sprint 16)
      - "openai"    → OpenAI API
      - "anthropic" → Anthropic API (via messages endpoint)

    Required env vars:
      LLM_API_KEY   — your API key
      LLM_PROVIDER  — one of: nebius | openai | anthropic  (default: nebius)
      LLM_MODEL     — model name override (optional)
    """

    PROVIDER_DEFAULTS = {
        "nebius": {
            "base_url": "https://api.studio.nebius.com/v1",
            "model":    "meta-llama/Meta-Llama-3.1-70B-Instruct",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model":    "gpt-4o-mini",
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "model":    "claude-3-haiku-20240307",
        },
    }

    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.provider = os.getenv("LLM_PROVIDER", "nebius").lower()
        self.model = os.getenv("LLM_MODEL") or \
            self.PROVIDER_DEFAULTS[self.provider]["model"]
        self.base_url = self.PROVIDER_DEFAULTS[self.provider]["base_url"]

        if not self.api_key:
            raise EnvironmentError(
                "LLM_API_KEY environment variable is not set.\n"
                "Set it with:  export LLM_API_KEY='your-key-here'\n"
                "Or add it to your .env file."
            )
        logger.info(f"LLM client: provider={self.provider} model={self.model}")

    def complete(self, system_prompt: str, user_message: str, max_tokens: int = 800) -> str:
        """Send a chat completion request and return the response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "max_tokens":   max_tokens,
            "temperature":  0.2,   # low temp for consistent feature extraction
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "LLM API request timed out. Check your connection.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"LLM API error {resp.status_code}: {resp.text}") from e


# ─────────────────────────────────────────────
# Feature schema — what the model needs
# ─────────────────────────────────────────────

FEATURE_SCHEMA = {
    # name              : (unit,          description,                          example)
    "DI Lateral Length": ("ft",          "Horizontal lateral length",          "9500"),
    "Total Proppant": ("lbs",         "Total proppant (frac sand)",         "10000000"),
    "Total Fluid": ("BBL",         "Total frac fluid volume",            "45000"),
    "Gross Perforated Interval": ("ft",   "Gross perforated interval",          "9200"),
    "Reservoir": ("category",    "Reservoir name",                     "MIDDLE BAKKEN"),
    "DI Landing Zone": ("category",    "Landing zone",                       "BAKKEN"),
    "oil_month_1": ("BBL",         "First month oil production",         "8000"),
    "oil_month_2": ("BBL",         "Second month oil production",        "6500"),
    "oil_month_3": ("BBL",         "Third month oil production",         "5500"),
}

REQUIRED_FEATURES = [
    "DI Lateral Length",
    "Total Proppant",
    "Reservoir",
]

OPTIONAL_FEATURES = [
    "Total Fluid",
    "Gross Perforated Interval",
    "DI Landing Zone",
    "oil_month_1",
    "oil_month_2",
    "oil_month_3",
]


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

PARSE_SYSTEM_PROMPT = """You are a petroleum engineering data extraction assistant.
Your ONLY job is to extract well completion parameters from the user's message
and return them as a JSON object.

Extract these fields if present:
- "DI Lateral Length": horizontal lateral length in feet (number)
- "Total Proppant": total proppant/frac sand in pounds (number). Convert MM lbs or million lbs to raw pounds (e.g. 10 MM lbs = 10000000)
- "Total Fluid": total frac fluid in barrels BBL (number)
- "Gross Perforated Interval": gross perforated interval in feet (number)
- "Reservoir": reservoir formation name as a string (e.g. "MIDDLE BAKKEN", "THREE FORKS")
- "DI Landing Zone": landing zone name as a string (e.g. "BAKKEN", "THREE FORKS")
- "oil_month_1": first month oil production in BBL (number)
- "oil_month_2": second month oil production in BBL (number)
- "oil_month_3": third month oil production in BBL (number)

Rules:
1. Return ONLY valid JSON — no explanation, no preamble, no markdown fences.
2. Only include fields that are clearly stated or strongly implied.
3. If a value is ambiguous, do NOT include it.
4. If NO well parameters are present (e.g. the user is asking a general question), return: {"off_topic": true}
5. If the message is a greeting or casual chat, return: {"greeting": true}

Example input: "I'm drilling a 10,500 ft lateral in the Middle Bakken with 12 million lbs of proppant and 50,000 BBL of fluid"
Example output: {"DI Lateral Length": 10500, "Total Proppant": 12000000, "Total Fluid": 50000, "Reservoir": "MIDDLE BAKKEN"}
"""

RESPONSE_SYSTEM_PROMPT = """You are a petroleum engineering advisor specializing in
Bakken Basin tight-oil wells. You explain ML model predictions clearly and helpfully
to both engineers and non-technical stakeholders.

When given a prediction, your response must include:
1. The predicted first-year cumulative oil production in BBL and K BBL
2. A plain-English interpretation of what that number means
3. Which input parameters most influenced the prediction
4. Relevant context about Bakken Basin performance benchmarks
5. Any important caveats about the prediction's reliability

Keep the tone professional but approachable. Use bullet points for the key factors.
Limit your response to 250 words.
"""

MISSING_FEATURES_PROMPT = """You are a petroleum engineering assistant.
The user is trying to predict first-year oil production for a Bakken well,
but their message is missing some required information.

Missing required fields: {missing}

Ask the user specifically for the missing information in a friendly,
concise way. Mention the units for each field. Do not ask for all
optional fields — only the required ones that are missing.
"""

CLARIFICATION_PROMPT = """You are a petroleum engineering assistant.
The user sent a message that does not appear to contain well completion parameters.
Their message: "{message}"

Politely explain that this tool predicts first-year Bakken oil production,
and ask them to describe their well's completion parameters. Give one
concrete example of what a valid query looks like. Keep it under 60 words.
"""


# ─────────────────────────────────────────────
# Feature parser
# ─────────────────────────────────────────────

def parse_features_from_text(
    user_message: str,
    llm: LLMClient,
) -> dict:
    """
    Use the LLM to extract well parameters from natural language.

    Returns a dict with parsed values, plus special keys:
      "__off_topic__"  : True if the message has no well parameters
      "__greeting__"   : True if it's a greeting/casual message
      "__parse_error__": True if the LLM returned malformed JSON
    """
    raw = llm.complete(
        system_prompt=PARSE_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=300,
    )

    # Strip any accidental markdown fences
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"LLM returned non-JSON: {raw!r}")
        return {"__parse_error__": True, "__raw__": raw}

    # Normalise special flags
    result = {}
    if parsed.get("off_topic"):
        result["__off_topic__"] = True
        return result
    if parsed.get("greeting"):
        result["__greeting__"] = True
        return result

    # Coerce numeric values and normalise strings
    for key, val in parsed.items():
        schema_entry = FEATURE_SCHEMA.get(key)
        if schema_entry and schema_entry[0] == "category":
            result[key] = str(val).strip().upper()
        else:
            try:
                result[key] = float(val)
            except (TypeError, ValueError):
                result[key] = val

    return result


# ─────────────────────────────────────────────
# Feature vector builder
# ─────────────────────────────────────────────

def derive_missing_features(parsed: dict) -> dict:
    """
    Automatically compute derived features from monthly oil values
    that the user provided, so the model receives realistic inputs
    rather than zero-defaults.

    Derived features:
      - cum_oil_3mo  : sum of months 1-3
      - cum_oil_6mo  : sum of months 1-6 (uses months 4-6 = month3 * decay if absent)
      - peak_monthly_oil : max of available monthly values
      - decline_rate_m1_to_m6 : (m1 - m6) / m1
      - oil_per_perf_ft_m1 : oil_month_1 / Gross Perforated Interval (if available)
    """
    parsed = parsed.copy()

    m1 = parsed.get("oil_month_1", 0) or 0
    m2 = parsed.get("oil_month_2", 0) or 0
    m3 = parsed.get("oil_month_3", 0) or 0

    # Only compute if at least month 1 is present
    if m1 > 0:
        # cum_oil_3mo
        if "cum_oil_3mo" not in parsed:
            parsed["cum_oil_3mo"] = m1 + m2 + m3

        # Estimate months 4-6 using typical Bakken hyperbolic decline (~15%/month)
        # if user didn't provide them
        if "cum_oil_6mo" not in parsed:
            decay = 0.85  # typical monthly decline rate
            m4 = m3 * decay if m3 > 0 else m2 * decay * decay
            m5 = m4 * decay
            m6 = m5 * decay
            parsed["cum_oil_6mo"] = parsed["cum_oil_3mo"] + m4 + m5 + m6

        # peak monthly oil
        if "peak_monthly_oil" not in parsed:
            parsed["peak_monthly_oil"] = max(m1, m2, m3)

        # decline rate month 1 → estimated month 6
        if "decline_rate_m1_to_m6" not in parsed and m1 > 0:
            m6_est = parsed["cum_oil_6mo"] - parsed["cum_oil_3mo"]
            m6_est = m6_est / 3  # rough average of months 4-6
            parsed["decline_rate_m1_to_m6"] = max(
                0, min(1, (m1 - m6_est) / m1))

        # oil per perforated ft (month 1)
        if "oil_per_perf_ft_m1" not in parsed:
            perf = parsed.get("Gross Perforated Interval")
            if perf and perf > 0:
                parsed["oil_per_perf_ft_m1"] = m1 / perf

    return parsed


def build_feature_vector(
    parsed: dict,
    feature_cols: list[str],
    encoders: dict,
    scaler,
    cfg: dict,
) -> np.ndarray | None:
    """
    Convert parsed feature dict into a scaled numpy array
    ready for model inference.

    Returns None if required features are missing.
    """
    missing = [f for f in REQUIRED_FEATURES if f not in parsed]
    if missing:
        return None, missing

    # Build a single-row DataFrame with all expected feature columns
    row = {}
    for col in feature_cols:
        if col in parsed:
            row[col] = parsed[col]
        elif col in cfg["features"]["numeric"]:
            # Use median (0 after StandardScaling) for missing optional numerics
            row[col] = 0.0
        else:
            row[col] = "UNKNOWN"

    df_row = pd.DataFrame([row])

    # Encode categoricals using fitted encoders
    from preprocess import CATEGORICAL_COLS
    for col in CATEGORICAL_COLS:
        if col in df_row.columns and col in encoders:
            le = encoders[col]
            val = str(df_row.at[0, col]).strip().upper()
            if val not in le.classes_:
                val = "UNKNOWN"
                if val not in le.classes_:
                    # Fallback: use first class
                    val = le.classes_[0]
            df_row.at[0, col] = le.transform([val])[0]

    # Scale numeric features
    from preprocess import NUMERIC_FEATURE_COLS
    scale_cols = [c for c in NUMERIC_FEATURE_COLS if c in df_row.columns]
    if scale_cols:
        df_row[scale_cols] = scaler.transform(df_row[scale_cols])

    # Align to exact model input order
    X = df_row[feature_cols].values.astype(float)
    return X, []


# ─────────────────────────────────────────────
# Response generator
# ─────────────────────────────────────────────

def generate_response(
    prediction_bbl: float,
    parsed_features: dict,
    feature_cols: list[str],
    model,
    llm: LLMClient,
) -> str:
    """
    Build a context-rich prompt and ask the LLM to explain
    the prediction in plain English.
    """
    # Build feature importance context if available
    fi_context = ""
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_cols, model.feature_importances_))
        top3 = sorted(importances.items(),
                      key=lambda x: x[1], reverse=True)[:3]
        fi_context = (
            "Top 3 most influential model features overall: "
            + ", ".join(f"{k} ({v:.3f})" for k, v in top3)
        )

    user_provided = {
        k: v for k, v in parsed_features.items()
        if not k.startswith("__")
    }

    context_prompt = f"""
Prediction result:
- First-year cumulative oil production: {prediction_bbl:,.0f} BBL ({prediction_bbl/1000:.1f}K BBL)

Well parameters provided by the user (+ auto-derived features):
{json.dumps(user_provided, indent=2)}

Note: Cumulative and peak features (cum_oil_3mo, cum_oil_6mo, peak_monthly_oil,
decline_rate_m1_to_m6) were automatically computed from the monthly production
values the user provided using typical Bakken hyperbolic decline rates.

{fi_context}

Bakken Basin context (from 20,601 wells with complete 12-month history):
- P25 (bottom 25% wells): ~41,500 BBL first year
- P50 (median well): ~83,000 BBL first year
- P75 (top 25%): ~144,900 BBL first year
- P10 (top 10%): >200,000 BBL
- Mean: ~100,200 BBL | Max observed: ~534,000 BBL
- Wells in the Middle Bakken and Three Forks intervals are the primary producers

Please provide a clear, helpful explanation of this prediction.
"""
    return llm.complete(
        system_prompt=RESPONSE_SYSTEM_PROMPT,
        user_message=context_prompt,
        max_tokens=400,
    )


# ─────────────────────────────────────────────
# Main advisor class
# ─────────────────────────────────────────────

class BakkenAdvisor:
    """
    End-to-end conversational interface for the Bakken oil production model.

    Usage:
        advisor = BakkenAdvisor(run_id="abc123", config_path="configs/config.yaml")
        response = advisor.chat("10,500 ft lateral, 12MM lbs proppant, Middle Bakken")
        print(response)
    """

    def __init__(self, run_id: str, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.llm = LLMClient()

        self.model, self.encoders, self.scaler = load_model_for_inference(
            run_id=run_id,
            encoders_path=self.cfg["data"]["encoders_path"],
            scaler_path=self.cfg["data"]["scaler_path"],
            tracking_uri=self.cfg["mlflow"]["tracking_uri"],
        )

        self.feature_cols = (
            [c for c in self.cfg["features"]["numeric"]]
            + [c for c in self.cfg["features"]["categorical"]]
        )

        self.history: list[dict] = []
        logger.info("BakkenAdvisor ready ✓")

    def chat(self, user_message: str) -> str:
        """
        Process a natural language query and return a prediction + explanation.
        Handles edge cases: off-topic queries, missing features, parse errors.
        """
        self.history.append({"role": "user", "content": user_message})

        # ── Step 1: Parse features from natural language ──
        parsed = parse_features_from_text(user_message, self.llm)

        # ── Auto-derive cumulative/peak features from monthly values ──
        if not any(k.startswith("__") for k in parsed):
            parsed = derive_missing_features(parsed)

        # ── Edge case: greeting ──
        if parsed.get("__greeting__"):
            response = (
                "Hello! I'm the Bakken Basin Production Advisor.\n\n"
                "Describe a horizontal well's completion parameters and I'll predict "
                "its first-year oil production. For example:\n\n"
                "  \"I have a 9,500 ft lateral in the Middle Bakken with 10 million lbs "
                "of proppant and 45,000 BBL of fluid. What's my expected first-year production?\""
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # ── Edge case: off-topic or parse error ──
        if parsed.get("__off_topic__") or parsed.get("__parse_error__"):
            response = self.llm.complete(
                system_prompt="",
                user_message=CLARIFICATION_PROMPT.format(message=user_message),
                max_tokens=120,
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # ── Edge case: missing required features ──
        missing = [f for f in REQUIRED_FEATURES if f not in parsed]
        if missing:
            prompt = MISSING_FEATURES_PROMPT.format(
                missing=", ".join(
                    f"{f} ({FEATURE_SCHEMA[f][0]})" for f in missing
                )
            )
            response = self.llm.complete(
                system_prompt="",
                user_message=prompt,
                max_tokens=150,
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # ── Step 2: Build feature vector ──
        X, missing_after = build_feature_vector(
            parsed=parsed,
            feature_cols=self.feature_cols,
            encoders=self.encoders,
            scaler=self.scaler,
            cfg=self.cfg,
        )

        if X is None:
            response = (
                f"I still need the following to make a prediction: "
                f"{', '.join(missing_after)}. Could you provide those?"
            )
            self.history.append({"role": "assistant", "content": response})
            return response

        # ── Step 3: Model inference ──
        log_pred = self.model.predict(X)[0]
        pred_bbl = float(np.expm1(log_pred)) if self.cfg["data"].get(
            "log_transform_target", True
        ) else float(log_pred)

        logger.info(f"Prediction: {pred_bbl:,.0f} BBL")

        # ── Step 4: LLM explanation ──
        response = generate_response(
            prediction_bbl=pred_bbl,
            parsed_features=parsed,
            feature_cols=self.feature_cols,
            model=self.model,
            llm=self.llm,
        )

        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        """Clear conversation history."""
        self.history = []
        print("Conversation history cleared.")


# ─────────────────────────────────────────────
# Interactive Jupyter loop
# ─────────────────────────────────────────────

def run_interactive_loop(advisor: BakkenAdvisor):
    """
    Launch an interactive chat loop suitable for Jupyter notebooks
    or terminal use. Type 'quit', 'exit', or 'q' to stop.
    Type 'reset' to clear history.
    """
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       🛢️  Bakken Basin Production Advisor                    ║
║       Powered by ML + LLM                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Describe a horizontal well's completion parameters and I    ║
║  will predict its first-year cumulative oil production.      ║
║                                                              ║
║  Example:                                                    ║
║  "10,500 ft lateral, 12MM lbs proppant, Middle Bakken,       ║
║   50,000 BBL fluid, Williams County ND"                      ║
║                                                              ║
║  Commands: 'reset' to clear history | 'quit' to exit        ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            advisor.reset()
            continue

        print("\n🤖 Advisor: ", end="", flush=True)
        try:
            response = advisor.chat(user_input)
            print(response)
        except RuntimeError as e:
            print(f"\n⚠️  Error: {e}")
        except Exception as e:
            logger.exception("Unexpected error during chat")
            print(f"\n⚠️  Unexpected error: {e}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bakken Basin LLM-powered production advisor"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run ID of the best trained model",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    advisor = BakkenAdvisor(run_id=args.run_id, config_path=args.config)
    run_interactive_loop(advisor)
