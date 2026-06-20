"""
test_interface.py
-----------------
Tests for the LLM-powered interface in src/app.py.

Covers the 2 required categories:
  1. Input parsing — LLM extracts correct feature values from natural language
  2. Edge case handling — incomplete, ambiguous, and off-topic inputs

All LLM calls are mocked so these tests run without an API key.

Run with:
    pytest tests/test_interface.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app import (
    REQUIRED_FEATURES,
    BakkenAdvisor,
    build_feature_vector,
    generate_response,
    parse_features_from_text,
)


# ─────────────────────────────────────────────
# Helpers — mock LLM that returns controlled JSON
# ─────────────────────────────────────────────

def make_mock_llm(return_json: dict) -> MagicMock:
    """Return a mock LLMClient whose .complete() returns the given dict as JSON."""
    mock = MagicMock()
    mock.complete.return_value = json.dumps(return_json)
    return mock


def make_mock_llm_text(text: str) -> MagicMock:
    """Return a mock LLMClient whose .complete() returns raw text."""
    mock = MagicMock()
    mock.complete.return_value = text
    return mock


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def full_feature_parse():
    """Expected parse output for a well-specified query."""
    return {
        "DI Lateral Length":         10500.0,
        "Total Proppant":            12000000.0,
        "Total Fluid":               50000.0,
        "Gross Perforated Interval": 9800.0,
        "Reservoir":                 "MIDDLE BAKKEN",
        "DI Landing Zone":           "BAKKEN",
    }


@pytest.fixture
def minimal_feature_parse():
    """Only the required features — optionals absent."""
    return {
        "DI Lateral Length": 9000.0,
        "Total Proppant":    8000000.0,
        "Reservoir":         "THREE FORKS",
    }


@pytest.fixture
def simple_encoders():
    """Minimal encoders that handle MIDDLE BAKKEN, THREE FORKS, UNKNOWN."""
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col, classes in {
        "Reservoir":      ["MIDDLE BAKKEN", "THREE FORKS", "UNKNOWN"],
        "DI Landing Zone": ["BAKKEN", "THREE FORKS", "UNKNOWN"],
        "Production Type": ["O&G", "OIL", "UNKNOWN"],
        "Well Status":     ["ACTIVE", "P & A", "UNKNOWN"],
    }.items():
        le = LabelEncoder()
        le.fit(classes)
        encoders[col] = le

    return encoders


@pytest.fixture
def simple_scaler():
    """StandardScaler fitted on dummy data — just needs to transform without error."""
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    scaler = StandardScaler()
    # Fit on 10 rows of 14 numeric features (matching NUMERIC_FEATURE_COLS count)
    scaler.fit(np.random.default_rng(0).uniform(0, 1, (10, 14)))
    return scaler


@pytest.fixture
def simple_cfg():
    """Minimal config dict that mirrors configs/config.yaml structure."""
    from app import FEATURE_SCHEMA
    from preprocess import NUMERIC_FEATURE_COLS, CATEGORICAL_COLS

    return {
        "data": {
            "log_transform_target": True,
            "encoders_path": "data/processed/encoders.pkl",
            "scaler_path":   "data/processed/scaler.pkl",
        },
        "features": {
            "numeric":      NUMERIC_FEATURE_COLS,
            "categorical":  CATEGORICAL_COLS,
        },
        "mlflow": {
            "tracking_uri": "mlruns",
        },
    }


# ─────────────────────────────────────────────
# Test 1: Input parsing accuracy
# ─────────────────────────────────────────────

class TestInputParsing:

    def test_lateral_length_extracted_correctly(self):
        """Parser must extract lateral length in feet."""
        llm = make_mock_llm({"DI Lateral Length": 10500, "Total Proppant": 12000000,
                              "Reservoir": "MIDDLE BAKKEN"})
        result = parse_features_from_text(
            "I have a 10,500 ft lateral with 12MM lbs proppant in the Middle Bakken",
            llm
        )
        assert result["DI Lateral Length"] == pytest.approx(10500.0)

    def test_proppant_converted_to_pounds(self):
        """Proppant expressed as 'MM lbs' must arrive as raw pounds."""
        llm = make_mock_llm({"DI Lateral Length": 9000, "Total Proppant": 10000000,
                              "Reservoir": "MIDDLE BAKKEN"})
        result = parse_features_from_text(
            "10MM lbs proppant, 9000 ft lateral, Middle Bakken",
            llm
        )
        assert result["Total Proppant"] == pytest.approx(10_000_000.0)

    def test_reservoir_normalised_to_uppercase(self):
        """Reservoir string must be stored in uppercase regardless of input case."""
        llm = make_mock_llm({"DI Lateral Length": 9500, "Total Proppant": 9000000,
                              "Reservoir": "middle bakken"})
        result = parse_features_from_text("9500 ft, Middle Bakken, 9MM lbs", llm)
        assert result["Reservoir"] == "MIDDLE BAKKEN"

    def test_all_optional_features_captured(self, full_feature_parse):
        """When all features are present, all should be in parsed output."""
        llm = make_mock_llm(full_feature_parse)
        result = parse_features_from_text("full well description", llm)
        for key in full_feature_parse:
            assert key in result, f"Expected key '{key}' missing from parsed output"

    def test_numeric_values_are_float(self, full_feature_parse):
        """All numeric feature values must be Python floats after parsing."""
        llm = make_mock_llm(full_feature_parse)
        result = parse_features_from_text("full well description", llm)
        numeric_keys = [k for k in full_feature_parse
                        if k not in ("Reservoir", "DI Landing Zone")]
        for key in numeric_keys:
            if key in result:
                assert isinstance(result[key], float), \
                    f"Expected float for '{key}', got {type(result[key])}"

    def test_parse_ignores_irrelevant_json_keys(self):
        """LLM returning extra keys should not crash the parser."""
        llm = make_mock_llm({
            "DI Lateral Length": 9500,
            "Total Proppant": 8000000,
            "Reservoir": "THREE FORKS",
            "some_random_key": "ignore me",
        })
        result = parse_features_from_text("9500 ft Three Forks 8MM lbs", llm)
        # Should not raise; known keys captured, unknown keys harmless
        assert "DI Lateral Length" in result

    def test_llm_called_once_per_parse(self):
        """Parser should make exactly one LLM call per user message."""
        llm = make_mock_llm({"DI Lateral Length": 9500, "Total Proppant": 8000000,
                              "Reservoir": "MIDDLE BAKKEN"})
        parse_features_from_text("9500 ft 8MM lbs Middle Bakken", llm)
        assert llm.complete.call_count == 1


# ─────────────────────────────────────────────
# Test 2: Edge case handling
# ─────────────────────────────────────────────

class TestEdgeCases:

    def test_off_topic_query_returns_clarification(self):
        """A message with no well parameters must trigger a clarification response."""
        llm_parse = make_mock_llm({"off_topic": True})

        mock_advisor = MagicMock()
        mock_advisor.llm = llm_parse

        result = parse_features_from_text("What is the weather like in Williston?", llm_parse)
        assert result.get("__off_topic__") is True

    def test_greeting_returns_greeting_flag(self):
        """A greeting message must return the __greeting__ flag."""
        llm = make_mock_llm({"greeting": True})
        result = parse_features_from_text("Hello!", llm)
        assert result.get("__greeting__") is True

    def test_missing_required_feature_detected(self, minimal_feature_parse):
        """
        If lateral length is missing from parsed output,
        it must appear in the missing-features list.
        """
        incomplete = {k: v for k, v in minimal_feature_parse.items()
                      if k != "DI Lateral Length"}
        missing = [f for f in REQUIRED_FEATURES if f not in incomplete]
        assert "DI Lateral Length" in missing

    def test_all_required_features_present_no_missing(self, minimal_feature_parse):
        """A parse result containing all required features must yield zero missing."""
        missing = [f for f in REQUIRED_FEATURES if f not in minimal_feature_parse]
        assert missing == [], f"Unexpected missing features: {missing}"

    def test_malformed_json_from_llm_returns_parse_error(self):
        """If the LLM returns garbage text, parser must return __parse_error__ flag."""
        llm = make_mock_llm_text("I'm sorry, I don't understand what you mean here.")
        result = parse_features_from_text("some well query", llm)
        assert result.get("__parse_error__") is True

    def test_build_feature_vector_returns_none_for_missing_required(
        self, simple_encoders, simple_scaler, simple_cfg
    ):
        """build_feature_vector must return (None, missing_list) when required features absent."""
        incomplete_parse = {"Total Proppant": 8000000.0}  # missing lateral + reservoir
        feature_cols = (
            simple_cfg["features"]["numeric"] + simple_cfg["features"]["categorical"]
        )
        X, missing = build_feature_vector(
            parsed=incomplete_parse,
            feature_cols=feature_cols,
            encoders=simple_encoders,
            scaler=simple_scaler,
            cfg=simple_cfg,
        )
        assert X is None
        assert len(missing) > 0

    def test_build_feature_vector_shape_with_full_input(
        self, full_feature_parse, simple_encoders, simple_scaler, simple_cfg
    ):
        """With all required features, build_feature_vector must return a (1, n) array."""
        feature_cols = (
            simple_cfg["features"]["numeric"] + simple_cfg["features"]["categorical"]
        )
        X, missing = build_feature_vector(
            parsed=full_feature_parse,
            feature_cols=feature_cols,
            encoders=simple_encoders,
            scaler=simple_scaler,
            cfg=simple_cfg,
        )
        assert X is not None, f"Expected feature array, got None (missing: {missing})"
        assert X.shape == (1, len(feature_cols)), \
            f"Expected shape (1, {len(feature_cols)}), got {X.shape}"

    def test_unseen_reservoir_handled_without_crash(
        self, simple_encoders, simple_scaler, simple_cfg
    ):
        """An unseen reservoir name at inference time must not raise an exception."""
        parse_with_unknown = {
            "DI Lateral Length": 9500.0,
            "Total Proppant":    10000000.0,
            "Reservoir":         "COMPLETELY_NEW_FORMATION_XYZ",
        }
        feature_cols = (
            simple_cfg["features"]["numeric"] + simple_cfg["features"]["categorical"]
        )
        try:
            X, _ = build_feature_vector(
                parsed=parse_with_unknown,
                feature_cols=feature_cols,
                encoders=simple_encoders,
                scaler=simple_scaler,
                cfg=simple_cfg,
            )
        except Exception as e:
            pytest.fail(f"Unseen reservoir caused a crash: {e}")

    def test_generate_response_calls_llm(self, full_feature_parse):
        """generate_response must call the LLM exactly once."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.2, 0.15, 0.1, 0.1, 0.1,
                                                     0.1, 0.05, 0.05, 0.05, 0.03,
                                                     0.03, 0.02, 0.02])
        llm = make_mock_llm_text(
            "Based on the parameters provided, the model estimates 75,000 BBL."
        )
        feature_cols = list(full_feature_parse.keys())
        # Pad importances to match feature_cols length
        mock_model.feature_importances_ = np.ones(len(feature_cols)) / len(feature_cols)

        generate_response(
            prediction_bbl=75000.0,
            parsed_features=full_feature_parse,
            feature_cols=feature_cols,
            model=mock_model,
            llm=llm,
        )
        assert llm.complete.call_count == 1
