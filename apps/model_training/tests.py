"""Unit tests for the extracted service layer.

These exercise the pure algorithm/metrics functions directly, without the
Django test client and without any database. The expected values come from a
behaviour snapshot taken on the pre-extraction views.py, so they pin the
exact pre-refactor behaviour.
"""
import numpy as np
import pandas as pd

from django.test import SimpleTestCase

from apps.model_training.services.algorithms import (
    _compute_trend_forecast,
    _croston_predict,
    _get_fallback_model_types,
    _get_retry_params,
    _sba_predict,
    _seasonal_decompose,
)
from apps.model_training.services.constants import VALID_MODEL_TYPES
from apps.model_training.services.metrics import _eval_metrics
from apps.model_training.services.trainer import _preprocess_time_series


class MetricsTests(SimpleTestCase):
    def test_known_values(self):
        actual = np.array([10, 12, 9, 15, 0, 8])
        forecast = np.array([11, 10, 9, 14, 1, 7])
        mae, rmse, accuracy = _eval_metrics(actual, forecast)
        self.assertAlmostEqual(mae, 1.0)
        self.assertAlmostEqual(rmse, 1.1547005384, places=8)
        self.assertAlmostEqual(accuracy, 75.6944444444, places=8)

    def test_zero_actuals_do_not_explode(self):
        mae, rmse, accuracy = _eval_metrics(np.array([0.0, 0, 0]), np.array([1, 2, 3]))
        self.assertAlmostEqual(mae, 2.0)
        self.assertAlmostEqual(accuracy, 0.0)

    def test_perfect_forecast(self):
        mae, rmse, accuracy = _eval_metrics(np.array([5, 5]), np.array([5, 5]))
        self.assertEqual((mae, rmse, accuracy), (0.0, 0.0, 100.0))


class IntermittentDemandTests(SimpleTestCase):
    """Croston and SBA on an intermittent series (values from the snapshot)."""

    SERIES = np.array([5, 0, 3, 0, 0, 8, 0, 2, 0, 6], dtype=float)

    def test_croston_matches_snapshot(self):
        forecast = _croston_predict(self.SERIES, periods=4)
        self.assertEqual(len(forecast), 4)
        for value in forecast:
            self.assertAlmostEqual(value, 2.2913704014, places=8)

    def test_sba_matches_snapshot(self):
        forecast = _sba_predict(self.SERIES, periods=4)
        self.assertEqual(len(forecast), 4)
        for value in forecast:
            self.assertAlmostEqual(value, 2.1195176213, places=8)

    def test_forecasts_are_deterministic_and_finite(self):
        again = _croston_predict(self.SERIES, periods=6)
        self.assertTrue(np.all(np.isfinite(again)))
        np.testing.assert_allclose(again, _croston_predict(self.SERIES, periods=6))


class TrendForecastTests(SimpleTestCase):
    def test_matches_snapshot(self):
        values = np.array([10, 12, 11, 13, 15, 14, 16, 18, 17, 19, 21, 20], dtype=float)
        forecast = _compute_trend_forecast(values, horizon=5)
        np.testing.assert_allclose(
            forecast,
            [21.8181818182, 22.7902097902, 23.7622377622, 24.7342657343, 25.7062937063],
            rtol=1e-9)


class SeasonalDecomposeTests(SimpleTestCase):
    def test_returns_expected_keys(self):
        idx = pd.date_range('2022-01-01', periods=24, freq='MS')
        series = pd.Series(50 + 10 * np.sin(np.arange(24) * np.pi / 6)
                           + np.arange(24) * 0.5, index=idx)
        result = _seasonal_decompose(series)
        self.assertEqual(
            sorted(result.keys()),
            ['deseasonalized', 'seasonal_index', 'seasonal_strength', 'trend'])
        self.assertTrue(0.0 <= result['seasonal_strength'] <= 1.0)


class PreprocessTests(SimpleTestCase):
    def test_cleans_nulls_and_flags_outliers(self):
        idx = pd.date_range('2022-01-01', periods=24, freq='MS')
        dirty = pd.Series(
            [100, 120, np.nan, -5, 5000, 110, 90, 0, 115, 130, 95, 105,
             108, 112, 98, np.nan, 125, 118, 102, 97, 111, 119, 104, 99],
            index=idx)
        cleaned, log = _preprocess_time_series(dirty, region='R1', sku='S1')
        self.assertFalse(cleaned.isna().any())
        self.assertEqual(log['original_length'], 24)
        self.assertEqual(log['null_count'], 2)
        self.assertEqual(log['outlier_count'], 1)
        # The 5000 outlier is replaced by a boundary value, matching the
        # pre-extraction behaviour (boundary [54, 162]).
        self.assertLessEqual(float(cleaned.iloc[4]), 162)


class ConfigTableTests(SimpleTestCase):
    def test_retry_params_grid(self):
        self.assertEqual(_get_retry_params('arima', 0), {'arima_order': (1, 1, 1)})
        self.assertEqual(_get_retry_params('arima', 1), {'arima_order': (0, 1, 1)})
        self.assertEqual(_get_retry_params('arima', 2), {'arima_order': (2, 1, 0)})
        self.assertEqual(
            _get_retry_params('lstm', 0), {'epochs': 60, 'lookback': 6})
        self.assertEqual(_get_retry_params('random-forest', 1), {})

    def test_fallback_chain(self):
        self.assertEqual(
            _get_fallback_model_types('prophet'),
            ['holt-winters', 'moving_avg', 'seasonal_naive'])
        self.assertEqual(
            _get_fallback_model_types('croston'),
            ['moving_avg', 'weighted_avg', 'seasonal_naive'])

    def test_valid_model_types_contains_the_always_available_base(self):
        for base in ('holt-winters', 'moving_avg', 'weighted_avg', 'seasonal_naive',
                     'croston', 'sba'):
            self.assertIn(base, VALID_MODEL_TYPES)
