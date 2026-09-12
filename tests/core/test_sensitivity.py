"""Tests for sensitivity analysis module."""

import numpy as np
import pandas as pd

from nozzle_clogging.sensitivity import (
    calculate_correlation_sensitivity,
    calculate_morris_indices,
    calculate_sobol_indices,
    run_sobol_convergence_test,
)
from nozzle_clogging.simulation import run_simulation


class TestCalculateCorrelationSensitivity:
    def test_returns_dataframe(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=200, seed=42)
        result = calculate_correlation_sensitivity(df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=200, seed=42)
        result = calculate_correlation_sensitivity(df)
        expected_cols = [
            'input',
            'output',
            'pearson_r',
            'pearson_p',
            'spearman_r',
            'spearman_p',
        ]
        assert list(result.columns) == expected_cols

    def test_correlation_in_range(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=200, seed=42)
        result = calculate_correlation_sensitivity(df)
        assert all(result['pearson_r'] >= -1)
        assert all(result['pearson_r'] <= 1)
        assert all(result['spearman_r'] >= -1)
        assert all(result['spearman_r'] <= 1)

    def test_p_value_in_range(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=200, seed=42)
        result = calculate_correlation_sensitivity(df)
        assert all(result['pearson_p'] >= 0)
        assert all(result['pearson_p'] <= 1)
        assert all(result['spearman_p'] >= 0)
        assert all(result['spearman_p'] <= 1)

    def test_correct_input_count(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=200, seed=42)
        result = calculate_correlation_sensitivity(df)
        unique_inputs = result['input'].nunique()
        assert unique_inputs == 5

    def test_correct_output_count(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=200, seed=42)
        result = calculate_correlation_sensitivity(df)
        unique_outputs = result['output'].nunique()
        assert unique_outputs == 3


class TestCalculateSobolIndices:
    def test_returns_dataframe(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        expected_cols = ['input', 'S1', 'S1_lo', 'S1_hi', 'ST', 'ST_lo', 'ST_hi']
        assert list(result.columns) == expected_cols

    def test_correct_row_count(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        assert len(result) == 5

    def test_s1_in_range(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        # S1 can be negative due to sampling noise
        assert all(result['S1'] >= -1)
        assert all(result['S1'] <= 2)

    def test_st_non_negative(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        assert all(result['ST'] >= 0)

    def test_confidence_intervals_non_negative(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        assert all(result['S1_lo'] >= 0)
        assert all(result['S1_hi'] >= 0)
        assert all(result['ST_lo'] >= 0)
        assert all(result['ST_hi'] >= 0)

    def test_confidence_intervals_ordered(self) -> None:
        result = calculate_sobol_indices(
            n_samples=50,
            seed=42,
            output_col='clogging_probability',
        )
        assert all(result['S1_lo'] <= result['S1_hi'])
        assert all(result['ST_lo'] <= result['ST_hi'])


class TestCalculateMorrisIndices:
    def test_returns_dataframe(self) -> None:
        result = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        result = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        expected_cols = ['input', 'mu', 'mu_star', 'sigma']
        assert list(result.columns) == expected_cols

    def test_correct_row_count(self) -> None:
        result = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        assert len(result) == 5

    def test_mu_star_non_negative(self) -> None:
        result = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        assert all(result['mu_star'] >= 0)

    def test_sigma_non_negative(self) -> None:
        result = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        assert all(result['sigma'] >= 0)

    def test_deterministic_with_seed(self) -> None:
        result1 = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        result2 = calculate_morris_indices(
            n_samples=30,
            seed=42,
            output_col='clogging_probability',
        )
        np.testing.assert_allclose(result1['mu'].values, result2['mu'].values)
        np.testing.assert_allclose(result1['mu_star'].values, result2['mu_star'].values)


class TestRunSobolConvergenceTest:
    def test_returns_dataframe(self) -> None:
        result = run_sobol_convergence_test(
            sample_sizes=[50, 100],
            seed=42,
            output_col='clogging_probability',
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self) -> None:
        result = run_sobol_convergence_test(
            sample_sizes=[50, 100],
            seed=42,
            output_col='clogging_probability',
        )
        assert len(result) == 10  # 2 sizes * 5 params

    def test_correct_columns(self) -> None:
        result = run_sobol_convergence_test(
            sample_sizes=[50, 100],
            seed=42,
            output_col='clogging_probability',
        )
        expected_cols = [
            'sample_size',
            'input',
            'S1',
            'ST',
            'S1_rel_change',
            'ST_rel_change',
            'converged',
        ]
        assert list(result.columns) == expected_cols

    def test_first_size_has_nan_rel_change(self) -> None:
        result = run_sobol_convergence_test(
            sample_sizes=[50, 100],
            seed=42,
            output_col='clogging_probability',
        )
        first_size = result[result['sample_size'] == 50]
        assert all(np.isnan(first_size['S1_rel_change']))
        assert all(np.isnan(first_size['ST_rel_change']))
