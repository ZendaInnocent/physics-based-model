"""Tests for convergence analysis module."""

import pandas as pd

from nozzle_clogging.convergence import (
    calculate_batch_variance,
    calculate_convergence_metrics,
    detect_convergence,
    run_convergence_study,
    run_multiple_seeds,
)
from nozzle_clogging.simulation import run_simulation


class TestRunConvergenceStudy:
    def test_returns_dataframe(self) -> None:
        result = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        result = run_convergence_study(
            sample_sizes=[100],
            outputs=['clogging_probability'],
        )
        expected_cols = [
            'sample_size',
            'seed',
            'output',
            'mean',
            'std',
            'ci_lower',
            'ci_upper',
            'ci_width',
        ]
        assert list(result.columns) == expected_cols

    def test_correct_row_count(self) -> None:
        sample_sizes = [100, 200, 300]
        outputs = ['clogging_probability', 'X']
        result = run_convergence_study(
            sample_sizes=sample_sizes,
            outputs=outputs,
        )
        expected_rows = len(sample_sizes) * len(outputs)
        assert len(result) == expected_rows

    def test_ci_lower_less_than_mean(self) -> None:
        result = run_convergence_study(
            sample_sizes=[100],
            outputs=['clogging_probability'],
        )
        assert all(result['ci_lower'] < result['mean'])

    def test_ci_upper_greater_than_mean(self) -> None:
        result = run_convergence_study(
            sample_sizes=[100],
            outputs=['clogging_probability'],
        )
        assert all(result['ci_upper'] > result['mean'])

    def test_ci_width_positive(self) -> None:
        result = run_convergence_study(
            sample_sizes=[100],
            outputs=['clogging_probability'],
        )
        assert all(result['ci_width'] > 0)

    def test_mean_in_valid_range(self) -> None:
        result = run_convergence_study(
            sample_sizes=[100],
            outputs=['clogging_probability'],
        )
        assert all(result['mean'] >= 0)
        assert all(result['mean'] <= 1)


class TestCalculateConvergenceMetrics:
    def test_returns_dataframe(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = calculate_convergence_metrics(conv_df, reference_size=200)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = calculate_convergence_metrics(conv_df, reference_size=200)
        expected_cols = [
            'sample_size',
            'output',
            'mean',
            'mean_change',
            'ci_width',
            'ci_width_change',
            'converged',
        ]
        assert list(result.columns) == expected_cols

    def test_converged_is_boolean(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = calculate_convergence_metrics(conv_df, reference_size=200)
        assert result['converged'].dtype == bool

    def test_mean_change_non_negative(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = calculate_convergence_metrics(conv_df, reference_size=200)
        assert all(result['mean_change'] >= 0)

    def test_ci_width_change_non_negative(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = calculate_convergence_metrics(conv_df, reference_size=200)
        assert all(result['ci_width_change'] >= 0)


class TestDetectConvergence:
    def test_returns_dict(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = detect_convergence(conv_df)
        assert isinstance(result, dict)

    def test_keys_are_outputs(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability', 'X'],
        )
        result = detect_convergence(conv_df)
        assert 'clogging_probability' in result
        assert 'X' in result

    def test_values_are_int(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = detect_convergence(conv_df)
        assert isinstance(result['clogging_probability'], int)

    def test_values_are_sample_sizes(self) -> None:
        sample_sizes = [100, 200, 300]
        conv_df = run_convergence_study(
            sample_sizes=sample_sizes,
            outputs=['clogging_probability'],
        )
        result = detect_convergence(conv_df)
        assert result['clogging_probability'] in sample_sizes


class TestRunMultipleSeeds:
    def test_returns_dataframe(self) -> None:
        result = run_multiple_seeds(
            sample_size=100,
            n_runs=2,
            outputs=['clogging_probability'],
        )
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        result = run_multiple_seeds(
            sample_size=100,
            n_runs=2,
            outputs=['clogging_probability'],
        )
        expected_cols = ['run', 'seed', 'output', 'mean', 'std', 'ci_lower', 'ci_upper']
        assert list(result.columns) == expected_cols

    def test_correct_row_count(self) -> None:
        n_runs = 3
        outputs = ['clogging_probability', 'X']
        result = run_multiple_seeds(
            sample_size=100,
            n_runs=n_runs,
            outputs=outputs,
        )
        expected_rows = n_runs * len(outputs)
        assert len(result) == expected_rows

    def test_different_seeds(self) -> None:
        result = run_multiple_seeds(
            sample_size=100,
            n_runs=3,
            outputs=['clogging_probability'],
        )
        seeds = result['seed'].unique()
        assert len(seeds) == 3


class TestCalculateBatchVariance:
    def test_returns_dataframe(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=100, seed=42)
        result = calculate_batch_variance(df, batch_size=100)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=100, seed=42)
        result = calculate_batch_variance(df, batch_size=100)
        expected_cols = ['output', 'between_batch_std', 'between_batch_cv', 'n_batches']
        assert list(result.columns) == expected_cols

    def test_between_batch_std_non_negative(self) -> None:
        df = run_simulation(total_samples=200, chunk_size=100, seed=42)
        result = calculate_batch_variance(df, batch_size=100)
        assert all(result['between_batch_std'] >= 0)
