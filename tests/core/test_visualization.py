"""Tests for visualization module."""

import numpy as np
import pandas as pd

from nozzle_clogging.convergence import run_convergence_study
from nozzle_clogging.sensitivity import calculate_correlation_sensitivity
from nozzle_clogging.simulation import run_simulation
from nozzle_clogging.visualization import (
    generate_results_table,
    generate_risk_proportions_table,
    prepare_cdf_data,
    prepare_convergence_plot_data,
    prepare_risk_distribution_data,
    prepare_sensitivity_tornado_data,
)


class TestGenerateResultsTable:
    def test_returns_dataframe(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_results_table(df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_results_table(df)
        expected_cols = [
            'output',
            'mean',
            'std',
            'ci_lower',
            'ci_upper',
            'min',
            'max',
            'median',
        ]
        assert list(result.columns) == expected_cols

    def test_ci_lower_less_than_mean(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_results_table(df)
        assert all(result['ci_lower'] < result['mean'])

    def test_ci_upper_greater_than_mean(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_results_table(df)
        assert all(result['ci_upper'] > result['mean'])

    def test_std_non_negative(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_results_table(df)
        assert all(result['std'] >= 0)


class TestGenerateRiskProportionsTable:
    def test_returns_dataframe(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_risk_proportions_table(df)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_risk_proportions_table(df)
        expected_cols = ['risk', 'count', 'proportion', 'ci_lower', 'ci_upper']
        assert list(result.columns) == expected_cols

    def test_proportion_sums_to_one(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_risk_proportions_table(df)
        total_proportion = result['proportion'].sum()
        np.testing.assert_allclose(total_proportion, 1.0, rtol=1e-5)

    def test_count_sums_to_total(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_risk_proportions_table(df)
        total_count = result['count'].sum()
        assert total_count == len(df)

    def test_proportion_in_range(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_risk_proportions_table(df)
        assert all(result['proportion'] >= 0)
        assert all(result['proportion'] <= 1)

    def test_ci_in_range(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = generate_risk_proportions_table(df)
        assert all(result['ci_lower'] >= 0)
        assert all(result['ci_upper'] <= 1)


class TestPrepareCdfData:
    def test_returns_dict(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_cdf_data(df, output='clogging_probability')
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_cdf_data(df, output='clogging_probability')
        assert 'values' in result
        assert 'probabilities' in result
        assert 'output' in result

    def test_values_sorted(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_cdf_data(df, output='clogging_probability')
        values = result['values']
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_probabilities_in_range(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_cdf_data(df, output='clogging_probability')
        probabilities = result['probabilities']
        assert all(p >= 0 for p in probabilities)
        assert all(p <= 1 for p in probabilities)

    def test_n_points_respected(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_cdf_data(df, output='clogging_probability', n_points=50)
        assert len(result['values']) <= 50


class TestPrepareRiskDistributionData:
    def test_returns_dict(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_risk_distribution_data(df)
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_risk_distribution_data(df)
        assert 'categories' in result
        assert 'counts' in result
        assert 'proportions' in result

    def test_three_categories(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_risk_distribution_data(df)
        assert len(result['categories']) == 3
        assert 'Low' in result['categories']
        assert 'Moderate' in result['categories']
        assert 'High' in result['categories']

    def test_proportions_sum_to_one(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        result = prepare_risk_distribution_data(df)
        total_proportion = sum(result['proportions'])
        np.testing.assert_allclose(total_proportion, 1.0, rtol=1e-5)


class TestPrepareConvergencePlotData:
    def test_returns_dict(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = prepare_convergence_plot_data(conv_df, output='clogging_probability')
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        conv_df = run_convergence_study(
            sample_sizes=[100, 200],
            outputs=['clogging_probability'],
        )
        result = prepare_convergence_plot_data(conv_df, output='clogging_probability')
        assert 'sample_sizes' in result
        assert 'means' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert 'output' in result

    def test_correct_length(self) -> None:
        sample_sizes = [100, 200, 300]
        conv_df = run_convergence_study(
            sample_sizes=sample_sizes,
            outputs=['clogging_probability'],
        )
        result = prepare_convergence_plot_data(conv_df, output='clogging_probability')
        assert len(result['sample_sizes']) == len(sample_sizes)


class TestPrepareSensitivityTornadoData:
    def test_returns_dict(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        corr = calculate_correlation_sensitivity(df)
        result = prepare_sensitivity_tornado_data(
            corr, output='clogging_probability', method='spearman'
        )
        assert isinstance(result, dict)

    def test_has_required_keys(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        corr = calculate_correlation_sensitivity(df)
        result = prepare_sensitivity_tornado_data(
            corr, output='clogging_probability', method='spearman'
        )
        assert 'inputs' in result
        assert 'correlations' in result
        assert 'p_values' in result
        assert 'output' in result
        assert 'method' in result

    def test_sorted_by_absolute_correlation(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=100, seed=42)
        corr = calculate_correlation_sensitivity(df)
        result = prepare_sensitivity_tornado_data(
            corr, output='clogging_probability', method='spearman'
        )
        correlations = result['correlations']
        abs_correlations = [abs(c) for c in correlations]
        assert all(
            abs_correlations[i] >= abs_correlations[i + 1]
            for i in range(len(abs_correlations) - 1)
        )
