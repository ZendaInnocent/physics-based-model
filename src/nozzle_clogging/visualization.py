"""Results visualization for Monte Carlo simulation.

Provides functions to generate data structures and plots for
convergence analysis, sensitivity analysis, and risk classification.
"""

import numpy as np
import pandas as pd
from beartype.typing import Any

from nozzle_clogging import config


def prepare_convergence_plot_data(
    convergence_df: pd.DataFrame,
    output: str = 'clogging_probability',
) -> dict[str, Any]:
    """Prepare data for convergence plot.

    Args:
        convergence_df: Output from convergence.run_convergence_study.
        output: Output column to plot.

    Returns:
        Dictionary with plot data:
        - sample_sizes: List of sample sizes
        - means: List of mean values
        - ci_lower: List of lower CI values
        - ci_upper: List of upper CI values
    """
    output_df = convergence_df[convergence_df['output'] == output].sort_values(
        'sample_size'
    )

    return {
        'sample_sizes': output_df['sample_size'].tolist(),
        'means': output_df['mean'].tolist(),
        'ci_lower': output_df['ci_lower'].tolist(),
        'ci_upper': output_df['ci_upper'].tolist(),
        'output': output,
    }


def prepare_sensitivity_tornado_data(
    correlation_df: pd.DataFrame,
    output: str = 'clogging_probability',
    method: str = 'spearman',
) -> dict[str, Any]:
    """Prepare data for tornado plot.

    Args:
        correlation_df: Output from sensitivity.calculate_correlation_sensitivity.
        output: Output column to plot.
        method: Correlation method ('pearson' or 'spearman').

    Returns:
        Dictionary with plot data:
        - inputs: List of input parameter names
        - correlations: List of correlation values
        - p_values: List of p-values
    """
    if method == 'pearson':
        corr_col = 'pearson_r'
        p_col = 'pearson_p'
    else:
        corr_col = 'spearman_r'
        p_col = 'spearman_p'

    output_df = correlation_df[correlation_df['output'] == output].sort_values(
        corr_col, key=abs, ascending=False
    )

    return {
        'inputs': output_df['input'].tolist(),
        'correlations': output_df[corr_col].tolist(),
        'p_values': output_df[p_col].tolist(),
        'output': output,
        'method': method,
    }


def prepare_cdf_data(
    df: pd.DataFrame,
    output: str = 'clogging_probability',
    n_points: int = 100,
) -> dict[str, Any]:
    """Prepare data for CDF plot.

    Args:
        df: Simulation results DataFrame.
        output: Output column to plot.
        n_points: Number of points for CDF.

    Returns:
        Dictionary with plot data:
        - values: Sorted output values
        - probabilities: Corresponding CDF probabilities
    """
    values = df[output].pint.magnitude.values
    sorted_values = np.sort(values)
    probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)

    # Downsample if needed
    if len(sorted_values) > n_points:
        indices = np.linspace(0, len(sorted_values) - 1, n_points, dtype=int)
        sorted_values = sorted_values[indices]
        probabilities = probabilities[indices]

    return {
        'values': sorted_values.tolist(),
        'probabilities': probabilities.tolist(),
        'output': output,
    }


def prepare_risk_distribution_data(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Prepare data for risk distribution pie/bar chart.

    Args:
        df: Simulation results DataFrame.

    Returns:
        Dictionary with plot data:
        - categories: Risk category names
        - counts: Count for each category
        - proportions: Proportion for each category
    """
    risk_counts = df['clogging_risk'].value_counts()

    categories = []
    counts = []
    proportions = []

    for level in config.RISK_LEVELS:
        if level in risk_counts.index:
            count = risk_counts[level]
            proportion = count / len(df)
        else:
            count = 0
            proportion = 0.0

        categories.append(level)
        counts.append(count)
        proportions.append(proportion)

    return {
        'categories': categories,
        'counts': counts,
        'proportions': proportions,
    }


def generate_results_table(
    df: pd.DataFrame,
    outputs: list[str] | None = None,
) -> pd.DataFrame:
    """Generate summary statistics table.

    Args:
        df: Simulation results DataFrame.
        outputs: List of output columns to summarize.
            Defaults to ['clogging_probability', 'X', 'volume_fraction'].

    Returns:
        DataFrame with columns:
        - output: Output variable name
        - mean: Mean value
        - std: Standard deviation
        - ci_lower: Lower 95% confidence interval
        - ci_upper: Upper 95% confidence interval
        - min: Minimum value
        - max: Maximum value
        - median: Median value
    """
    if outputs is None:
        outputs = ['clogging_probability', 'X', 'volume_fraction']

    results = []

    for output in outputs:
        if output in df.columns:
            values = df[output].pint.magnitude

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            se = std_val / np.sqrt(len(values))
            ci_lower = mean_val - 1.96 * se
            ci_upper = mean_val + 1.96 * se

            results.append(
                {
                    'output': output,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                }
            )

    return pd.DataFrame(results)


def generate_risk_proportions_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate risk proportions table with confidence intervals.

    Args:
        df: Simulation results DataFrame.

    Returns:
        DataFrame with columns:
        - risk: Risk category
        - count: Count
        - proportion: Proportion
        - ci_lower: Lower 95% confidence interval
        - ci_upper: Upper 95% confidence interval
    """
    risk_counts = df['clogging_risk'].value_counts()
    n_total = len(df)

    results = []

    for level in config.RISK_LEVELS:
        if level in risk_counts.index:
            count = risk_counts[level]
            proportion = count / n_total

            # Wilson score interval for binomial proportion
            z = 1.96
            denominator = 1 + z**2 / n_total
            centre_adjusted_probability = proportion + z**2 / (2 * n_total)
            adjusted_standard_deviation = np.sqrt(
                (proportion * (1 - proportion) + z**2 / (4 * n_total)) / n_total
            )

            ci_lower = (
                centre_adjusted_probability - z * adjusted_standard_deviation
            ) / denominator
            ci_upper = (
                centre_adjusted_probability + z * adjusted_standard_deviation
            ) / denominator
        else:
            count = 0
            proportion = 0.0
            ci_lower = 0.0
            ci_upper = 0.0

        results.append(
            {
                'risk': level,
                'count': count,
                'proportion': proportion,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }
        )

    return pd.DataFrame(results)


def generate_sensitivity_table(
    correlation_df: pd.DataFrame,
    sobol_df: pd.DataFrame | None = None,
    morris_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate combined sensitivity analysis table.

    Args:
        correlation_df: Output from sensitivity.calculate_correlation_sensitivity.
        sobol_df: Optional output from sensitivity.calculate_sobol_indices.
        morris_df: Optional output from sensitivity.calculate_morris_indices.

    Returns:
        DataFrame with columns:
        - input: Input parameter name
        - spearman_r: Spearman correlation
        - spearman_p: Spearman p-value
        - pearson_r: Pearson correlation (if available)
        - S1: First-order Sobol index (if available)
        - ST: Total-order Sobol index (if available)
        - mu_star: Morris mu_star (if available)
    """
    # Start with correlation data
    result = correlation_df[
        ['input', 'output', 'spearman_r', 'spearman_p', 'pearson_r', 'pearson_p']
    ].copy()

    # Pivot to wide format
    result = result.pivot(
        index='input',
        columns='output',
        values=['spearman_r', 'spearman_p', 'pearson_r', 'pearson_p'],
    )

    # Flatten column names
    result.columns = [f'{col[0]}_{col[1]}' for col in result.columns]
    result = result.reset_index()

    # Add Sobol indices if available
    if sobol_df is not None:
        sobol_pivot = sobol_df.pivot(index='input', columns=None, values=['S1', 'ST'])
        result = result.merge(sobol_pivot, on='input', how='left')

    # Add Morris indices if available
    if morris_df is not None:
        morris_pivot = morris_df.pivot(
            index='input', columns=None, values=['mu_star', 'sigma']
        )
        result = result.merge(morris_pivot, on='input', how='left')

    return result
