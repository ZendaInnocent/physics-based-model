"""Convergence analysis for Monte Carlo simulation.

Provides functions to assess convergence of Monte Carlo estimates
by running simulations at multiple sample sizes and analyzing
how statistics stabilize.
"""

import numpy as np
import pandas as pd

from nozzle_clogging import config
from nozzle_clogging.simulation import run_simulation


def run_convergence_study(
    sample_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
    chunk_size: int = 2_000,
    outputs: list[str] | None = None,
) -> pd.DataFrame:
    """Run convergence study at multiple sample sizes.

    For each sample size, runs simulation and calculates statistics
    (mean, std, 95% CI) for specified outputs.

    Args:
        sample_sizes: List of sample sizes to test.
            Defaults to [1_000, 5_000, 10_000, 20_000, 50_000].
        seeds: List of random seeds for each sample size.
            Defaults to [config.RANDOM_SEED + i for i in range(len(sample_sizes))].
        chunk_size: Batch size for simulation.
        outputs: List of output column names to analyze.
            Defaults to ['clogging_probability', 'X', 'volume_fraction'].

    Returns:
        DataFrame with columns:
        - sample_size: Number of samples
        - seed: Random seed used
        - output: Output column name
        - mean: Mean value
        - std: Standard deviation
        - ci_lower: Lower 95% confidence interval
        - ci_upper: Upper 95% confidence interval
        - ci_width: Width of confidence interval
    """
    if sample_sizes is None:
        sample_sizes = [1_000, 5_000, 10_000, 20_000, 50_000]

    if seeds is None:
        seeds = [config.RANDOM_SEED + i for i in range(len(sample_sizes))]

    if outputs is None:
        outputs = ['clogging_probability', 'X', 'volume_fraction']

    results = []

    for n_samples, seed in zip(sample_sizes, seeds):
        df = run_simulation(
            total_samples=n_samples,
            chunk_size=min(chunk_size, n_samples),
            seed=seed,
        )

        for output in outputs:
            if output in df.columns:
                values = df[output].pint.magnitude

                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                se = std_val / np.sqrt(n_samples)
                ci_lower = mean_val - 1.96 * se
                ci_upper = mean_val + 1.96 * se
                ci_width = ci_upper - ci_lower

                results.append(
                    {
                        'sample_size': n_samples,
                        'seed': seed,
                        'output': output,
                        'mean': mean_val,
                        'std': std_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_width,
                    }
                )

    return pd.DataFrame(results)


def calculate_convergence_metrics(
    convergence_df: pd.DataFrame,
    reference_size: int | None = None,
) -> pd.DataFrame:
    """Calculate convergence metrics comparing each sample size to a reference.

    For each sample size (optionally excluding the reference size), calculates:
    - Relative change in mean (vs reference sample size)
    - Relative change in CI width (vs reference sample size)
    - Convergence indicator (True if both < threshold)

    Args:
        convergence_df: Output from run_convergence_study.
        reference_size: Sample size to use as reference baseline.
            If None, compares each size to the previous (smaller) size.

    Returns:
        DataFrame with convergence metrics for each sample size and output.
    """
    convergence_df = convergence_df.sort_values(['output', 'sample_size'])

    metrics = []

    for output in convergence_df['output'].unique():
        output_df = convergence_df[convergence_df['output'] == output].sort_values(
            'sample_size'
        )

        if len(output_df) < 2:
            continue

        # Find reference row if specified
        ref_row = None
        if reference_size is not None:
            ref_match = output_df[output_df['sample_size'] == reference_size]
            if not ref_match.empty:
                ref_row = ref_match.iloc[0]

        for i in range(len(output_df)):
            curr_row = output_df.iloc[i]

            # Skip the reference size itself if it's specified
            if reference_size is not None and curr_row['sample_size'] == reference_size:
                continue

            curr_mean = curr_row['mean']
            curr_ci_width = curr_row['ci_width']
            n_samples = curr_row['sample_size']

            # Determine reference values
            if ref_row is not None:
                # Use specified reference size
                ref_mean = ref_row['mean']
                ref_ci_width = ref_row['ci_width']
            else:
                # Use previous sample size (default behavior)
                if i == 0:
                    continue  # Skip smallest size as there's no previous
                prev_row = output_df.iloc[i - 1]
                ref_mean = prev_row['mean']
                ref_ci_width = prev_row['ci_width']

            # Calculate relative changes
            if ref_mean != 0:
                mean_change = abs(curr_mean - ref_mean) / abs(ref_mean)
            else:
                mean_change = abs(curr_mean - ref_mean)

            if ref_ci_width != 0:
                ci_change = abs(curr_ci_width - ref_ci_width) / abs(ref_ci_width)
            else:
                ci_change = abs(curr_ci_width - ref_ci_width)

            converged = mean_change < 0.01 and ci_change < 0.05

            metrics.append(
                {
                    'sample_size': n_samples,
                    'output': output,
                    'mean': curr_mean,
                    'mean_change': mean_change,
                    'ci_width': curr_ci_width,
                    'ci_width_change': ci_change,
                    'converged': converged,
                }
            )

    return pd.DataFrame(metrics)


def detect_convergence(
    convergence_df: pd.DataFrame,
    mean_threshold: float = 0.01,
    ci_threshold: float = 0.05,
) -> dict[str, int]:
    """Detect minimum sample size for convergence for each output.

    Convergence is defined as:
    - Relative change in mean < mean_threshold (default 1%)
    - Relative change in CI width < ci_threshold (default 5%)
    compared to the largest sample size.

    Args:
        convergence_df: Output from run_convergence_study.
        mean_threshold: Threshold for mean change (default 0.01 = 1%).
        ci_threshold: Threshold for CI width change (default 0.05 = 5%).

    Returns:
        Dictionary mapping output name to minimum sample size for convergence.
    """
    metrics = calculate_convergence_metrics(convergence_df)

    convergence_sizes = {}

    for output in metrics['output'].unique():
        output_metrics = metrics[metrics['output'] == output].sort_values('sample_size')

        # Find first sample size where both conditions are met
        converged_rows = output_metrics[
            (output_metrics['mean_change'] < mean_threshold)
            & (output_metrics['ci_width_change'] < ci_threshold)
        ]

        if not converged_rows.empty:
            convergence_sizes[output] = int(converged_rows.iloc[0]['sample_size'])
        else:
            # If no convergence, return largest tested size
            convergence_sizes[output] = int(output_metrics['sample_size'].max())

    return convergence_sizes


def run_multiple_seeds(
    sample_size: int,
    n_runs: int = 5,
    base_seed: int = config.RANDOM_SEED,
    chunk_size: int = 2_000,
    outputs: list[str] | None = None,
) -> pd.DataFrame:
    """Run simulation multiple times with different seeds to assess variance.

    Args:
        sample_size: Number of samples per run.
        n_runs: Number of runs with different seeds.
        base_seed: Base seed (each run uses base_seed + i).
        chunk_size: Batch size for simulation.
        outputs: List of output column names to analyze.

    Returns:
        DataFrame with columns:
        - run: Run number (0-indexed)
        - seed: Random seed used
        - output: Output column name
        - mean: Mean value
        - std: Standard deviation
        - ci_lower: Lower 95% confidence interval
        - ci_upper: Upper 95% confidence interval
    """
    if outputs is None:
        outputs = ['clogging_probability', 'X', 'volume_fraction']

    results = []

    for i in range(n_runs):
        seed = base_seed + i
        df = run_simulation(
            total_samples=sample_size,
            chunk_size=min(chunk_size, sample_size),
            seed=seed,
        )

        for output in outputs:
            if output in df.columns:
                values = df[output].pint.magnitude

                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)
                se = std_val / np.sqrt(sample_size)
                ci_lower = mean_val - 1.96 * se
                ci_upper = mean_val + 1.96 * se

                results.append(
                    {
                        'run': i,
                        'seed': seed,
                        'output': output,
                        'mean': mean_val,
                        'std': std_val,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                    }
                )

    return pd.DataFrame(results)


def calculate_batch_variance(
    df: pd.DataFrame,
    batch_size: int = 2_000,
    outputs: list[str] | None = None,
) -> pd.DataFrame:
    """Analyze variance between batches.

    Splits simulation results into batches and calculates statistics
    for each batch to assess between-batch variance.

    Args:
        df: Full simulation results (from run_simulation).
        batch_size: Number of samples per batch.
        outputs: List of output column names to analyze.

    Returns:
        DataFrame with columns:
        - batch: Batch number (0-indexed)
        - output: Output column name
        - mean: Mean value for batch
        - std: Standard deviation for batch
        - n_samples: Number of samples in batch
    """
    if outputs is None:
        outputs = ['clogging_probability', 'X', 'volume_fraction']

    n_samples = len(df)
    n_batches = n_samples // batch_size

    results = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]

        for output in outputs:
            if output in batch_df.columns:
                values = batch_df[output].pint.magnitude

                results.append(
                    {
                        'batch': i,
                        'output': output,
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'n_samples': len(values),
                    }
                )

    batch_df = pd.DataFrame(results)

    # Calculate between-batch variance
    variance_results = []
    for output in outputs:
        output_batches = batch_df[batch_df['output'] == output]
        batch_means = output_batches['mean'].to_numpy(dtype=float)

        variance_results.append(
            {
                'output': output,
                'between_batch_std': np.std(batch_means, ddof=1),
                'between_batch_cv': np.std(batch_means, ddof=1) / np.mean(batch_means)
                if np.mean(batch_means) != 0
                else np.nan,
                'n_batches': len(output_batches),
            }
        )

    return pd.DataFrame(variance_results)
