"""Sensitivity analysis for Monte Carlo simulation.

Provides functions to assess parameter sensitivity using:
- Correlation coefficients (Pearson, Spearman)
- Sobol indices (variance-based)
- Morris method (elementary effects)

All implementations use vectorized NumPy operations for speed.
"""

from typing import cast

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from scipy import stats
from scipy.stats import qmc

from nozzle_clogging import config
from nozzle_clogging.generation import generate_vectorized_lognormal_particle_sizes
from nozzle_clogging.orchestration import (
    compute_and_classify_clogging_probability,
    compute_physics,
)
from nozzle_clogging.schemas import (
    SimulationInputSchema,
)


def calculate_correlation_sensitivity(
    df: pd.DataFrame,
    input_cols: list[str] | None = None,
    output_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate correlation sensitivity between inputs and outputs.

    Computes Pearson and Spearman correlation coefficients with p-values.
    Uses vectorized scipy operations.

    Args:
        df: Simulation results DataFrame.
        input_cols: List of input column names.
            Defaults to ['TSS_mg_L', 'pressure_kPa', 'nozzle_diameter_mm',
            'duration_hrs', 'particle_diameter_um'].
        output_cols: List of output column names.
            Defaults to ['clogging_probability', 'X', 'volume_fraction'].

    Returns:
        DataFrame with columns:
        - input: Input parameter name
        - output: Output variable name
        - pearson_r: Pearson correlation coefficient
        - pearson_p: Pearson p-value
        - spearman_r: Spearman correlation coefficient
        - spearman_p: Spearman p-value
    """
    if input_cols is None:
        input_cols = [
            'TSS_mg_L',
            'pressure_kPa',
            'nozzle_diameter_mm',
            'duration_hrs',
            'particle_diameter_um',
        ]

    if output_cols is None:
        output_cols = ['clogging_probability', 'X', 'volume_fraction']

    # Extract all input values as matrix (n_samples x n_inputs)
    input_matrix = np.column_stack(
        [df[col].pint.magnitude.values for col in input_cols if col in df.columns]
    )

    # Extract all output values as matrix (n_samples x n_outputs)
    output_matrix = np.column_stack(
        [df[col].pint.magnitude.values for col in output_cols if col in df.columns]
    )

    actual_input_cols = [col for col in input_cols if col in df.columns]
    actual_output_cols = [col for col in output_cols if col in df.columns]

    # Calculate all correlations
    results = []

    for i, input_col in enumerate(actual_input_cols):
        input_vals = input_matrix[:, i]

        for j, output_col in enumerate(actual_output_cols):
            output_vals = output_matrix[:, j]

            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(input_vals, output_vals)

            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(input_vals, output_vals)

            results.append(
                {
                    'input': input_col,
                    'output': output_col,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                }
            )

    return pd.DataFrame(results)


def calculate_sobol_indices(
    n_samples: int = 10_000,
    seed: int = config.RANDOM_SEED,
    output_col: str = 'clogging_probability',
    n_bootstrap: int = 100,
    bootstrap_ci: float = 0.95,
) -> pd.DataFrame:
    """Calculate Sobol sensitivity indices using Jansen estimator with bootstrap CIs.

    Implements variance-based sensitivity analysis using Sobol's method
    with the Jansen (1999) estimator for total-order indices. Bootstrap
    confidence intervals provide robust uncertainty bounds.

    Uses fully vectorized operations for speed.

    Args:
        n_samples: Number of base samples (total samples = n_samples * (2D + 2)).
        seed: Random seed.
        output_col: Output column to analyze.
        n_bootstrap: Number of bootstrap resamples for CI estimation.
        bootstrap_ci: Confidence level for bootstrap intervals (default 0.95).

    Returns:
        DataFrame with columns:
        - input: Input parameter name
        - S1: First-order Sobol index (Saltelli estimator)
        - S1_lo: Lower bound of S1 bootstrap CI
        - S1_hi: Upper bound of S1 bootstrap CI
        - ST: Total-order Sobol index (Jansen estimator)
        - ST_lo: Lower bound of ST bootstrap CI
        - ST_hi: Upper bound of ST bootstrap CI
    """
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=5, scramble=True, rng=rng)

    # Generate Saltelli samples: A, B, and AB matrices
    # For 5 parameters, we need n_samples * (2*5 + 2) = n_samples * 12 samples

    # Generate base samples
    base_samples = sampler.random(n_samples)

    # Create A and B matrices
    A = base_samples
    B = sampler.random(n_samples)

    # Create AB matrices (one column from A, rest from B)
    # AB_i = A with column i taken from B (Saltelli convention; the estimators
    # below require AB_i to differ from A in exactly the i-th column).
    AB_matrices = []
    for i in range(5):
        AB = A.copy()
        AB[:, i] = B[:, i]
        AB_matrices.append(AB)

    # Combine all samples
    all_samples = np.vstack([A, B] + AB_matrices)

    # Scale to parameter ranges (vectorized)
    TSS = config.PARAM_RANGES['TSS'][0] + all_samples[:, 0] * (
        config.PARAM_RANGES['TSS'][1] - config.PARAM_RANGES['TSS'][0]
    )
    pressure = config.PARAM_RANGES['pressure'][0] + all_samples[:, 1] * (
        config.PARAM_RANGES['pressure'][1] - config.PARAM_RANGES['pressure'][0]
    )
    nozzle_diam = config.PARAM_RANGES['nozzle_diameter'][0] + all_samples[:, 2] * (
        config.PARAM_RANGES['nozzle_diameter'][1]
        - config.PARAM_RANGES['nozzle_diameter'][0]
    )
    duration = config.PARAM_RANGES['duration'][0] + all_samples[:, 3] * (
        config.PARAM_RANGES['duration'][1] - config.PARAM_RANGES['duration'][0]
    )

    # Particle size: categorical + lognormal (vectorized)
    particle_classes = np.array(list(config.PARTICLE_SIZE_RANGES.keys()))
    class_idx = (all_samples[:, 4] * len(particle_classes)).astype(int)
    class_idx = np.clip(class_idx, 0, len(particle_classes) - 1)

    particle_diam = generate_vectorized_lognormal_particle_sizes(
        class_idx, rng, config.PARTICLE_SIZE_RANGES
    )

    # Truncate to each class's diameter bounds (matches the truncated-lognormal
    # sampling used in the main simulation; keeps draws inside the design space).
    class_bounds = np.array(
        [config.PARTICLE_SIZE_RANGES[c] for c in particle_classes[class_idx]]
    )
    particle_diam = np.clip(particle_diam, class_bounds[:, 0], class_bounds[:, 1])
    import pint_pandas

    df = pd.DataFrame(
        {
            'TSS_mg_L': pint_pandas.PintArray(TSS, dtype='pint[milligram / liter]'),
            'pressure_kPa': pint_pandas.PintArray(pressure, dtype='pint[kilopascal]'),
            'nozzle_diameter_mm': pint_pandas.PintArray(
                nozzle_diam, dtype='pint[millimeter]'
            ),
            'duration_hrs': pint_pandas.PintArray(duration, dtype='pint[hour]'),
            'particle_diameter_um': pint_pandas.PintArray(
                particle_diam, dtype='pint[micrometer]'
            ),
            'particle_size_range': particle_classes[class_idx],
        }
    )

    # Run simulation (vectorized)
    df = compute_physics(cast(DataFrame[SimulationInputSchema], df))
    df = compute_and_classify_clogging_probability(df)

    # Extract output values
    y = df[output_col].pint.magnitude.values

    # Split outputs into A, B, and AB groups
    y_A = y[:n_samples]
    y_B = y[n_samples : 2 * n_samples]
    y_AB = [
        y[2 * n_samples + i * n_samples : 2 * n_samples + (i + 1) * n_samples]
        for i in range(5)
    ]

    # Stack y_AB for vectorized operations
    y_AB_stack = np.column_stack(y_AB)  # (n_samples, 5)

    def _compute_sobol_indices(
        y_a: np.ndarray,
        y_b: np.ndarray,
        y_ab: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute S1 (Saltelli) and ST (Jansen) for given samples."""
        var_y = np.var(y_a, ddof=1)

        # First-order indices (Saltelli estimator)
        diff_ab_a = y_ab - y_a[:, np.newaxis]
        s1 = np.mean(y_b[:, np.newaxis] * diff_ab_a, axis=0) / var_y

        # Total-order indices (Jansen estimator)
        # Jansen (1999): ST_i = 1 / (2N) * sum( (y_A - y_AB_i)^2 ) / var(y)
        st = 0.5 * np.mean((y_a[:, np.newaxis] - y_ab) ** 2, axis=0) / var_y

        return s1, st

    # Compute full-sample indices
    S1_full, ST_full = _compute_sobol_indices(y_A, y_B, y_AB_stack)

    # Bootstrap confidence intervals (vectorized)
    rng_boot = np.random.default_rng(seed + 1)
    alpha = 1 - bootstrap_ci
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)

    if n_bootstrap > 0:
        # Vectorized bootstrap resampling: (n_bootstrap, n_samples)
        idx = rng_boot.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)

        # Gather resampled outputs: (n_bootstrap, n_samples) -> (n_bootstrap, n_samples)
        S1_boot = np.empty((n_bootstrap, 5))
        ST_boot = np.empty((n_bootstrap, 5))

        for b in range(n_bootstrap):
            s1_b, st_b = _compute_sobol_indices(
                y_A[idx[b]], y_B[idx[b]], y_AB_stack[idx[b], :]
            )
            S1_boot[b, :] = s1_b
            ST_boot[b, :] = st_b

        S1_lo = np.percentile(S1_boot, lo_pct, axis=0)
        S1_hi = np.percentile(S1_boot, hi_pct, axis=0)
        ST_lo = np.percentile(ST_boot, lo_pct, axis=0)
        ST_hi = np.percentile(ST_boot, hi_pct, axis=0)
    else:
        S1_lo = np.full(5, np.nan)
        S1_hi = np.full(5, np.nan)
        ST_lo = np.full(5, np.nan)
        ST_hi = np.full(5, np.nan)

    input_names = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]

    S1_clamped = np.clip(S1_full, 0, 1)
    ST_clamped = np.clip(ST_full, 0, 1)

    return pd.DataFrame(
        {
            'input': input_names,
            'S1': S1_clamped,
            'S1_lo': np.clip(S1_lo, 0, 1),
            'S1_hi': np.clip(S1_hi, 0, 1),
            'ST': ST_clamped,
            'ST_lo': np.clip(ST_lo, 0, 1),
            'ST_hi': np.clip(ST_hi, 0, 1),
        }
    )


def run_sobol_convergence_test(
    sample_sizes: list[int] = [1_000, 2_500, 5_000, 10_000],
    seed: int = config.RANDOM_SEED,
    output_col: str = 'clogging_probability',
    convergence_threshold: float = 0.05,
) -> pd.DataFrame:
    """Run Sobol' indices at multiple sample sizes to assess convergence.

    Computes S1 and ST indices at increasing sample sizes and checks
    whether the relative change between successive sizes falls below
    the convergence threshold.

    Args:
        sample_sizes: List of base sample sizes to test.
            Defaults to [1000, 2500, 5000, 10000].
        seed: Random seed.
        output_col: Output column to analyze.
        convergence_threshold: Maximum relative change to declare convergence.

    Returns:
        DataFrame with columns:
        - sample_size: Base sample size
        - input: Parameter name
        - S1: First-order index
        - ST: Total-order index
        - S1_rel_change: Relative change from previous size
        - ST_rel_change: Relative change from previous size
        - converged: Whether both indices converged
    """

    input_names = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]

    results = []
    prev_s1: np.ndarray | None = None
    prev_st: np.ndarray | None = None

    for n in sample_sizes:
        df = calculate_sobol_indices(
            n_samples=n,
            seed=seed,
            output_col=output_col,
            n_bootstrap=0,
        )
        s1 = df['S1'].to_numpy(dtype=float)
        st = df['ST'].to_numpy(dtype=float)

        if prev_s1 is not None:
            s1_arr = np.asarray(s1, dtype=float)
            prev_s1_arr = np.asarray(prev_s1, dtype=float)
            s1_rel = np.abs(s1_arr - prev_s1_arr) / (np.abs(prev_s1_arr) + 1e-10)
            st_arr = np.asarray(st, dtype=float)
            prev_st_arr = np.asarray(prev_st, dtype=float)
            st_rel = np.abs(st_arr - prev_st_arr) / (np.abs(prev_st_arr) + 1e-10)
            converged = (s1_rel < convergence_threshold) & (
                st_rel < convergence_threshold
            )
        else:
            s1_rel = np.full(5, np.nan)
            st_rel = np.full(5, np.nan)
            converged = np.full(5, False)

        for i, name in enumerate(input_names):
            results.append(
                {
                    'sample_size': n,
                    'input': name,
                    'S1': s1[i],
                    'ST': st[i],
                    'S1_rel_change': s1_rel[i],
                    'ST_rel_change': st_rel[i],
                    'converged': converged[i],
                }
            )

        prev_s1 = s1.copy()
        prev_st = st.copy()

    return pd.DataFrame(results)


def calculate_morris_indices(
    n_samples: int = 1_000,
    n_levels: int = 4,
    seed: int = config.RANDOM_SEED,
    output_col: str = 'clogging_probability',
) -> pd.DataFrame:
    """Calculate Morris elementary effects for screening sensitivity.

    Implements Morris method for identifying influential parameters
    with fewer samples than Sobol. Uses batched vectorized simulation
    for speed.

    Args:
        n_samples: Number of trajectories.
        n_levels: Number of levels for discretization.
        seed: Random seed.
        output_col: Output column to analyze.

    Returns:
        DataFrame with columns:
        - input: Input parameter name
        - mu: Mean of elementary effects
        - mu_star: Mean of absolute elementary effects
        - sigma: Standard deviation of elementary effects
    """
    import pint_pandas

    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=5, scramble=True, rng=rng)

    # Generate all base points at once
    base_points = sampler.random(n_samples)  # (n_samples, 5)

    # Generate all perturbed points at once
    # For each base point, create 5 perturbed points (one per parameter)
    delta = 1.0 / n_levels

    # Create all perturbation pairs: (n_samples * 5, 5)
    all_base = np.repeat(base_points, 5, axis=0)
    all_perturbed = all_base.copy()

    # Apply perturbation to each parameter in sequence
    param_indices = np.tile(np.arange(5), n_samples)
    all_perturbed[np.arange(len(all_perturbed)), param_indices] = (
        all_perturbed[np.arange(len(all_perturbed)), param_indices] + delta
    ) % 1.0

    # Scale to parameter ranges (vectorized)
    def scale_params(
        samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Scale normalized samples [0, 1] to parameter ranges."""
        TSS = config.PARAM_RANGES['TSS'][0] + samples[:, 0] * (
            config.PARAM_RANGES['TSS'][1] - config.PARAM_RANGES['TSS'][0]
        )
        pressure = config.PARAM_RANGES['pressure'][0] + samples[:, 1] * (
            config.PARAM_RANGES['pressure'][1] - config.PARAM_RANGES['pressure'][0]
        )
        nozzle_diam = config.PARAM_RANGES['nozzle_diameter'][0] + samples[:, 2] * (
            config.PARAM_RANGES['nozzle_diameter'][1]
            - config.PARAM_RANGES['nozzle_diameter'][0]
        )
        duration = config.PARAM_RANGES['duration'][0] + samples[:, 3] * (
            config.PARAM_RANGES['duration'][1] - config.PARAM_RANGES['duration'][0]
        )
        class_idx = (samples[:, 4] * len(config.PARTICLE_SIZE_RANGES)).astype(int)
        class_idx = np.clip(class_idx, 0, len(config.PARTICLE_SIZE_RANGES) - 1)
        return TSS, pressure, nozzle_diam, duration, class_idx

    # Scale base and perturbed points
    TSS_base, pressure_base, nozzle_base, duration_base, class_idx_base = scale_params(
        all_base
    )
    TSS_pert, pressure_pert, nozzle_pert, duration_pert, class_idx_pert = scale_params(
        all_perturbed
    )

    # Generate particle sizes (vectorized)
    particle_classes = np.array(list(config.PARTICLE_SIZE_RANGES.keys()))
    particle_diam_base = generate_vectorized_lognormal_particle_sizes(
        class_idx_base, rng, config.PARTICLE_SIZE_RANGES
    )
    particle_diam_pert = generate_vectorized_lognormal_particle_sizes(
        class_idx_pert, rng, config.PARTICLE_SIZE_RANGES
    )

    # Truncate to each class's diameter bounds (matches the truncated-lognormal
    # sampling used in the main simulation; keeps draws inside the design space).
    bounds_base = np.array(
        [config.PARTICLE_SIZE_RANGES[c] for c in particle_classes[class_idx_base]]
    )
    bounds_pert = np.array(
        [config.PARTICLE_SIZE_RANGES[c] for c in particle_classes[class_idx_pert]]
    )
    particle_diam_base = np.clip(
        particle_diam_base, bounds_base[:, 0], bounds_base[:, 1]
    )
    particle_diam_pert = np.clip(
        particle_diam_pert, bounds_pert[:, 0], bounds_pert[:, 1]
    )

    # Create DataFrames for base and perturbed (vectorized)
    df_base = pd.DataFrame(
        {
            'TSS_mg_L': pint_pandas.PintArray(
                TSS_base, dtype='pint[milligram / liter]'
            ),
            'pressure_kPa': pint_pandas.PintArray(
                pressure_base, dtype='pint[kilopascal]'
            ),
            'nozzle_diameter_mm': pint_pandas.PintArray(
                nozzle_base, dtype='pint[millimeter]'
            ),
            'duration_hrs': pint_pandas.PintArray(duration_base, dtype='pint[hour]'),
            'particle_diameter_um': pint_pandas.PintArray(
                particle_diam_base, dtype='pint[micrometer]'
            ),
            'particle_size_range': particle_classes[class_idx_base],
        }
    )

    df_pert = pd.DataFrame(
        {
            'TSS_mg_L': pint_pandas.PintArray(
                TSS_pert, dtype='pint[milligram / liter]'
            ),
            'pressure_kPa': pint_pandas.PintArray(
                pressure_pert, dtype='pint[kilopascal]'
            ),
            'nozzle_diameter_mm': pint_pandas.PintArray(
                nozzle_pert, dtype='pint[millimeter]'
            ),
            'duration_hrs': pint_pandas.PintArray(duration_pert, dtype='pint[hour]'),
            'particle_diameter_um': pint_pandas.PintArray(
                particle_diam_pert, dtype='pint[micrometer]'
            ),
            'particle_size_range': particle_classes[class_idx_pert],
        }
    )

    # Run simulations (vectorized, both at once)
    df_base = compute_physics(cast(DataFrame[SimulationInputSchema], df_base))
    df_base = compute_and_classify_clogging_probability(df_base)
    y_base = df_base[output_col].pint.magnitude.values

    df_pert = compute_physics(cast(DataFrame[SimulationInputSchema], df_pert))
    df_pert = compute_and_classify_clogging_probability(df_pert)
    y_pert = df_pert[output_col].pint.magnitude.values

    # Calculate elementary effects (vectorized)
    ee = (y_pert - y_base) / delta  # (n_samples * 5,)

    # Reshape to (n_samples, 5) for parameter-wise statistics
    ee_matrix = ee.reshape(n_samples, 5)

    # Calculate Morris indices (vectorized)
    mu = np.mean(ee_matrix, axis=0)
    mu_star = np.mean(np.abs(ee_matrix), axis=0)
    sigma = np.std(ee_matrix, axis=0, ddof=1)

    input_names = [
        'TSS_mg_L',
        'pressure_kPa',
        'nozzle_diameter_mm',
        'duration_hrs',
        'particle_diameter_um',
    ]

    return pd.DataFrame(
        {
            'input': input_names,
            'mu': mu,
            'mu_star': mu_star,
            'sigma': sigma,
        }
    )
