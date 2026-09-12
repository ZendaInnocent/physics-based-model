from typing import cast

import numpy as np
import pandas as pd
import pint_pandas
from beartype.typing import Any
from pandera.typing import DataFrame
from scipy.stats import qmc

from nozzle_clogging import config
from nozzle_clogging.schemas import SimulationInputSchema


def compute_lognormal_params(
    ps_min: int | float, ps_max: int | float
) -> tuple[float, float]:
    """
    Convert a particle size range [ps_min, ps_max] (µm) into
    lognormal parameters mu and sigma for sampling.

    Returns:
        mu, sigma : floats
            Lognormal parameters in log-space
    """
    median = np.sqrt(ps_min * ps_max)
    sigma = np.log(ps_max / median)  # geometric standard deviation
    mu = np.log(median)
    return mu, sigma


def generate_vectorized_lognormal_particle_sizes(
    ps_indices: np.ndarray,
    rng: np.random.Generator,
    particle_size_ranges: dict[str, Any],
) -> np.ndarray:
    """
    Fully vectorized lognormal particle diameter generator.

    Parameters
    ----------
    ps_indices : ndarray
        Integer indices of particle size classes for each sample.
    rng : np.random.Generator
        NumPy random generator
    particle_size_ranges : dict
        Mapping from class name to (min_diameter_um, max_diameter_um) tuple

    Returns
    -------
    particle_diameters : ndarray
        Array of particle diameters in µm, float64
    """
    n = len(ps_indices)
    particle_diameters = np.empty(n, dtype=np.float64)

    # Build mu and sigma arrays for each particle class
    params = [
        compute_lognormal_params(*particle_size_ranges[name])
        for name in particle_size_ranges
    ]
    mu_arr = np.array([p[0] for p in params], dtype=np.float64)
    sigma_arr = np.array([p[1] for p in params], dtype=np.float64)

    # Map class indices to lognormal parameters
    mu = mu_arr[ps_indices]
    sigma = sigma_arr[ps_indices]

    # Vectorized sampling
    particle_diameters[:] = rng.lognormal(mean=mu, sigma=sigma)

    # Clip tiny values to avoid zero (prevents Stokes validation errors)
    particle_diameters = np.clip(particle_diameters, 1e-3, None)

    return particle_diameters


def generate_lhs_samples(
    n_samples: int, seed: int = config.RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Latin Hypercube samples across the sprinkler clogging
    parameter space.
    """

    rng = np.random.default_rng(seed)

    sampler = qmc.LatinHypercube(d=5, scramble=True, rng=rng)
    lhs = sampler.random(n_samples)

    # Continuous scaling
    TSS = config.PARAM_RANGES['TSS'][0] + lhs[:, 0] * (
        config.PARAM_RANGES['TSS'][1] - config.PARAM_RANGES['TSS'][0]
    )

    pressure = config.PARAM_RANGES['pressure'][0] + lhs[:, 1] * (
        config.PARAM_RANGES['pressure'][1] - config.PARAM_RANGES['pressure'][0]
    )

    nozzle_diam = config.PARAM_RANGES['nozzle_diameter'][0] + lhs[:, 2] * (
        config.PARAM_RANGES['nozzle_diameter'][1]
        - config.PARAM_RANGES['nozzle_diameter'][0]
    )

    duration = config.PARAM_RANGES['duration'][0] + lhs[:, 3] * (
        config.PARAM_RANGES['duration'][1] - config.PARAM_RANGES['duration'][0]
    )

    # particle class sampling
    particle_classes = np.array(list(config.PARTICLE_SIZE_RANGES.keys()))
    class_idx = (lhs[:, 4] * len(particle_classes)).astype(int)
    class_idx = np.clip(class_idx, 0, len(particle_classes) - 1)

    ps_name = particle_classes[class_idx]

    particle_diam = generate_vectorized_lognormal_particle_sizes(
        class_idx, rng, config.PARTICLE_SIZE_RANGES
    )

    return (
        TSS,
        pressure,
        nozzle_diam,
        duration,
        particle_diam,
        ps_name,
    )


def generate_simulation_inputs(
    n_samples: int, seed: int = config.RANDOM_SEED
) -> DataFrame[SimulationInputSchema]:
    """
    Generate Latin Hypercube samples across the sprinkler clogging parameter space.

    Args:
        n_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame adhering to :class:`SimulationInputSchema` with pint extension
        dtypes for all numeric columns.

    Raises:
        ValueError: If generated data violates SimulationInputSchema constraints.
        BeartypeCallHintViolation: If input is not the expected type.
    """
    (TSS, pressure, nozzle_diam, duration, particle_diam, ps_range) = (
        generate_lhs_samples(n_samples, seed)
    )

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
            'particle_size_range': ps_range,
        }
    )

    return cast(DataFrame[SimulationInputSchema], df)