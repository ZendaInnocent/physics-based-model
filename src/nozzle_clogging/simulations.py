import math

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import trange

from .config import (
    PARAM_RANGES,
    PARTICLE_SIZE_RANGES,
    RANDOM_SEED,
)
from .physics import (
    calculate_dp_dn_ratio_and_factor,
    calculate_settling_velocity_and_factor,
    calculate_stokes_factor,
    calculate_stokes_number,
    calculate_velocity_from_pressure,
    calculate_velocity_shear_factor,
)
from .probability import (
    calculate_clogging_probability,
    classify_clogging_risk,
)


def compute_lognormal_params(ps_min, ps_max):
    """Convert a particle size range [ps_min, ps_max] (um) into lognormal parameters."""
    median = np.sqrt(ps_min * ps_max)
    sigma = np.log(ps_max / median)
    mu = np.log(median)
    return mu, sigma


def generate_vectorized_lognormal_particle_sizes(ps_indices, rng, particle_size_ranges):
    """Fully vectorized lognormal particle diameter generator."""
    n = len(ps_indices)
    particle_diameters = np.empty(n, dtype=np.float64)

    params = [
        compute_lognormal_params(*particle_size_ranges[name])
        for name in particle_size_ranges
    ]
    mu_arr = np.array([p[0] for p in params], dtype=np.float64)
    sigma_arr = np.array([p[1] for p in params], dtype=np.float64)

    mu = mu_arr[ps_indices]
    sigma = sigma_arr[ps_indices]

    particle_diameters[:] = rng.lognormal(mean=mu, sigma=sigma)
    particle_diameters = np.clip(particle_diameters, 1e-3, None)

    return particle_diameters


def generate_lhs_samples(n_samples, seed):
    """Generate Latin Hypercube samples across the sprinkler clogging parameter space."""
    rng = np.random.default_rng(seed)

    sampler = qmc.LatinHypercube(d=5, scramble=True, seed=seed)
    lhs = sampler.random(n_samples)

    TSS = PARAM_RANGES["TSS"][0] + lhs[:, 0] * (
        PARAM_RANGES["TSS"][1] - PARAM_RANGES["TSS"][0]
    )

    pressure = PARAM_RANGES["pressure"][0] + lhs[:, 1] * (
        PARAM_RANGES["pressure"][1] - PARAM_RANGES["pressure"][0]
    )

    nozzle_diam = PARAM_RANGES["nozzle_diameter"][0] + lhs[:, 2] * (
        PARAM_RANGES["nozzle_diameter"][1] - PARAM_RANGES["nozzle_diameter"][0]
    )

    duration = PARAM_RANGES["duration"][0] + lhs[:, 3] * (
        PARAM_RANGES["duration"][1] - PARAM_RANGES["duration"][0]
    )

    particle_classes = np.array(list(PARTICLE_SIZE_RANGES.keys()))
    class_idx = (lhs[:, 4] * len(particle_classes)).astype(int)
    class_idx = np.clip(class_idx, 0, len(particle_classes) - 1)

    ps_name = particle_classes[class_idx]

    particle_diam = generate_vectorized_lognormal_particle_sizes(
        class_idx, rng, PARTICLE_SIZE_RANGES
    )

    return (
        TSS,
        pressure,
        nozzle_diam,
        duration,
        particle_diam,
        ps_name,
    )


def generate_simulation_inputs(n_samples, seed):
    """Generate complete simulation inputs as DataFrame."""
    (TSS, pressure, nozzle_diam, duration, particle_diam, ps_range) = (
        generate_lhs_samples(n_samples, seed)
    )

    velocity = calculate_velocity_from_pressure(pressure)

    return pd.DataFrame(
        {
            "TSS_mg_L": TSS,
            "pressure_kPa": pressure,
            "velocity_m_s": velocity,
            "nozzle_diameter_mm": nozzle_diam,
            "duration_hrs": duration,
            "particle_diameter_um": particle_diam,
            "particle_size_range": ps_range,
        }
    )


def compute_physics(df):
    """Calculate all physical quantities."""
    df["stokes_number"] = calculate_stokes_number(
        df["particle_diameter_um"], df["velocity_m_s"], df["nozzle_diameter_mm"]
    )

    df["stokes_factor"] = calculate_stokes_factor(
        df["velocity_m_s"], df["stokes_number"]
    )

    (df["dp_dn_ratio"], df["dp_dn_factor"]) = calculate_dp_dn_ratio_and_factor(
        df["particle_diameter_um"], df["nozzle_diameter_mm"]
    )

    df["velocity_shear_factor"] = calculate_velocity_shear_factor(df["velocity_m_s"])

    (df["settling_velocity"], df["settling_velocity_factor"]) = (
        calculate_settling_velocity_and_factor(
            df["particle_diameter_um"], df["velocity_m_s"]
        )
    )

    return df


def compute_clogging(df):
    """Calculate clogging probability and risk classification."""
    (
        df["volume_fraction"],
        df["X_base"],
        df["physical_factor"],
        df["X"],
        df["clogging_probability"],
    ) = calculate_clogging_probability(
        df["TSS_mg_L"],
        df["particle_diameter_um"],
        df["nozzle_diameter_mm"],
        df["velocity_m_s"],
        df["duration_hrs"],
        df["stokes_factor"],
        df["dp_dn_ratio"],
        df["dp_dn_factor"],
        df["velocity_shear_factor"],
        df["settling_velocity_factor"],
    )

    df["clogging_risk"] = classify_clogging_risk(df["clogging_probability"])

    return df


def simulate_nozzle_clogging(total_samples=20_000, chunk_size=2_000, seed=RANDOM_SEED):
    """Run the sprinkler clogging simulation in memory-safe batches."""
    results = []

    batches = math.ceil(total_samples / chunk_size)

    for i in trange(batches):
        n = min(chunk_size, total_samples - i * chunk_size)
        df = generate_simulation_inputs(n, seed + i)
        df = compute_physics(df)
        df = compute_clogging(df)
        results.append(df)
    return pd.concat(results, ignore_index=True)
