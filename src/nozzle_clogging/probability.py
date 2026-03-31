import numpy as np
import pandas as pd
import scipy.special as special

from .config import (
    RHO_SEDIMENT,
    RISK_LEVELS,
    RISK_LOW_THRESHOLD,
    RISK_MODERATE_THRESHOLD,
)
from .physics import broadcast_inputs, calculate_physical_modifiers, validate_positive


def calculate_clogging_probability(
    tss_mg_L,
    particle_diameter_um,
    nozzle_diameter_mm,
    velocity_m_s,
    duration_hrs,
    stokes_factor,
    dp_dn_ratio,
    dp_dn_factor,
    velocity_shear_factor,
    settling_velocity_factor,
    logistic_scale=1.0,
    centering_offset=3.0,
    apply_physical_constraints=True,
    rho_particle=RHO_SEDIMENT,
):
    """Compute clogging probability using a physics-informed model.

    Core dimensionless clogging potential:
        X = phi x (dp/Dn)^2 x (V x t / Dn)

    Physical modifiers applied multiplicatively:
        X_final = X x Phi

    Final probability via logistic transformation:
        Pc = 1 / (1 + exp(-logistic_scale x (X_final - offset)))
    """
    validate_positive(tss_mg_L, "tss_mg_L")
    validate_positive(particle_diameter_um, "particle_diameter_um")
    validate_positive(nozzle_diameter_mm, "nozzle_diameter_mm")
    validate_positive(velocity_m_s, "velocity_m_s")
    validate_positive(duration_hrs, "duration_hrs")

    tss_arr = np.asarray(tss_mg_L, dtype=np.float64)
    particle_diameter_arr = np.asarray(particle_diameter_um, dtype=np.float64)
    nozzle_diameter_arr = np.asarray(nozzle_diameter_mm, dtype=np.float64)
    velocity_arr = np.asarray(velocity_m_s, dtype=np.float64)
    duration_arr = np.asarray(duration_hrs, dtype=np.float64)

    (
        tss_mg_L,
        particle_diameter_um,
        nozzle_diameter_mm,
        velocity_m_s,
        duration_hrs,
        stokes_factor,
        dp_dn_ratio,
        dp_dn_factor,
        velocity_shear_factor,
        settling_velocity_factor,
    ) = broadcast_inputs(
        tss_arr,
        particle_diameter_arr,
        nozzle_diameter_arr,
        velocity_arr,
        duration_arr,
        stokes_factor,
        dp_dn_ratio,
        dp_dn_factor,
        velocity_shear_factor,
        settling_velocity_factor,
    )

    # Volume fraction
    volume_fraction = (tss_mg_L * 1e-3) / rho_particle

    safe_velocity = np.maximum(velocity_arr, 0.01)
    duration_s = duration_hrs * 3600
    nozzle_diameter_m = nozzle_diameter_mm * 1e-3
    residence_term = (safe_velocity * duration_s) / nozzle_diameter_m
    X_base = volume_fraction * (dp_dn_ratio**2) * residence_term
    physical_factor = 1.0

    if apply_physical_constraints:
        physical_factor = calculate_physical_modifiers(
            dp_dn_factor,
            stokes_factor,
            velocity_shear_factor,
            settling_velocity_factor,
        )
        physical_factor = np.clip(physical_factor, 0.1, 5.0)
        X = X_base * physical_factor
    else:
        X = X_base

    X = np.clip(X, 0, 50)
    Pc = special.expit(logistic_scale * (X - centering_offset))
    Pc = np.clip(Pc, 0.0, 1.0)

    return volume_fraction, X_base, physical_factor, X, Pc


def classify_clogging_risk(clogging_probability):
    """Classify clogging probability into categorical risk levels."""
    arr = np.asarray(clogging_probability, dtype=np.float64)
    arr = np.clip(arr, 0.0, 1.0)
    bins = np.array([RISK_LOW_THRESHOLD, RISK_MODERATE_THRESHOLD])
    labels = np.array(RISK_LEVELS, dtype=object)
    idx = np.digitize(arr, bins)
    result = labels[idx]

    if np.isscalar(clogging_probability):
        return result.item()

    if isinstance(clogging_probability, pd.Series):
        return pd.Series(result, index=clogging_probability.index, name="clogging_risk")
