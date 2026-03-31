import numpy as np

from .config import (
    CD,
    DP_DN_OBSTRUCTION_THRESHOLD,
    DP_DN_RISK_THRESHOLD,
    NU,
    RHO_SEDIMENT,
    RHO_WATER,
    SETTLING_VELOCITY_RATIO_CRITICAL,
    STOKES_CRITICAL,
    VELOCITY_SHEAR_THRESHOLD,
    g,
)


def broadcast_inputs(*arrays):
    """Safely broadcast multiple inputs to the same shape."""
    arrays = [np.asarray(a) for a in arrays]
    return np.broadcast_arrays(*arrays)


def validate_positive(value, name, allow_zero=False):
    """Validate that a value is positive."""
    val_arr = np.asarray(value)
    if allow_zero:
        if np.any(val_arr < 0):
            bad_val = val_arr[val_arr < 0][0]
            raise ValueError(f"{name} must be non-negative, got {bad_val}")
    else:
        if np.any(val_arr <= 0):
            bad_val = val_arr[val_arr <= 0][0]
            raise ValueError(f"{name} must be positive, got {bad_val}")


def calculate_velocity_from_pressure(pressure_kpa, cd=CD, rho=RHO_WATER):
    """Calculate exit velocity from nozzle pressure."""
    pressure_pa = np.asarray(pressure_kpa) * 1000
    v = cd * np.sqrt((2 * pressure_pa) / rho)
    return v


def calculate_settling_velocity_and_factor(
    particle_diameter_um,
    velocity_m_s,
    settling_ratio_critical=SETTLING_VELOCITY_RATIO_CRITICAL,
    rho_particle=RHO_SEDIMENT,
    rho_fluid=RHO_WATER,
    nu=NU,
):
    """Compute settling velocity using Ferguson & Church equation."""
    validate_positive(particle_diameter_um, "particle_diameter_um")
    validate_positive(rho_particle, "rho_particle")
    validate_positive(rho_fluid, "rho_fluid")
    validate_positive(nu, "nu")

    R = (rho_particle - rho_fluid) / rho_fluid
    particle_diameter_m = np.asarray(particle_diameter_um) * 1e-6

    c1, c2 = 18, 1.0
    settling_velocity = (R * g * (particle_diameter_m**2)) / (
        c1 * nu + np.sqrt(0.75 * c2 * R * g * (particle_diameter_m**3))
    )
    velocity = np.asarray(velocity_m_s)
    ratio = np.asarray(settling_velocity) / np.where(velocity <= 0, 1e-10, velocity)

    factor = np.where(
        ratio < settling_ratio_critical,
        np.maximum(0.2, ratio / settling_ratio_critical),
        1.0 + 0.15 * np.log1p(ratio / settling_ratio_critical),
    )

    settling_velocity_factor = np.where(velocity <= 0, 1.0, factor)
    return settling_velocity, settling_velocity_factor


def calculate_dp_dn_ratio_and_factor(
    particle_diameter_um,
    nozzle_diameter_mm,
    dp_dn_obstruction_threshold=DP_DN_OBSTRUCTION_THRESHOLD,
    dp_dn_risk_threshold=DP_DN_RISK_THRESHOLD,
):
    """Calculate particle-to-nozzle diameter ratio and geometric factor."""
    validate_positive(particle_diameter_um, "particle_diameter_um")
    validate_positive(nozzle_diameter_mm, "nozzle_diameter_mm")

    dp_m = np.asarray(particle_diameter_um) * 1e-6
    dn_m = np.asarray(nozzle_diameter_mm) * 1e-3
    dp_m, dn_m = broadcast_inputs(dp_m, dn_m)
    ratio = dp_m / np.maximum(dn_m, 1e-12)

    factor = np.where(
        ratio < dp_dn_obstruction_threshold,
        0.5 + 0.5 * (ratio / dp_dn_obstruction_threshold),
        np.where(
            ratio < dp_dn_risk_threshold,
            1.0,
            1.0 + 0.3 * np.log1p(ratio / dp_dn_risk_threshold),
        ),
    )

    return ratio, factor


def calculate_stokes_number(
    particle_diameter_um,
    velocity_m_s,
    nozzle_diameter_mm,
    rho_particle=RHO_SEDIMENT,
    rho_fluid=RHO_WATER,
):
    """Calculate Stokes number."""
    validate_positive(particle_diameter_um, "particle_diameter_um")
    validate_positive(velocity_m_s, "velocity_m_s")
    validate_positive(nozzle_diameter_mm, "nozzle_diameter_mm")

    particle_diameter_m = np.asarray(particle_diameter_um) * 1e-6
    nozzle_diameter_m = np.asarray(nozzle_diameter_mm) * 1e-3
    velocity = np.asarray(velocity_m_s)

    particle_diameter_m, velocity, nozzle_diameter_m = broadcast_inputs(
        particle_diameter_m, velocity, nozzle_diameter_m
    )

    mu = NU * rho_fluid
    Stk = (rho_particle * (particle_diameter_m**2) * np.maximum(velocity, 1e-10)) / (
        18 * mu * nozzle_diameter_m
    )
    return Stk


def calculate_stokes_factor(
    velocity_m_s,
    stokes_number,
    stokes_critical=STOKES_CRITICAL,
):
    """Calculate Stokes constraint factor."""
    validate_positive(velocity_m_s, "velocity_m_s")

    velocity = np.asarray(velocity_m_s)
    stokes_number = np.asarray(stokes_number)
    velocity, stokes_number = broadcast_inputs(velocity, stokes_number)

    factor = np.where(
        stokes_number < stokes_critical,
        0.3 + 0.7 * (stokes_number / stokes_critical),
        1.0 + 0.1 * np.log1p(stokes_number / stokes_critical),
    )

    return np.where(velocity <= 0, 1.0, factor)


def calculate_velocity_shear_factor(
    velocity_m_s,
    velocity_shear_threshold=VELOCITY_SHEAR_THRESHOLD,
    min_factor=0.25,
):
    """Calculate velocity shear self-cleaning factor."""
    velocity = np.asarray(velocity_m_s)
    velocity_shear_threshold = np.asarray(velocity_shear_threshold)
    velocity, velocity_shear_threshold = broadcast_inputs(
        velocity, velocity_shear_threshold
    )

    factor = 1.0 / (1.0 + 0.15 * np.maximum(0, velocity - velocity_shear_threshold / 4))
    return np.maximum(factor, min_factor)


def calculate_physical_modifiers(
    dp_dn_factor,
    stokes_factor,
    velocity_shear_factor,
    settling_velocity_factor=1.0,
):
    """Combine all physical clogging modifiers."""
    (
        dp_dn_factor,
        stokes_factor,
        velocity_shear_factor,
        settling_velocity_factor,
    ) = broadcast_inputs(
        dp_dn_factor,
        stokes_factor,
        velocity_shear_factor,
        settling_velocity_factor,
    )

    physical_factor = (
        dp_dn_factor * stokes_factor * velocity_shear_factor * settling_velocity_factor
    )
    physical_factor = np.clip(physical_factor, 0.1, 5.0)
    return physical_factor
