"""Physics calculations for the nozzle-clogging model.

All functions accept pint Quantities (scalar or array) and return
pint Quantities with correct units. Pint's native arithmetic handles
both scalar and array operations automatically.
"""

import numpy as np
import pint
from beartype.typing import Any

from nozzle_clogging import config
from nozzle_clogging.units import ureg

__all__ = [
    'calculate_velocity_from_pressure',
    'calculate_physical_modifiers',
    'calculate_stokes_number',
    'calculate_stokes_factor',
    'calculate_dp_dn_ratio_and_factor',
    'calculate_velocity_shear_factor',
    'calculate_settling_velocity_and_factor',
]


def calculate_velocity_from_pressure(
    pressure_kpa: pint.Quantity[Any],
    cd: float = config.CD,
    rho: pint.Quantity[Any] = config.PhysicsConstants.RHO_WATER,
) -> pint.Quantity[Any]:
    """Calculate exit velocity from nozzle pressure.

    Uses Bernoulli's principle: v = Cd * sqrt(2 * P / rho)

    Args:
        pressure_kpa: Pressure in kPa as pint Quantity.
        cd: Coefficient of Discharge (dimensionless).
        rho: Fluid density as pint Quantity. Defaults to water.

    Returns:
        Exit velocity as pint Quantity in m/s.

    Raises:
        DimensionalityError: If pressure or rho have incompatible units.
    """
    v = cd * np.sqrt((2 * pressure_kpa.to('Pa')) / rho)
    return v.to('m/s')


def calculate_physical_modifiers(
    dp_dn_factor: pint.Quantity[Any],
    stokes_factor: pint.Quantity[Any],
    velocity_shear_factor: pint.Quantity[Any],
    settling_velocity_factor: pint.Quantity[Any] = ureg.Quantity(1.0, 'dimensionless'),
) -> pint.Quantity[Any]:
    """Combine physical clogging modifiers.

    Phi = f(dp/Dn) * f(Stk) * f(ws/V) * f(V_shear)

    All factors are designed to be >= 1.0 to amplify clogging potential,
    not reduce it. The combined factor provides discrimination across
    the range of operating conditions.

    Args:
        dp_dn_factor: Geometric bridging factor (dimensionless, >= 1.0).
        stokes_factor: Particle inertia factor (dimensionless, >= 1.0).
        velocity_shear_factor: Flow self-cleaning factor (dimensionless).
        settling_velocity_factor: Gravitational settling factor (dimensionless).

    Returns:
        Combined multiplicative scaling factor (dimensionless).

    Raises:
        DimensionalityError: If any input has incompatible units.
    """
    # Combine all factors - all should be >= 1.0 for risk amplification
    # The minimum bound of 0.5 ensures the combined factor doesn't
    # collapse to zero for low-risk conditions
    physical_factor = (
        dp_dn_factor * stokes_factor * velocity_shear_factor * settling_velocity_factor
    )
    return np.clip(physical_factor, 0.5, 10.0)


def calculate_stokes_number(
    particle_diameter_um: pint.Quantity[Any],
    velocity_m_s: pint.Quantity[Any],
    nozzle_diameter_mm: pint.Quantity[Any],
    rho_particle: pint.Quantity[Any] = config.PhysicsConstants.RHO_SEDIMENT,
    rho_fluid: pint.Quantity[Any] = config.PhysicsConstants.RHO_WATER,
    nu_fluid: pint.Quantity[Any] = config.PhysicsConstants.NU_WATER,
) -> pint.Quantity[Any]:
    """Calculate Stokes number (dimensionless).

    Stk = (rho_p * dp^2 * V) / (18 * mu * Dn)

    Stk << 1: particles follow flow lines (low deposition)
    Stk >> 1: particles maintain trajectory (high deposition)

    Args:
        particle_diameter_um: Particle diameter as pint Quantity.
        velocity_m_s: Flow velocity as pint Quantity.
        nozzle_diameter_mm: Nozzle diameter as pint Quantity.
        rho_particle: Particle density as pint Quantity. Defaults to sediment.
        rho_fluid: Fluid density as pint Quantity. Defaults to water.
        nu_fluid: Kinematic viscosity as pint Quantity. Defaults to water.

    Returns:
        Stokes number as pint Quantity (dimensionless).

    Raises:
        DimensionalityError: If any input has incompatible units.
    """
    mu = nu_fluid * rho_fluid
    dp_m = particle_diameter_um.to('m')
    dn_m = nozzle_diameter_mm.to('m')
    v_safe = np.maximum(velocity_m_s, ureg.Quantity(1e-10, 'm/s'))  # type: ignore
    Stk = (rho_particle * (dp_m**2) * v_safe) / (18 * mu * dn_m)
    return Stk.to('dimensionless')


def calculate_stokes_factor(
    velocity_m_s: pint.Quantity[Any],
    stokes_number: pint.Quantity[Any],
    stokes_critical: float = config.STOKES_CRITICAL,
) -> pint.Quantity[Any]:
    """Calculate multiplicative factor based on Stokes number.

    Low Stk (< 0.1): particles follow flow, reduced deposition (factor < 1)
    High Stk (> 0.1): particles maintain inertia, increased deposition (factor > 1)

    Args:
        velocity_m_s: Velocity as pint Quantity.
        stokes_number: Stokes number (dimensionless).
        stokes_critical: Critical Stokes number threshold (dimensionless).

    Returns:
        Stokes factor as pint Quantity (dimensionless).

    Raises:
        DimensionalityError: If velocity has incompatible units.
    """
    # For high Stokes numbers (particles maintain inertia), risk increases
    factor = np.where(
        stokes_number < stokes_critical,
        0.5 + 0.5 * (stokes_number / stokes_critical),
        1.0 + 0.15 * np.log1p(stokes_number / stokes_critical),
    )

    # Allow factor to exceed 1.0 for high inertia conditions
    return np.where(  # type: ignore
        velocity_m_s <= ureg.Quantity(0, 'm/s'),
        ureg.Quantity(1.0, 'dimensionless'),
        np.clip(factor, 0.5, 2.0),
    )


def calculate_dp_dn_ratio_and_factor(
    particle_diameter_um: pint.Quantity[Any],
    nozzle_diameter_mm: pint.Quantity[Any],
    dp_dn_obstruction_threshold: float = config.DP_DN_OBSTRUCTION_THRESHOLD,
    dp_dn_risk_threshold: float = config.DP_DN_RISK_THRESHOLD,
) -> tuple[pint.Quantity[Any], pint.Quantity[Any]]:
    """Calculate particle-to-nozzle diameter ratio and multiplicative factor.

    When dp/Dn < obstruction threshold: particles pass easily (reduced clogging)
    When dp/Dn > risk threshold: geometric obstruction increases clogging

    Args:
        particle_diameter_um: Particle diameter as pint Quantity.
        nozzle_diameter_mm: Nozzle diameter as pint Quantity.
        dp_dn_obstruction_threshold: Threshold below which particles pass easily.
        dp_dn_risk_threshold: Threshold above which obstruction increases clogging.

    Returns:
        ratio: dp/Dn particle-to-nozzle diameter ratio (dimensionless).
        factor: Clogging modifier factor (dimensionless).

    Raises:
        DimensionalityError: If any input has incompatible units.
    """
    dp = particle_diameter_um.to('m')
    dn = nozzle_diameter_mm.to('m')

    ratio = dp / np.maximum(dn, ureg.Quantity(1e-12, 'm'))  # type: ignore

    # Factor increases with ratio to represent geometric bridging risk
    factor = np.where(
        ratio < dp_dn_obstruction_threshold,
        0.5 + 0.5 * (ratio / dp_dn_obstruction_threshold),
        1.0 + 0.8 * np.log1p(ratio / dp_dn_obstruction_threshold),
    )

    return ratio, np.clip(factor, 0.5, 2.5)


def calculate_velocity_shear_factor(
    velocity_m_s: pint.Quantity[Any],
    velocity_shear_threshold: pint.Quantity[
        Any
    ] = config.PhysicsConstants.VELOCITY_SHEAR_THRESHOLD,
    min_factor: float = 0.25,
) -> pint.Quantity[Any]:
    """Calculate multiplicative factor based on velocity shear threshold.

    Above velocity_shear_threshold (default 8 m/s), shear stress prevents
    deposition. Uses smooth transition to avoid discontinuity.

    Note: For typical irrigation pressures (100-400 kPa), velocities range
    12-24 m/s, which all exceed the 8 m/s threshold. This function therefore
    returns values in the high-velocity regime where self-cleaning dominates.
    The factor provides decreasing protection as velocity increases further
    (diminishing returns on cleaning).

    Args:
        velocity_m_s: Velocity as pint Quantity.
        velocity_shear_threshold: Velocity threshold as pint Quantity.
        min_factor: Minimum factor value (dimensionless).

    Returns:
        Velocity shear factor as pint Quantity (dimensionless).

    Raises:
        DimensionalityError: If velocity has incompatible units.
    """
    v = velocity_m_s.to('m/s')
    v_thresh = velocity_shear_threshold.to('m/s')

    # Below threshold: rapid increase in clogging risk
    # Above threshold: gradual decrease (high-velocity regime provides
    # diminishing additional benefit beyond the threshold)
    ratio = v / v_thresh

    # Linear rise below threshold, gentle logarithmic decline above
    # At ratio=1 (v=8): factor ~1.0
    # At ratio=1.5 (v=12): factor ~0.94
    # At ratio=3 (v=24): factor ~0.84
    factor = np.where(
        ratio < 1.0,
        0.5 + 0.5 * ratio,  # Rise from 0.5 to 1.0 as v approaches threshold
        1.0 - 0.15 * np.log1p(ratio - 1.0),  # Gradual decline above threshold
    )

    # Ensure minimum factor
    factor = np.maximum(factor, min_factor)

    return ureg.Quantity(factor, 'dimensionless')


def calculate_shields_critical_velocity(
    particle_diameter_um: pint.Quantity[Any],
    shields_parameter: float = 0.045,
    rho_particle: pint.Quantity[Any] = config.PhysicsConstants.RHO_SEDIMENT,
    rho_fluid: pint.Quantity[Any] = config.PhysicsConstants.RHO_WATER,
) -> pint.Quantity[Any]:
    """Calculate critical velocity from Shields criterion.

    Derives the critical flow velocity at which sediment particles begin
    to move (incipient motion). Uses the Shields criterion to determine
    critical shear stress, then converts to critical velocity using
    the relationship between shear velocity and average velocity for
    turbulent pipe flow.

    The Shields criterion states:
        τ_c = θ_c * (ρ_s - ρ_w) * g * d_p

    where θ_c is the Shields parameter (typically 0.03-0.06 for
    non-cohesive sediments). The critical shear velocity is:
        u*_c = sqrt(τ_c / ρ_w)

    For turbulent pipe flow, the average velocity relates to shear
    velocity through the Darcy-Weisbach friction factor. For smooth
    pipes at high Reynolds numbers:
        V ≈ u* * sqrt(8/f)
    where f ≈ 0.02-0.04 for typical irrigation pipes.

    Args:
        particle_diameter_um: Particle diameter as pint Quantity.
        shields_parameter: Shields parameter (dimensionless).
            Default 0.045 for sand-sized particles.
        rho_particle: Particle density as pint Quantity. Defaults to sediment.
        rho_fluid: Fluid density as pint Quantity. Defaults to water.

    Returns:
        Critical velocity as pint Quantity in m/s.

    Raises:
        DimensionalityError: If any input has incompatible units.
    """
    # Critical shear stress from Shields criterion
    # Convert particle diameter to meters for consistent units
    dp = particle_diameter_um.to('m')
    tau_c = (
        shields_parameter * (rho_particle - rho_fluid) * config.PhysicsConstants.g * dp
    )

    # Critical shear velocity
    u_star_c = np.sqrt(tau_c / rho_fluid)

    # Convert to average velocity for turbulent pipe flow
    # Using Darcy-Weisbach: V = u* * sqrt(8/f)
    # Typical friction factor for smooth irrigation pipes: f ≈ 0.03
    friction_factor = 0.03
    v_c = u_star_c * np.sqrt(8.0 / friction_factor)

    return v_c.to('m/s')


def calculate_settling_velocity_and_factor(
    particle_diameter_um: pint.Quantity[Any],
    velocity_m_s: pint.Quantity[Any],
    settling_ratio_critical: float = config.SETTLING_VELOCITY_RATIO_CRITICAL,
    rho_particle: pint.Quantity[Any] = config.PhysicsConstants.RHO_SEDIMENT,
    rho_fluid: pint.Quantity[Any] = config.PhysicsConstants.RHO_WATER,
    nu_fluid: pint.Quantity[Any] = config.PhysicsConstants.NU_WATER,
) -> tuple[pint.Quantity[Any], pint.Quantity[Any]]:
    """Compute terminal settling velocity and multiplicative factor.

    Uses the Cheng (1997) formula for settling velocity.

    Args:
        particle_diameter_um: Particle diameter as pint Quantity.
        velocity_m_s: Flow velocity as pint Quantity.
        settling_ratio_critical: Critical ws/V ratio threshold (dimensionless).
        rho_particle: Particle density as pint Quantity. Defaults to sediment.
        rho_fluid: Fluid density as pint Quantity. Defaults to water.
        nu_fluid: Kinematic viscosity as pint Quantity. Defaults to water.

    Returns:
        settling_velocity: Terminal settling velocity (m/s).
        settling_velocity_factor: Factor based on ws/V ratio (dimensionless).

    Raises:
        DimensionalityError: If any input has incompatible units.
    """
    dp = particle_diameter_um.to('m')
    v = velocity_m_s.to('m/s')

    R = (rho_particle - rho_fluid) / rho_fluid

    c1 = 18
    c2 = 1.0

    settling_velocity = (R * config.PhysicsConstants.g * (dp**2)) / (
        c1 * nu_fluid + np.sqrt(0.75 * c2 * R * config.PhysicsConstants.g * (dp**3))
    )

    velocity_safe = np.where(  # type: ignore
        v <= ureg.Quantity(0, 'm/s'),
        ureg.Quantity(1e-10, 'm/s'),
        v,
    )
    ratio = (settling_velocity / velocity_safe).to('dimensionless')

    # For high settling velocity relative to flow, risk increases
    # Factor >= 1.0 to amplify risk
    factor = np.where(
        ratio < settling_ratio_critical,
        1.0 - 0.5 * (1 - ratio / settling_ratio_critical),  # 0.5 to 1.0 range
        1.0 + 0.2 * np.log1p(ratio / settling_ratio_critical),  # Increases above 1.0
    )

    if np.any(v <= ureg.Quantity(0, 'm/s')):
        settling_velocity_factor = ureg.Quantity(np.array(1.0), 'dimensionless')
    else:
        settling_velocity_factor = ureg.Quantity(
            np.clip(factor, 0.5, 2.0), 'dimensionless'
        )

    return ureg.Quantity(settling_velocity.magnitude, 'm/s'), settling_velocity_factor
