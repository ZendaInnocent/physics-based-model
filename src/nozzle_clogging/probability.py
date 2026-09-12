import numpy as np
import pandas as pd
import pint
from beartype.typing import Any
from pandera.typing import DataFrame
from scipy import special

from nozzle_clogging import config
from nozzle_clogging.physics import calculate_physical_modifiers
from nozzle_clogging.schemas import PhysicsComputedSchema, SimulationOutputSchema
from nozzle_clogging.units import ureg


def calculate_clogging_probability(
    df: DataFrame[PhysicsComputedSchema],
    logistic_scale: float = config.LOGISTIC_SCALE,
    centering_offset: float = config.CENTERING_OFFSET,
    apply_physical_constraints: bool = True,
) -> tuple[
    pint.Quantity[Any],
    pint.Quantity[Any],
    pint.Quantity[Any],
    pint.Quantity[Any],
    pint.Quantity[Any],
]:
    """Compute the probability of sprinkler nozzle clogging using a physics-informed,
    dimensionless clogging potential model.

    This model estimates a dimensionless clogging potential index (X) based on:
    - sediment concentration (TSS),
    - particle size,
    - nozzle geometry,
    - flow velocity,
    - operating duration,
    - and physical modifiers such as particle inertia, gravitational settling,
      geometric ratio, and shear self-cleaning.

    The potential is then converted into a probability using a logistic function.

    Core dimensionless clogging potential:

        X = φ x (dp/Dn)^2 x (V x t / Dn)

    where:
        φ        = particle volume fraction (dimensionless)
        dp/Dn    = particle-to-nozzle diameter ratio (dimensionless)
        V x t/Dn = residence term capturing the relative exposure of particles
                   to flow (dimensionless)

    Physical modifiers are applied multiplicatively:

        X_final = X x Φ

    where Φ = f(dp/Dn) x f(Stk) x f(ws/V) x f(V_shear), all dimensionless.

    The final clogging probability is:

        Pc = 1 / (1 + exp(-logistic_scale * (X_final - offset)))

    Args:
        df: A DataFrame adhering to :class:`PhysicsComputedSchema` containing
            all required input and physics columns.
        logistic_scale: Scale factor controlling the magnitude of X before
            logistic transformation. Defaults to config.LOGISTIC_SCALE.
        centering_offset: Offset to shift the logistic curve.
            Defaults to config.CENTERING_OFFSET.
        apply_physical_constraints: If True, apply all physical modifiers.
            Default True.

    Returns:
        Tuple of (volume_fraction, X_base, physical_factor, X, Pc) as
        pint Quantities.

    Raises:
        SchemaError: If input DataFrame violates PhysicsComputedSchema constraints.
    """
    tss = df['TSS_mg_L'].pint.quantity
    nozzle_diameter = df['nozzle_diameter_mm'].pint.quantity
    velocity = df['velocity_m_s'].pint.quantity
    duration = df['duration_hrs'].pint.quantity
    stokes_factor = df['stokes_factor'].pint.quantity
    dp_dn_ratio = df['dp_dn_ratio'].pint.quantity
    dp_dn_factor = df['dp_dn_factor'].pint.quantity
    velocity_shear_factor = df['velocity_shear_factor'].pint.quantity
    settling_velocity_factor = df['settling_velocity_factor'].pint.quantity

    # Volume fraction: φ = TSS × 10⁻³ / ρ_s (dimensionless)
    # TSS is in mg/L -> convert to kg/m³: 1 mg/L = 1 g/m³ = 0.001 kg/m³
    # So: φ = (TSS in mg/L × 0.001) / (ρ_s in kg/m³)
    # Equivalent: φ = TSS.to('kg/m³') / RHO_SEDIMENT
    tss_kg_m3 = tss.to('kg/m³')
    volume_fraction = tss_kg_m3 / config.PhysicsConstants.RHO_SEDIMENT

    # Convert to SI units for dimensionless residence term
    # velocity: m/s -> already correct
    # duration: hours -> seconds
    # nozzle_diameter: mm -> meters
    safe_velocity = np.maximum(velocity, ureg.Quantity(0.01, 'm/s'))  # type: ignore
    duration_s = duration.to('s')
    nozzle_diameter_m = nozzle_diameter.to('m')
    residence_term = (safe_velocity * duration_s) / nozzle_diameter_m
    X_base = volume_fraction * (dp_dn_ratio**2) * residence_term

    if apply_physical_constraints:
        physical_factor = calculate_physical_modifiers(
            dp_dn_factor, stokes_factor, velocity_shear_factor, settling_velocity_factor
        )
        X = X_base * physical_factor
    else:
        physical_factor = ureg.Quantity(1.0, 'dimensionless')
        X = X_base

    X = np.clip(X, 0, 50)

    Pc = special.expit(logistic_scale * (X.magnitude - centering_offset))
    Pc = ureg.Quantity(np.clip(Pc, 0.0, 1.0), 'dimensionless')

    return volume_fraction, X_base, physical_factor, X, Pc


def classify_clogging_risk(
    clogging_probability: pint.Quantity[Any],
) -> np.ndarray:
    """Classify clogging probability into categorical risk levels.

    Args:
        clogging_probability: Clogging probability (0-1) as pint Quantity.

    Returns:
        Array of risk category strings.

    Raises:
        DimensionalityError: If input has incompatible units.
    """
    arr = np.atleast_1d(
        np.asarray(clogging_probability.to('dimensionless').magnitude, dtype=np.float64)
    )

    arr = np.clip(arr, 0.0, 1.0)
    bins = np.array([config.RISK_LOW_THRESHOLD, config.RISK_MODERATE_THRESHOLD])
    labels = np.array(config.RISK_LEVELS, dtype=object)
    idx = np.digitize(arr, bins)
    return labels[idx]


def calculate_risk_proportions(
    df: DataFrame[PhysicsComputedSchema],
    logistic_scale: float = config.LOGISTIC_SCALE,
    centering_offset: float = config.CENTERING_OFFSET,
) -> dict[str, float]:
    """Calculate risk category proportions for given calibration parameters.

    Args:
        df: DataFrame with computed physics columns.
        logistic_scale: Logistic scale parameter (γ). Defaults to config.LOGISTIC_SCALE.
        centering_offset: Center offset parameter. Defaults to config.CENTERING_OFFSET.

    Returns:
        Dict with 'Low', 'Moderate', 'High' proportions (0-1).
    """
    _, _, _, _, Pc = calculate_clogging_probability(
        df, logistic_scale=logistic_scale, centering_offset=centering_offset
    )
    risk_labels = classify_clogging_risk(Pc)
    unique, counts = np.unique(risk_labels, return_counts=True)
    counts_dict = dict(zip(unique, counts / len(risk_labels)))
    return {level: counts_dict.get(level, 0.0) for level in config.RISK_LEVELS}


def run_calibration_sensitivity_sweep(
    df: DataFrame[SimulationOutputSchema],
    logistic_scales: list[float] = config.CALIBRATION_SENSITIVITY_VALUES[
        'logistic_scale'
    ],
    centering_offsets: list[float] = config.CALIBRATION_SENSITIVITY_VALUES[
        'centering_offset'
    ],
) -> pd.DataFrame:
    """Run calibration parameter sweep and compute risk proportions.

    Args:
        df: DataFrame with computed physics columns.
        logistic_scales: List of γ values to sweep. Defaults to config.
        centering_offsets: List of x₀ values to sweep. Defaults to config.

    Returns:
        DataFrame with columns: logistic_scale, centering_offset,
        Low_prop, Moderate_prop, High_prop, Moderate_gte_15pct.
    """

    X = df['X'].pint.quantity.magnitude

    gamma_arr = np.array(logistic_scales)
    x0_arr = np.array(centering_offsets)

    gamma_grid, x0_grid = np.meshgrid(gamma_arr, x0_arr, indexing='ij')

    logits = gamma_arr[:, None, None] * (X[None, None, :] - x0_arr[None, :, None])
    Pc_grid = special.expit(logits)
    Pc_grid = np.clip(Pc_grid, 0.0, 1.0)

    low_thresh = config.RISK_LOW_THRESHOLD
    mod_thresh = config.RISK_MODERATE_THRESHOLD

    is_low = Pc_grid <= low_thresh
    is_moderate = (Pc_grid > low_thresh) & (Pc_grid <= mod_thresh)
    is_high = Pc_grid > mod_thresh

    n_samples = X.shape[0]
    Low_prop = is_low.sum(axis=2) / n_samples
    Moderate_prop = is_moderate.sum(axis=2) / n_samples
    High_prop = is_high.sum(axis=2) / n_samples

    results = pd.DataFrame(
        {
            'logistic_scale': gamma_grid.ravel(),
            'centering_offset': x0_grid.ravel(),
            'Low_prop': Low_prop.ravel(),
            'Moderate_prop': Moderate_prop.ravel(),
            'High_prop': High_prop.ravel(),
            'Moderate_gte_15pct': (Moderate_prop >= 0.15).ravel(),
        }
    )

    return results