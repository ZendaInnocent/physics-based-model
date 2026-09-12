from __future__ import annotations

from beartype.typing import Any
from pandas import CategoricalDtype

from nozzle_clogging.units import ureg


class PhysicsConstants:
    """Physical constants for the nozzle-clogging model.

    All constants are pint Quantities with explicit units.
    """

    RHO_WATER = ureg.Quantity(1_000, 'kg/m³')
    RHO_SEDIMENT = ureg.Quantity(2_650, 'kg/m³')
    NU_WATER = ureg.Quantity(1e-6, 'm²/s')
    g = ureg.Quantity(9.81, 'm/s²')
    VELOCITY_SHEAR_THRESHOLD = ureg.Quantity(12.0, 'm/s')  # Manuscript value
    NOZZLE_DIAMETER = ureg.Quantity(5, 'mm')
    PRESSURE = ureg.Quantity(300, 'kPa')


# Simulation parameters
RANDOM_SEED: int = 42

CD: float = 0.85
STOKES_CRITICAL: float = 0.1
DP_DN_OBSTRUCTION_THRESHOLD: float = 0.05
DP_DN_RISK_THRESHOLD: float = 0.14
SETTLING_VELOCITY_RATIO_CRITICAL: float = 0.1

TSS_VALUES: list[int] = [10, 50, 100, 200, 350, 500]  # mg/L

PARTICLE_TYPE = CategoricalDtype(categories=['Fine', 'Medium', 'Coarse'], ordered=True)

PARTICLE_SIZE_RANGES: dict[str, tuple[int, int]] = {
    'Fine': (10, 50),
    'Medium': (50, 150),
    'Coarse': (150, 300),
}  # µm

PRESSURE_VALUES: list[int] = [200, 300, 400]  # kPa

NOZZLE_DIAMETERS: list[int] = [2, 3, 4, 5, 6]  # mm

DURATION_VALUES: list[int] = [2, 4, 8]  # hrs

PARAM_RANGES: dict[str, Any] = {
    'TSS': (10, 500),  # mg/L
    'pressure': (100, 400),  # kPa
    'nozzle_diameter': (1.5, 6),  # mm
    'duration': (0.5, 8),  # hrs
}

RISK_LOW_THRESHOLD: float = 0.30
RISK_MODERATE_THRESHOLD: float = 0.50
RISK_LEVELS: list[str] = ['Low', 'Moderate', 'High']

# Manuscript parameters for alignment
LOGISTIC_SCALE: float = 1.0  # γ
CENTERING_OFFSET: float = 3.0  # x₀

CALIBRATION_SENSITIVITY_VALUES: dict[str, list[float]] = {
    'logistic_scale': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0],
    'centering_offset': [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
}

VELOCITY_THRESHOLD_VALUES: list[float] = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # m/s

# Regime switching for semi-solid set sprinkler systems
EXPOSURE_REGIME_THRESHOLD: float = 8.0  # hours - lateral pipes moved >=8 hours
SELF_CLEANING_FACTOR: float = 0.5  # reduced effective exposure during self-cleaning