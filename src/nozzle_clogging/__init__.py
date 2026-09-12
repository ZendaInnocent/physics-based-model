"""
Nozzle Clogging Model

A physics-based Monte Carlo simulation framework for assessing
sprinkler nozzle clogging risk from sediment-laden irrigation water.

Key features:
- Dimensionless clogging potential index from Buckingham π analysis
- Latin Hypercube sampling (20,000 samples)
- Four physical modifiers: Stokes, dp/Dn, velocity shear, settling
- Logistic probability mapping (γ=1.0, x₀=3.0 for manuscript alignment)
- Saltelli Sobol sensitivity analysis
- Full unit-aware calculations with pint
- Schema validation with pandera + beartype
"""

from nozzle_clogging.config import (
    LOGISTIC_SCALE,
    CENTERING_OFFSET,
    PARAM_RANGES,
    PARTICLE_SIZE_RANGES,
    PhysicsConstants,
    RANDOM_SEED,
)
from nozzle_clogging.generation import (
    compute_lognormal_params,
    generate_lhs_samples,
    generate_simulation_inputs,
    generate_vectorized_lognormal_particle_sizes,
)
from nozzle_clogging.orchestration import (
    compute_and_classify_clogging_probability,
    compute_physics,
    run_batched,
    to_pint_array,
    to_quantity,
)
from nozzle_clogging.physics import (
    calculate_dp_dn_ratio_and_factor,
    calculate_physical_modifiers,
    calculate_settling_velocity_and_factor,
    calculate_shields_critical_velocity,
    calculate_stokes_factor,
    calculate_stokes_number,
    calculate_velocity_from_pressure,
    calculate_velocity_shear_factor,
)
from nozzle_clogging.probability import (
    calculate_clogging_probability,
    calculate_risk_proportions,
    classify_clogging_risk,
    run_calibration_sensitivity_sweep,
)
from nozzle_clogging.schemas import (
    PhysicsComputedSchema,
    SimulationInputSchema,
    SimulationOutputSchema,
)
from nozzle_clogging.simulation import run_simulation
from nozzle_clogging.units import ureg

__all__ = [
    # Config
    'LOGISTIC_SCALE',
    'CENTERING_OFFSET',
    'PARAM_RANGES',
    'PARTICLE_SIZE_RANGES',
    'PhysicsConstants',
    'RANDOM_SEED',
    # Units
    'ureg',
    # Generation
    'compute_lognormal_params',
    'generate_lhs_samples',
    'generate_simulation_inputs',
    'generate_vectorized_lognormal_particle_sizes',
    # Physics
    'calculate_dp_dn_ratio_and_factor',
    'calculate_physical_modifiers',
    'calculate_settling_velocity_and_factor',
    'calculate_shields_critical_velocity',
    'calculate_stokes_factor',
    'calculate_stokes_number',
    'calculate_velocity_from_pressure',
    'calculate_velocity_shear_factor',
    # Probability
    'calculate_clogging_probability',
    'calculate_risk_proportions',
    'classify_clogging_risk',
    'run_calibration_sensitivity_sweep',
    # Schemas
    'PhysicsComputedSchema',
    'SimulationInputSchema',
    'SimulationOutputSchema',
    # Orchestration
    'compute_and_classify_clogging_probability',
    'compute_physics',
    'run_batched',
    'to_pint_array',
    'to_quantity',
    # Simulation
    'run_simulation',
]

__version__ = '0.2.0'