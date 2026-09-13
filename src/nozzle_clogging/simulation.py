"""Simulation runner for the nozzle clogging Monte Carlo pipeline.

Provides the batched simulation entry point that chains generation,
physics computation, and probability classification.
"""

import pandera.pandas as pa
from pandera.typing import DataFrame

from nozzle_clogging import config
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
from nozzle_clogging.schemas import (
    SimulationOutputSchema,
)

__all__ = [
    'compute_lognormal_params',
    'generate_lhs_samples',
    'generate_simulation_inputs',
    'generate_vectorized_lognormal_particle_sizes',
    'compute_physics',
    'compute_and_classify_clogging_probability',
    'to_pint_array',
    'to_quantity',
    'run_simulation',
]


@pa.check_types
def run_simulation(
    total_samples: int = 20_000, chunk_size: int = 2_000, seed: int = config.RANDOM_SEED
) -> DataFrame[SimulationOutputSchema]:
    """Run the sprinkler clogging simulation in memory-safe batches.

    Parameters
    ----------
    total_samples : int
        Total number of Monte Carlo samples.
    chunk_size : int
        Samples per batch.
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Combined results adhering to :class:`SimulationOutputSchema`.
    """
    return run_batched(total_samples, chunk_size, seed, generate_simulation_inputs)
