"""Orchestration functions for the nozzle clogging simulation pipeline.

Provides the core pipeline stages: physics computation and probability
classification. Each stage accepts a validated DataFrame, adds new columns,
and returns a DataFrame conforming to the next schema level.
"""

import math
from typing import cast

import pandas as pd
import pandera.pandas as pa
import pint
import pint_pandas
from beartype.typing import Any, Callable
from pandera.typing import DataFrame
from tqdm import trange

from nozzle_clogging.physics import (
    calculate_dp_dn_ratio_and_factor,
    calculate_settling_velocity_and_factor,
    calculate_stokes_factor,
    calculate_stokes_number,
    calculate_velocity_from_pressure,
    calculate_velocity_shear_factor,
)
from nozzle_clogging.probability import (
    calculate_clogging_probability,
    classify_clogging_risk,
)
from nozzle_clogging.schemas import (
    PhysicsComputedSchema,
    SimulationInputSchema,
    SimulationOutputSchema,
)

__all__ = [
    'to_quantity',
    'to_pint_array',
    'compute_physics',
    'compute_and_classify_clogging_probability',
    'run_batched',
]


def to_quantity(series: pd.Series) -> pint.Quantity[Any]:
    """Extract a pint.Quantity from a PintArray-backed Series."""
    if isinstance(series, pd.Series) and hasattr(series, 'pint'):
        return series.pint.quantity
    raise TypeError(f'Expected PintArray Series, got {type(series)}')


def to_pint_array(
    values: pint.Quantity[Any], column_name: str
) -> pint_pandas.PintArray:
    """Wrap a pint.Quantity in a PintArray with the correct unit for the column."""
    unit_str = str(values.units)
    return pint_pandas.PintArray(
        values.to(unit_str).magnitude, dtype=f'pint[{unit_str}]'
    )


@pa.check_types
def compute_physics(
    df: DataFrame[SimulationInputSchema],
) -> DataFrame[PhysicsComputedSchema]:
    """Compute physics parameters for simulation inputs.

    Derives velocity from pressure, then computes all intermediate
    physics quantities (Stokes number, settling velocity, etc.).

    Args:
        df: A DataFrame adhering to :class:`SimulationInputSchema`.

    Returns:
        DataFrame with added physics columns,
        adhering to :class:`PhysicsComputedSchema`.

    Raises:
        SchemaError: If input or output violates schema constraints.
        BeartypeCallHintViolation: If input is not a pandas DataFrame.
    """
    df = df.copy()

    velocity = calculate_velocity_from_pressure(to_quantity(df['pressure_kPa']))
    df['velocity_m_s'] = to_pint_array(velocity, 'velocity_m_s')

    df['stokes_number'] = to_pint_array(
        calculate_stokes_number(
            to_quantity(df['particle_diameter_um']),
            to_quantity(df['velocity_m_s']),
            to_quantity(df['nozzle_diameter_mm']),
        ),
        'stokes_number',
    )

    df['stokes_factor'] = to_pint_array(
        calculate_stokes_factor(
            to_quantity(df['velocity_m_s']),
            to_quantity(df['stokes_number']),
        ),
        'stokes_factor',
    )

    (dp_dn_ratio, dp_dn_factor) = calculate_dp_dn_ratio_and_factor(
        to_quantity(df['particle_diameter_um']),
        to_quantity(df['nozzle_diameter_mm']),
    )
    df['dp_dn_ratio'] = to_pint_array(dp_dn_ratio, 'dp_dn_ratio')
    df['dp_dn_factor'] = to_pint_array(dp_dn_factor, 'dp_dn_factor')

    df['velocity_shear_factor'] = to_pint_array(
        calculate_velocity_shear_factor(to_quantity(df['velocity_m_s'])),
        'velocity_shear_factor',
    )

    (settling_velocity, settling_velocity_factor) = (
        calculate_settling_velocity_and_factor(
            to_quantity(df['particle_diameter_um']),
            to_quantity(df['velocity_m_s']),
        )
    )
    df['settling_velocity'] = to_pint_array(settling_velocity, 'settling_velocity')
    df['settling_velocity_factor'] = to_pint_array(
        settling_velocity_factor, 'settling_velocity_factor'
    )

    return cast(DataFrame[PhysicsComputedSchema], df)


@pa.check_types
def compute_and_classify_clogging_probability(
    df: DataFrame[PhysicsComputedSchema],
) -> DataFrame[SimulationOutputSchema]:
    """Compute clogging probability and risk classification.

    Args:
        df: A DataFrame adhering to :class:`PhysicsComputedSchema`.

    Returns:
        DataFrame with added clogging probability columns,
        adhering to :class:`SimulationOutputSchema`.

    Raises:
        SchemaError: If input or output violates schema constraints.
        BeartypeCallHintViolation: If input is not a pandas DataFrame.
    """
    (
        volume_fraction,
        X_base,
        physical_factor,
        X,
        clogging_probability,
    ) = calculate_clogging_probability(df)

    df['volume_fraction'] = to_pint_array(volume_fraction, 'volume_fraction')
    df['X_base'] = to_pint_array(X_base, 'X_base')
    df['physical_factor'] = to_pint_array(physical_factor, 'physical_factor')
    df['X'] = to_pint_array(X, 'X')
    df['clogging_probability'] = to_pint_array(
        clogging_probability, 'clogging_probability'
    )
    df['clogging_risk'] = classify_clogging_risk(clogging_probability)

    return cast(DataFrame[SimulationOutputSchema], df)


def run_batched(
    total_samples: int,
    chunk_size: int,
    seed: int,
    generate_fn: Callable[[int, int], DataFrame[SimulationInputSchema]],
) -> DataFrame[SimulationOutputSchema]:
    """Run a batched simulation pipeline.

    Shared batching logic used by both `run_simulation()` and
    `SimulationPipeline.run()`.

    Args:
        total_samples: Total number of Monte Carlo samples.
        chunk_size: Number of samples per batch.
        seed: Random seed for reproducibility.
        generate_fn: Callable(batch_n, batch_seed) returning a
            DataFrame[SimulationInputSchema].

    Returns:
        Concatenated results DataFrame.
    """
    results: list[pd.DataFrame] = []
    batches: int = math.ceil(total_samples / chunk_size)

    for i in trange(batches):
        n = min(chunk_size, total_samples - i * chunk_size)
        df: Any = generate_fn(n, seed + i)
        df = compute_physics(df)
        df = compute_and_classify_clogging_probability(df)
        results.append(df)
    return cast(
        DataFrame[SimulationOutputSchema], pd.concat(results, ignore_index=True)
    )
