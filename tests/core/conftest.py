import numpy as np
import pandas as pd
import pint_pandas
from hypothesis import strategies as st
from hypothesis.strategies._internal.core import DrawFn


@st.composite
def simulation_input_df(
    draw: DrawFn, min_size: int = 1, max_size: int = 20
) -> pd.DataFrame:
    """Generate a DataFrame conforming to SimulationInputSchema with varied inputs.

    Uses Hypothesis to generate random values within the schema constraints,
    creating pint-pandas DataFrames with proper units.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))

    tss = draw(
        st.lists(
            st.floats(
                min_value=1, max_value=999, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    pressure = draw(
        st.lists(
            st.floats(
                min_value=10, max_value=990, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    nozzle_diam = draw(
        st.lists(
            st.floats(
                min_value=0.5, max_value=49, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    duration = draw(
        st.lists(
            st.floats(
                min_value=0.1, max_value=990, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    particle_diam = draw(
        st.lists(
            st.floats(
                min_value=1, max_value=999, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        )
    )
    particle_size_range = draw(
        st.lists(
            st.sampled_from(['Fine', 'Medium', 'Coarse']),
            min_size=size,
            max_size=size,
        )
    )

    return pd.DataFrame(
        {
            'TSS_mg_L': pint_pandas.PintArray(
                np.array(tss), dtype='pint[milligram / liter]'
            ),
            'pressure_kPa': pint_pandas.PintArray(
                np.array(pressure), dtype='pint[kilopascal]'
            ),
            'nozzle_diameter_mm': pint_pandas.PintArray(
                np.array(nozzle_diam), dtype='pint[millimeter]'
            ),
            'duration_hrs': pint_pandas.PintArray(
                np.array(duration), dtype='pint[hour]'
            ),
            'particle_diameter_um': pint_pandas.PintArray(
                np.array(particle_diam), dtype='pint[micrometer]'
            ),
            'particle_size_range': particle_size_range,
        }
    )


@st.composite
def physics_input_df(
    draw: DrawFn, min_size: int = 1, max_size: int = 20
) -> pd.DataFrame:
    """Generate a DataFrame with physics-computed columns.

    Extends simulation_input_df by computing physics columns through
    the compute_physics pipeline.
    """
    from nozzle_clogging.simulation import compute_physics

    input_df = draw(simulation_input_df(min_size=min_size, max_size=max_size))
    return compute_physics(input_df)


@st.composite
def simulation_output_df(
    draw: DrawFn, min_size: int = 1, max_size: int = 20
) -> pd.DataFrame:
    """Generate a DataFrame with full simulation output.

    Extends physics_input_df by computing clogging probability.
    """
    from nozzle_clogging.simulation import (
        compute_and_classify_clogging_probability,
        compute_physics,
    )

    input_df = draw(simulation_input_df(min_size=min_size, max_size=max_size))
    physics_df = compute_physics(input_df)
    return compute_and_classify_clogging_probability(physics_df)
