import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from nozzle_clogging.pint_types import PanderaPintDtype


class SimulationInputSchema(pa.DataFrameModel):
    """
    Schema for validated simulation input parameters.

    Each row represents a unique combination of irrigation system
    and environmental parameters for a single Monte Carlo sample.
    Used as input to the physics computation pipeline.

    Pint preservation:
        All numeric columns use :class:`~nozzle_clogging.pint_types.PanderaPintDtype`
        with ``dtype_kwargs={'units': '...'}`` so that pint-pandas unit metadata
        is preserved through schema validation AND the column's units are enforced
        to match the expected unit. Range constraints are enforced via
        ``@pa.dataframe_check`` using ``.pint.magnitude``.
    """

    TSS_mg_L: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'milligram / liter'},
        description='Total suspended solids concentration in mg/L',
    )
    pressure_kPa: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'kilopascal'},
        description='System pressure in kilopascals',
    )
    nozzle_diameter_mm: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'millimeter'},
        description='Nozzle diameter in millimeters',
    )
    duration_hrs: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'hour'},
        description='Exposure duration in hours',
    )
    particle_diameter_um: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'micrometer'},
        description='Particle diameter in micrometers',
    )
    particle_size_range: Series[str] = pa.Field(
        isin=['Fine', 'Medium', 'Coarse'], description='Categorical particle size range'
    )

    @pa.dataframe_check
    def tss_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """TSS must be between 0 and 1000 mg/L."""
        m = df['TSS_mg_L'].pint.magnitude
        return (m >= 0) & (m <= 1000)

    @pa.dataframe_check
    def pressure_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Pressure must be between 0 and 1000 kPa."""
        m = df['pressure_kPa'].pint.magnitude
        return (m >= 0) & (m <= 1000)

    @pa.dataframe_check
    def nozzle_diameter_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Nozzle diameter must be between 0 and 50 mm."""
        m = df['nozzle_diameter_mm'].pint.magnitude
        return (m > 0) & (m <= 50)

    @pa.dataframe_check
    def duration_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Duration must be between 0 and 1000 hours."""
        m = df['duration_hrs'].pint.magnitude
        return (m >= 0) & (m <= 1000)

    @pa.dataframe_check
    def particle_diameter_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Particle diameter must be between 0 and 1000 um."""
        m = df['particle_diameter_um'].pint.magnitude
        return (m > 0) & (m <= 1000)


class PhysicsComputedSchema(SimulationInputSchema):
    """
    Schema for validated physics-computed parameters.

    Each row represents the simulation inputs with physics parameters
    computed but before clogging probability calculation.
    Inherits all input fields and checks from :class:`SimulationInputSchema`.

    Pint preservation:
        All numeric columns use :class:`~nozzle_clogging.pint_types.PanderaPintDtype`
        with ``dtype_kwargs={'units': '...'}`` so that pint-pandas unit metadata
        is preserved through schema validation AND the column's units are enforced
        to match the expected unit.
    """

    velocity_m_s: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'meter / second'},
        description='Flow velocity in m/s',
    )
    stokes_number: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Stokes number (dimensionless)',
    )
    stokes_factor: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Stokes factor (dimensionless)',
    )
    dp_dn_ratio: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Particle-to-nozzle diameter ratio (dimensionless)',
    )
    dp_dn_factor: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Diameter ratio factor (dimensionless)',
    )
    velocity_shear_factor: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Velocity shear factor (dimensionless)',
    )
    settling_velocity: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'meter / second'},
        description='Particle settling velocity in m/s',
    )
    settling_velocity_factor: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Settling velocity factor (dimensionless)',
    )

    @pa.dataframe_check
    def stokes_number_non_negative(cls, df: pd.DataFrame) -> pd.Series:
        """Stokes number must be non-negative."""
        m = df['stokes_number'].pint.magnitude
        return m >= 0

    @pa.dataframe_check
    def stokes_factor_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Stokes factor must be between 0 and 2."""
        m = df['stokes_factor'].pint.magnitude
        return (m >= 0) & (m <= 2)

    @pa.dataframe_check
    def dp_dn_ratio_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """dp/Dn ratio must be between 0 and 2."""
        m = df['dp_dn_ratio'].pint.magnitude
        return (m >= 0) & (m <= 2)

    @pa.dataframe_check
    def dp_dn_factor_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """dp/Dn factor must be between 0 and 2.5."""
        m = df['dp_dn_factor'].pint.magnitude
        return (m >= 0) & (m <= 2.5)

    @pa.dataframe_check
    def velocity_shear_factor_non_negative(cls, df: pd.DataFrame) -> pd.Series:
        """Velocity shear factor must be non-negative."""
        m = df['velocity_shear_factor'].pint.magnitude
        return m >= 0

    @pa.dataframe_check
    def settling_velocity_non_negative(cls, df: pd.DataFrame) -> pd.Series:
        """Settling velocity must be non-negative."""
        m = df['settling_velocity'].pint.magnitude
        return m >= 0

    @pa.dataframe_check
    def settling_velocity_factor_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Settling velocity factor must be between 0 and 2."""
        m = df['settling_velocity_factor'].pint.magnitude
        return (m >= 0) & (m <= 2)


class SimulationOutputSchema(PhysicsComputedSchema):
    """
    Schema for validated simulation results.

    Each row represents the computed physics and clogging probability
    for a single Monte Carlo sample after full processing pipeline.
    Inherits all input, physics fields, and checks from :class:`PhysicsComputedSchema`.

    Pint preservation:
        All numeric columns use :class:`~nozzle_clogging.pint_types.PanderaPintDtype`
        with ``dtype_kwargs={'units': '...'}`` so that pint-pandas unit metadata
        is preserved through schema validation AND the column's units are enforced
        to match the expected unit.
    """

    volume_fraction: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Volume fraction of particles (dimensionless)',
    )
    X_base: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Base clogging parameter (dimensionless)',
    )
    physical_factor: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Physical factor combining multiple effects (dimensionless)',
    )
    X: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Final clogging parameter (dimensionless)',
    )
    clogging_probability: Series[PanderaPintDtype] = pa.Field(
        dtype_kwargs={'units': 'dimensionless'},
        description='Probability of clogging occurrence (0-1)',
    )
    clogging_risk: Series[str] = pa.Field(
        isin=['Low', 'Moderate', 'High'], description='Categorical clogging risk level'
    )

    @pa.dataframe_check
    def volume_fraction_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Volume fraction must be between 0 and 1."""
        m = df['volume_fraction'].pint.magnitude
        return (m >= 0) & (m <= 1)

    @pa.dataframe_check
    def x_base_non_negative(cls, df: pd.DataFrame) -> pd.Series:
        """X_base must be non-negative."""
        m = df['X_base'].pint.magnitude
        return m >= 0

    @pa.dataframe_check
    def physical_factor_non_negative(cls, df: pd.DataFrame) -> pd.Series:
        """Physical factor must be non-negative."""
        m = df['physical_factor'].pint.magnitude
        return m >= 0

    @pa.dataframe_check
    def x_non_negative(cls, df: pd.DataFrame) -> pd.Series:
        """X must be non-negative."""
        m = df['X'].pint.magnitude
        return m >= 0

    @pa.dataframe_check
    def clogging_probability_in_range(cls, df: pd.DataFrame) -> pd.Series:
        """Clogging probability must be between 0 and 1."""
        m = df['clogging_probability'].pint.magnitude
        return (m >= 0) & (m <= 1)