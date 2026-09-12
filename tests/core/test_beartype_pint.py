"""Tests for beartype compatibility with pint-pandas extension types.

Verifies that beartype's automatic type checking (via beartype_this_package)
works correctly with pint-pandas DataFrames used throughout the pipeline.

This ensures:
1. @pa.check_types decorated functions accept pint DataFrames without false positives
2. Beartype does not raise BeartypeCallHintViolation for pint extension types
3. Actual type/schema errors are still caught
"""

import numpy as np
import pandas as pd
import pint_pandas
import pytest
from beartype.roar import BeartypeCallHintViolation

from nozzle_clogging.generation import generate_simulation_inputs
from nozzle_clogging.physics import (
    calculate_dp_dn_ratio_and_factor,
    calculate_settling_velocity_and_factor,
    calculate_stokes_factor,
    calculate_stokes_number,
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
from nozzle_clogging.simulation import (
    compute_and_classify_clogging_probability,
    compute_physics,
)
from nozzle_clogging.units import ureg


class TestBeartypePintDataFrameCompatibility:
    """Test that beartype works with pint-pandas DataFrames through @pa.check_types."""

    def test_compute_physics_accepts_pint_dataframe(self) -> None:
        """compute_physics accepts pint DataFrame."""
        df = generate_simulation_inputs(5, seed=42)
        # Verify input has pint dtypes
        assert isinstance(df['TSS_mg_L'].dtype, pint_pandas.PintType)

        # This should NOT raise BeartypeCallHintViolation
        result = compute_physics(df)
        assert isinstance(result, pd.DataFrame)
        assert 'stokes_number' in result.columns

    def test_compute_physics_returns_valid_schema(self) -> None:
        """compute_physics output conforms to PhysicsComputedSchema."""
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        validated = PhysicsComputedSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)

    def test_compute_probability_accepts_pint_dataframe(self) -> None:
        """compute_and_classify_clogging_probability accepts pint DataFrame."""
        df = generate_simulation_inputs(5, seed=42)
        df_physics = compute_physics(df)

        # This should NOT raise BeartypeCallHintViolation
        result = compute_and_classify_clogging_probability(df_physics)
        assert isinstance(result, pd.DataFrame)
        assert 'clogging_probability' in result.columns

    def test_compute_probability_returns_valid_schema(self) -> None:
        """compute_and_classify_clogging_probability output conforms to schema."""
        df = generate_simulation_inputs(5, seed=42)
        df_physics = compute_physics(df)
        result = compute_and_classify_clogging_probability(df_physics)
        validated = SimulationOutputSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)

    def test_full_pipeline_no_beartype_violations(self) -> None:
        """Full pipeline works without beartype errors."""
        df = generate_simulation_inputs(10, seed=42)
        df_physics = compute_physics(df)
        df_output = compute_and_classify_clogging_probability(df_physics)

        assert len(df_output) == 10
        assert 'clogging_risk' in df_output.columns


class TestBeartypePintScalarCompatibility:
    """Test that beartype works with pint Quantity scalars and arrays."""

    def test_stokes_number_with_pint_quantities(self) -> None:
        """calculate_stokes_number accepts pint Quantities."""
        dp = ureg.Quantity(50.0, 'um')
        V = ureg.Quantity(5.0, 'm/s')
        Dn = ureg.Quantity(3.0, 'mm')

        # Should NOT raise BeartypeCallHintViolation
        result = calculate_stokes_number(dp, V, Dn)
        assert result is not None

    def test_stokes_number_with_pint_arrays(self) -> None:
        """calculate_stokes_number should accept pint Quantity arrays."""
        dp = ureg.Quantity(np.array([50.0, 100.0, 200.0]), 'um')
        V = ureg.Quantity(np.array([3.0, 5.0, 8.0]), 'm/s')
        Dn = ureg.Quantity(np.array([3.0, 4.0, 5.0]), 'mm')

        result = calculate_stokes_number(dp, V, Dn)
        assert len(result) == 3

    def test_stokes_factor_with_pint_quantities(self) -> None:
        """calculate_stokes_factor should accept pint Quantities."""
        V = ureg.Quantity(5.0, 'm/s')
        Stk = ureg.Quantity(0.05, 'dimensionless')

        result = calculate_stokes_factor(V, Stk)
        assert result is not None

    def test_dp_dn_ratio_with_pint_quantities(self) -> None:
        """calculate_dp_dn_ratio_and_factor should accept pint Quantities."""
        dp = ureg.Quantity(50.0, 'um')
        Dn = ureg.Quantity(3.0, 'mm')

        ratio, factor = calculate_dp_dn_ratio_and_factor(dp, Dn)
        assert ratio is not None
        assert factor is not None

    def test_velocity_shear_with_pint_quantity(self) -> None:
        """calculate_velocity_shear_factor should accept pint Quantity."""
        V = ureg.Quantity(5.0, 'm/s')

        result = calculate_velocity_shear_factor(V)
        assert result is not None

    def test_settling_velocity_with_pint_quantities(self) -> None:
        """calculate_settling_velocity_and_factor should accept pint Quantities."""
        dp = ureg.Quantity(100.0, 'um')
        V = ureg.Quantity(3.0, 'm/s')

        vel, factor = calculate_settling_velocity_and_factor(dp, V)
        assert vel is not None
        assert factor is not None

    def test_clogging_probability_with_pint_quantities(self) -> None:
        """calculate_clogging_probability should accept schema DataFrame."""
        df = generate_simulation_inputs(5, seed=42)
        df_physics = compute_physics(df)

        result = calculate_clogging_probability(df_physics)
        assert len(result) == 5

    def test_classify_risk_with_pint_quantity(self) -> None:
        """classify_clogging_risk should accept pint Quantity."""
        prob = ureg.Quantity(0.45, 'dimensionless')

        result = classify_clogging_risk(prob)
        assert result is not None

    def test_classify_risk_with_pint_array(self) -> None:
        """classify_clogging_risk should accept pint Quantity array."""
        prob = ureg.Quantity(np.array([0.1, 0.4, 0.7]), 'dimensionless')

        result = classify_clogging_risk(prob)
        assert len(result) == 3


class TestBeartypeCatchesActualErrors:
    """Test that beartype still catches actual type errors."""

    def test_non_dataframe_rejected(self) -> None:
        """Non-DataFrame to compute_physics raises error."""
        with pytest.raises((BeartypeCallHintViolation, AttributeError, TypeError)):
            compute_physics('not a dataframe')

    def test_non_dataframe_rejected_for_probability(self) -> None:
        """Non-DataFrame to compute_probability raises error."""
        with pytest.raises((BeartypeCallHintViolation, AttributeError, TypeError)):
            compute_and_classify_clogging_probability('not a dataframe')

    def test_list_rejected_for_compute_physics(self) -> None:
        """Passing a list to compute_physics should raise error."""
        with pytest.raises((BeartypeCallHintViolation, AttributeError, TypeError)):
            compute_physics([1, 2, 3])


class TestPintDataFrameSchemaValidation:
    """Test that schema validation works correctly with pint DataFrames."""

    def test_pint_dataframe_validates_as_input_schema(self) -> None:
        """Pint DataFrame should validate against SimulationInputSchema."""
        df = generate_simulation_inputs(10, seed=42)
        validated = SimulationInputSchema.validate(df)
        assert isinstance(validated, pd.DataFrame)

    def test_pint_dataframe_preserves_units_after_validation(self) -> None:
        """Pint DataFrame units should be preserved after schema validation.

        With PanderaPintDtype and coerce=False, pint-pandas unit metadata
        is preserved through schema validation.
        """
        df = generate_simulation_inputs(5, seed=42)
        validated = SimulationInputSchema.validate(df)

        # Units should be preserved
        assert isinstance(validated['TSS_mg_L'].dtype, pint_pandas.PintType)
        assert str(validated['TSS_mg_L'].dtype.units) == 'milligram / liter'

    def test_schema_catches_missing_columns(self) -> None:
        """Schema validation should catch missing columns."""
        df = pd.DataFrame({'TSS_mg_L': [1, 2, 3]})
        with pytest.raises(Exception):
            SimulationInputSchema.validate(df)

    def test_schema_catches_invalid_values(self) -> None:
        """Schema validation should catch values outside constraints."""
        df = pd.DataFrame(
            {
                'TSS_mg_L': [-1.0, 2.0, 3.0],
                'pressure_kPa': [100.0, 200.0, 300.0],
                'nozzle_diameter_mm': [2.0, 3.0, 4.0],
                'duration_hrs': [1.0, 2.0, 3.0],
                'particle_diameter_um': [50.0, 60.0, 70.0],
                'particle_size_range': ['Fine', 'Medium', 'Coarse'],
            }
        )
        with pytest.raises(Exception):
            SimulationInputSchema.validate(df)
