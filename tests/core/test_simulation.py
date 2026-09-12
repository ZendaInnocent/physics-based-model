import numpy as np
import pandas as pd
import pint_pandas
from hypothesis import given, settings

from nozzle_clogging.schemas import (
    PhysicsComputedSchema,
    SimulationOutputSchema,
)
from nozzle_clogging.simulation import (
    compute_and_classify_clogging_probability,
    compute_physics,
    run_simulation,
)
from tests.core.conftest import simulation_input_df


class TestSimulationWithProperties:
    @given(simulation_input_df())
    @settings(deadline=None)
    def test_generate_simulation_inputs_returns_valid_schema(
        self, sample_df: pd.DataFrame
    ) -> None:
        """
        Property-based test: simulation_input_df strategy should produce DataFrames
        with pint extension dtypes that match the expected schema.
        """
        assert isinstance(sample_df, pd.DataFrame)
        for col in [
            'TSS_mg_L',
            'pressure_kPa',
            'nozzle_diameter_mm',
            'duration_hrs',
            'particle_diameter_um',
        ]:
            assert isinstance(sample_df[col].dtype, pint_pandas.PintType), (
                f'{col} should be pint dtype, got {sample_df[col].dtype}'
            )

    @given(simulation_input_df())
    @settings(deadline=None)
    def test_compute_physics_preserves_input_and_adds_physics_columns(
        self, sample_df: pd.DataFrame
    ) -> None:
        """
        Property-based test: compute_physics should preserve input columns
        and add expected physics columns, accepting pint-pandas input DataFrames.
        """
        result = compute_physics(sample_df)

        for col in sample_df.columns:
            assert col in result.columns

        expected_physics_cols = [
            'velocity_m_s',
            'stokes_number',
            'stokes_factor',
            'dp_dn_ratio',
            'dp_dn_factor',
            'velocity_shear_factor',
            'settling_velocity',
            'settling_velocity_factor',
        ]
        for col in expected_physics_cols:
            assert col in result.columns

        assert len(result) == len(sample_df)

        for col in [
            'TSS_mg_L',
            'pressure_kPa',
            'nozzle_diameter_mm',
            'duration_hrs',
            'particle_diameter_um',
        ]:
            expected_magnitudes = sample_df[col].pint.magnitude
            actual = result[col]
            actual_magnitudes = (
                actual.pint.magnitude
                if isinstance(actual.dtype, pint_pandas.PintType)
                else actual.values
            )
            np.testing.assert_allclose(
                actual_magnitudes,
                expected_magnitudes,
                rtol=1e-10,
            )

        validated_result = PhysicsComputedSchema.validate(result)
        assert isinstance(validated_result, pd.DataFrame)

    @given(simulation_input_df())
    @settings(deadline=None)
    def test_compute_and_classify_clogging_probability_adds_expected_columns_pint(
        self, sample_df: pd.DataFrame
    ) -> None:
        """
        Property-based test: compute_and_classify_clogging_probability should
        preserve all input and physics columns, and add clogging probability columns.
        """
        physics_df = compute_physics(sample_df)
        result = compute_and_classify_clogging_probability(physics_df)

        for col in sample_df.columns:
            assert col in result.columns

        physics_cols = [
            'velocity_m_s',
            'stokes_number',
            'stokes_factor',
            'dp_dn_ratio',
            'dp_dn_factor',
            'velocity_shear_factor',
            'settling_velocity',
            'settling_velocity_factor',
        ]
        for col in physics_cols:
            assert col in result.columns

        expected_clogging_cols = [
            'volume_fraction',
            'X_base',
            'physical_factor',
            'X',
            'clogging_probability',
            'clogging_risk',
        ]
        for col in expected_clogging_cols:
            assert col in result.columns

        assert len(result) == len(sample_df)

        prob_col = result['clogging_probability']
        prob_values = (
            prob_col.pint.magnitude
            if isinstance(prob_col.dtype, pint_pandas.PintType)
            else prob_col.values
        )
        assert (prob_values >= 0).all()
        assert (prob_values <= 1).all()
        assert result['clogging_risk'].isin(['Low', 'Moderate', 'High']).all()

        validated_result = SimulationOutputSchema.validate(result)
        assert isinstance(validated_result, pd.DataFrame)


class TestRunSimulation:
    def test_default_simulation(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=50)
        assert df is not None
        assert len(df) == 100
        validated_df = SimulationOutputSchema.validate(df)
        assert isinstance(validated_df, pd.DataFrame)

    def test_default_sample_count_is_20k(self) -> None:
        """Verify default total_samples is 20,000 as per convergence analysis."""
        import inspect

        sig = inspect.signature(run_simulation)
        default_total_samples = sig.parameters['total_samples'].default
        assert default_total_samples == 20_000, (
            f'Default total_samples should be 20,000 based on convergence analysis, '
            f'got {default_total_samples}'
        )

    def test_simulation_columns(self) -> None:
        df = run_simulation(total_samples=10, chunk_size=10)
        expected_cols = [
            'TSS_mg_L',
            'pressure_kPa',
            'nozzle_diameter_mm',
            'duration_hrs',
            'particle_diameter_um',
            'particle_size_range',
            'velocity_m_s',
            'stokes_number',
            'stokes_factor',
            'dp_dn_ratio',
            'dp_dn_factor',
            'velocity_shear_factor',
            'settling_velocity',
            'settling_velocity_factor',
            'volume_fraction',
            'X_base',
            'physical_factor',
            'X',
            'clogging_probability',
            'clogging_risk',
        ]
        assert list(df.columns) == expected_cols
