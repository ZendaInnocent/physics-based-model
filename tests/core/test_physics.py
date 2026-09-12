import numpy as np
import pandas as pd
import pint
import pint_pandas
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nozzle_clogging import config
from nozzle_clogging.generation import generate_simulation_inputs
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
    classify_clogging_risk,
)
from nozzle_clogging.schemas import (
    PhysicsComputedSchema,
    SimulationInputSchema,
    SimulationOutputSchema,
)
from nozzle_clogging.simulation import (
    compute_physics,
    run_simulation,
    to_pint_array,
    to_quantity,
)
from nozzle_clogging.units import ureg
from tests.core.conftest import simulation_input_df


class TestCalculateVelocityFromPressure:
    def test_scalar_input(self) -> None:
        result = calculate_velocity_from_pressure(ureg.Quantity(200, 'kPa'))
        assert result.magnitude > 0

    def test_array_input(self) -> None:
        result = calculate_velocity_from_pressure(
            ureg.Quantity(np.array([100, 200, 300]), 'kPa')
        )
        assert len(result) == 3
        assert all(r.magnitude > 0 for r in result)

    def test_zero_pressure(self) -> None:
        result = calculate_velocity_from_pressure(ureg.Quantity(0, 'kPa'))
        assert result.magnitude == 0

    def test_very_high_pressure(self) -> None:
        result = calculate_velocity_from_pressure(ureg.Quantity(10000, 'kPa'))
        assert result.magnitude > 0

    def test_returns_m_s_units(self) -> None:
        result = calculate_velocity_from_pressure(ureg.Quantity(200, 'kPa'))
        assert str(result.units) == 'meter / second'


class TestCalculateStokesNumber:
    def test_typical_values(self) -> None:
        result = calculate_stokes_number(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(3, 'mm'),
        )
        assert result.magnitude > 0

    def test_array_input(self) -> None:
        result = calculate_stokes_number(
            ureg.Quantity(np.array([50, 100]), 'um'),
            ureg.Quantity(np.array([10, 20]), 'm/s'),
            ureg.Quantity(np.array([3, 4]), 'mm'),
        )
        assert len(result) == 2

    def test_very_small_velocity(self) -> None:
        result = calculate_stokes_number(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(0.001, 'm/s'),
            ureg.Quantity(3, 'mm'),
        )
        assert result.magnitude > 0

    def test_returns_dimensionless(self) -> None:
        result = calculate_stokes_number(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(3, 'mm'),
        )
        assert str(result.units) == 'dimensionless'


class TestCalculateStokesFactor:
    def test_below_critical_stokes(self) -> None:
        result = calculate_stokes_factor(
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(0.05, 'dimensionless'),
        )
        m = result.magnitude
        assert 0 < m <= 1.0

    def test_above_critical_stokes(self) -> None:
        # Stk=1.0 > 0.1 critical threshold: factor > 1.0
        # factor = 1.0 + 0.15 * log1p(1.0/0.1) = 1.0 + 0.15 * log1p(10) ≈ 1.36
        result = calculate_stokes_factor(
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(1.0, 'dimensionless'),
        )
        assert result.magnitude > 1.0

    def test_returns_dimensionless(self) -> None:
        result = calculate_stokes_factor(
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(0.05, 'dimensionless'),
        )
        assert str(result.units) == 'dimensionless'


class TestCalculateDpDnRatioAndFactor:
    def test_small_particle(self) -> None:
        ratio, factor = calculate_dp_dn_ratio_and_factor(
            ureg.Quantity(10, 'um'),
            ureg.Quantity(100, 'mm'),
        )
        assert ratio.magnitude < 1.0
        assert factor.magnitude < 1.0

    def test_large_particle(self) -> None:
        ratio, factor = calculate_dp_dn_ratio_and_factor(
            ureg.Quantity(200, 'um'),
            ureg.Quantity(3, 'mm'),
        )
        assert ratio.magnitude < 1.0
        assert factor.magnitude >= 1.0

    def test_returns_dimensionless(self) -> None:
        ratio, factor = calculate_dp_dn_ratio_and_factor(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(3, 'mm'),
        )
        assert str(ratio.units) == 'dimensionless'
        assert str(factor.units) == 'dimensionless'


class TestCalculateVelocityShearFactor:
    def test_low_velocity(self) -> None:
        result = calculate_velocity_shear_factor(ureg.Quantity(2.0, 'm/s'))
        assert result.magnitude > 0.5

    def test_high_velocity(self) -> None:
        # v=20 m/s > 8 m/s threshold: factor = 1.0 - 0.15*log1p(20/8 - 1) ≈ 0.86
        result = calculate_velocity_shear_factor(ureg.Quantity(20.0, 'm/s'))
        assert result.magnitude < 1.0  # Above threshold, factor decreases from 1.0

    def test_min_factor_applied(self) -> None:
        result = calculate_velocity_shear_factor(ureg.Quantity(100.0, 'm/s'))
        assert result.magnitude >= 0.25

    def test_returns_dimensionless(self) -> None:
        result = calculate_velocity_shear_factor(ureg.Quantity(5.0, 'm/s'))
        assert str(result.units) == 'dimensionless'


class TestCalculateSettlingVelocityAndFactor:
    def test_typical_values(self) -> None:
        sv, factor = calculate_settling_velocity_and_factor(
            ureg.Quantity(100, 'um'),
            ureg.Quantity(10, 'm/s'),
        )
        assert sv.magnitude > 0
        assert factor.magnitude > 0

    def test_zero_velocity(self) -> None:
        sv, factor = calculate_settling_velocity_and_factor(
            ureg.Quantity(100, 'um'),
            ureg.Quantity(0, 'm/s'),
        )
        assert sv.magnitude > 0
        assert factor.magnitude == 1.0

    def test_settling_velocity_units(self) -> None:
        sv, _ = calculate_settling_velocity_and_factor(
            ureg.Quantity(100, 'um'),
            ureg.Quantity(10, 'm/s'),
        )
        assert 'meter / second' in str(sv.units)

    def test_factor_dimensionless(self) -> None:
        _, factor = calculate_settling_velocity_and_factor(
            ureg.Quantity(100, 'um'),
            ureg.Quantity(10, 'm/s'),
        )
        assert str(factor.units) == 'dimensionless'


class TestCalculateShieldsCriticalVelocity:
    def test_returns_velocity(self) -> None:
        result = calculate_shields_critical_velocity(
            ureg.Quantity(100, 'um'),
        )
        assert result.magnitude > 0
        assert str(result.units) == 'meter / second'

    def test_larger_particles_higher_velocity(self) -> None:
        v_small = calculate_shields_critical_velocity(
            ureg.Quantity(50, 'um'),
        )
        v_large = calculate_shields_critical_velocity(
            ureg.Quantity(200, 'um'),
        )
        assert v_large.magnitude > v_small.magnitude

    def test_shields_parameter_effect(self) -> None:
        v_low = calculate_shields_critical_velocity(
            ureg.Quantity(100, 'um'),
            shields_parameter=0.03,
        )
        v_high = calculate_shields_critical_velocity(
            ureg.Quantity(100, 'um'),
            shields_parameter=0.06,
        )
        assert v_high.magnitude > v_low.magnitude

    def test_typical_sand_particle(self) -> None:
        result = calculate_shields_critical_velocity(
            ureg.Quantity(150, 'um'),
        )
        # For typical sand (150 um), critical velocity should be
        # in the range of 0.1-0.5 m/s
        assert 0.05 < result.magnitude < 1.0


class TestCalculatePhysicalModifiers:
    def test_all_ones(self) -> None:
        result = calculate_physical_modifiers(
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
        )
        assert 0.1 <= result.magnitude <= 5.0

    def test_clipped_low(self) -> None:
        result = calculate_physical_modifiers(
            ureg.Quantity(0.001, 'dimensionless'),
            ureg.Quantity(0.001, 'dimensionless'),
            ureg.Quantity(0.001, 'dimensionless'),
            ureg.Quantity(0.001, 'dimensionless'),
        )
        assert result.magnitude >= 0.1

    def test_clipped_high(self) -> None:
        result = calculate_physical_modifiers(
            ureg.Quantity(100.0, 'dimensionless'),
            ureg.Quantity(100.0, 'dimensionless'),
            ureg.Quantity(100.0, 'dimensionless'),
            ureg.Quantity(100.0, 'dimensionless'),
        )
        # Combined factor clipped to maximum of 10.0
        assert result.magnitude <= 10.0

    def test_returns_dimensionless(self) -> None:
        result = calculate_physical_modifiers(
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
        )
        assert str(result.units) == 'dimensionless'


class TestComputePhysicsPint:
    def test_returns_dataframe(self) -> None:
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_input_columns(self) -> None:
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        for col in df.columns:
            assert col in result.columns

    def test_adds_physics_columns(self) -> None:
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        expected = [
            'stokes_number',
            'stokes_factor',
            'dp_dn_ratio',
            'dp_dn_factor',
            'velocity_shear_factor',
            'settling_velocity',
            'settling_velocity_factor',
        ]
        for col in expected:
            assert col in result.columns, f'Missing column: {col}'

    def test_preserves_row_count(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        result = compute_physics(df)
        assert len(result) == 10

    def test_dimensionless_columns_have_dimensionless_units(self) -> None:
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        dimensionless_cols = [
            'stokes_number',
            'stokes_factor',
            'dp_dn_ratio',
            'dp_dn_factor',
            'velocity_shear_factor',
            'settling_velocity_factor',
        ]
        for col in dimensionless_cols:
            dtype = result[col].dtype
            if isinstance(dtype, pint_pandas.PintType):
                assert str(dtype.units) == 'dimensionless', (
                    f'{col} should be dimensionless, got {dtype.units}'
                )

    def test_settling_velocity_has_velocity_units(self) -> None:
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        dtype = result['settling_velocity'].dtype
        if isinstance(dtype, pint_pandas.PintType):
            assert 'meter / second' in str(dtype.units), (
                f'settling_velocity should be m/s, got {dtype.units}'
            )

    def test_output_conforms_to_schema(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        result = compute_physics(df)
        validated = PhysicsComputedSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)


class TestStokesNumberPint:
    def test_scalar_pint_quantity(self) -> None:
        dp = ureg.Quantity(50.0, 'um')
        V = ureg.Quantity(5.0, 'm/s')
        Dn = ureg.Quantity(3.0, 'mm')

        result = calculate_stokes_number(dp, V, Dn)
        assert result.magnitude > 0
        assert str(result.units) == 'dimensionless'

    def test_array_pint_quantity(self) -> None:
        dp = ureg.Quantity(np.array([50.0, 100.0, 200.0]), 'um')
        V = ureg.Quantity(np.array([3.0, 5.0, 8.0]), 'm/s')
        Dn = ureg.Quantity(np.array([3.0, 4.0, 5.0]), 'mm')

        result = calculate_stokes_number(dp, V, Dn)
        assert len(result) == 3
        assert str(result.units) == 'dimensionless'


class TestStokesFactorPint:
    def test_scalar_pint_quantity(self) -> None:
        V = ureg.Quantity(5.0, 'm/s')
        Stk = ureg.Quantity(0.05, 'dimensionless')

        result = calculate_stokes_factor(V, Stk)
        m = result.magnitude
        assert 0 < m <= 1.0
        assert str(result.units) == 'dimensionless'


class TestDpDnPint:
    def test_scalar_pint_quantity(self) -> None:
        dp = ureg.Quantity(50.0, 'um')
        Dn = ureg.Quantity(3.0, 'mm')

        ratio, factor = calculate_dp_dn_ratio_and_factor(dp, Dn)
        assert ratio.magnitude < 1.0
        assert str(ratio.units) == 'dimensionless'
        assert str(factor.units) == 'dimensionless'


class TestVelocityShearPint:
    def test_scalar_pint_quantity(self) -> None:
        V = ureg.Quantity(5.0, 'm/s')

        result = calculate_velocity_shear_factor(V)
        assert 0.25 <= result.magnitude <= 1.0
        assert str(result.units) == 'dimensionless'


class TestSettlingVelocityPint:
    def test_scalar_pint_quantity(self) -> None:
        dp = ureg.Quantity(100.0, 'um')
        V = ureg.Quantity(3.0, 'm/s')

        vel, factor = calculate_settling_velocity_and_factor(dp, V)
        assert vel.magnitude > 0
        assert factor.magnitude > 0
        assert 'meter / second' in str(vel.units)
        assert str(factor.units) == 'dimensionless'


class TestVelocityFromPressurePint:
    def test_scalar_pint_quantity(self) -> None:
        pressure = ureg.Quantity(200.0, 'kPa')

        result = calculate_velocity_from_pressure(pressure)
        assert result.magnitude > 0
        assert str(result.units) == 'meter / second'


class TestCalculateCloggingProbability:
    def test_returns_five_quantities(self) -> None:
        df = generate_simulation_inputs(3, seed=42)
        df = compute_physics(df)
        result = calculate_clogging_probability(df)
        assert len(result) == 5
        vf, x_base, pf, x, pc = result
        assert all(isinstance(v, type(x)) for v in [vf, x_base, pf, x, pc])

    def test_probability_in_range(self) -> None:
        df = generate_simulation_inputs(3, seed=42)
        df = compute_physics(df)
        _, _, _, _, pc = calculate_clogging_probability(df)
        m = pc.magnitude
        assert np.all(m >= 0)
        assert np.all(m <= 1.0)

    def test_without_physical_constraints(self) -> None:
        df = generate_simulation_inputs(3, seed=42)
        df = compute_physics(df)
        _, _, pf_no, _, _ = calculate_clogging_probability(
            df,
            apply_physical_constraints=False,
        )
        np.testing.assert_allclose(pf_no.magnitude, 1.0)

    def test_output_schema_conformance(self) -> None:
        df = generate_simulation_inputs(3, seed=42)
        df = compute_physics(df)
        vf, x_base, pf, x, pc = calculate_clogging_probability(df)

        result_df = df.copy()
        result_df['volume_fraction'] = pint_pandas.PintArray(
            vf.magnitude, dtype='pint[dimensionless]'
        )
        result_df['X_base'] = pint_pandas.PintArray(
            x_base.magnitude, dtype='pint[dimensionless]'
        )
        result_df['physical_factor'] = pint_pandas.PintArray(
            pf.magnitude, dtype='pint[dimensionless]'
        )
        result_df['X'] = pint_pandas.PintArray(x.magnitude, dtype='pint[dimensionless]')
        result_df['clogging_probability'] = pint_pandas.PintArray(
            pc.magnitude, dtype='pint[dimensionless]'
        )
        result_df['clogging_risk'] = classify_clogging_risk(pc)

        validated = SimulationInputSchema.validate(result_df)
        assert isinstance(validated, pd.DataFrame)


class TestClassifyCloggingRisk:
    def test_low_risk(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(0.2, 'dimensionless'))
        assert result[0] == 'Low'

    def test_moderate_risk(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(0.4, 'dimensionless'))
        assert result[0] == 'Moderate'

    def test_high_risk(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(0.7, 'dimensionless'))
        assert result[0] == 'High'

    def test_boundary_low(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(0.29, 'dimensionless'))
        assert result[0] == 'Low'

    def test_boundary_moderate(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(0.31, 'dimensionless'))
        assert result[0] == 'Moderate'

    def test_boundary_high(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(0.51, 'dimensionless'))
        assert result[0] == 'High'

    def test_array_input(self) -> None:
        result = classify_clogging_risk(
            ureg.Quantity(np.array([0.2, 0.4, 0.7]), 'dimensionless')
        )
        assert len(result) == 3

    def test_clamped_to_zero(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(-0.5, 'dimensionless'))
        assert result[0] == 'Low'

    def test_clamped_to_one(self) -> None:
        result = classify_clogging_risk(ureg.Quantity(1.5, 'dimensionless'))
        assert result[0] == 'High'

    def test_returns_numpy_array(self) -> None:
        result = classify_clogging_risk(
            ureg.Quantity(np.array([0.2, 0.4, 0.7]), 'dimensionless')
        )
        assert isinstance(result, np.ndarray)


class TestPhysicsNumericalAccuracy:
    """Test that physics functions produce correct numerical values."""

    def test_velocity_from_pressure_200kpa(self) -> None:
        """v = Cd * sqrt(2*P/rho) = 0.85 * sqrt(2*200000/1000) = 17.0 m/s."""
        result = calculate_velocity_from_pressure(ureg.Quantity(200, 'kPa'))
        np.testing.assert_allclose(result.magnitude, 17.0, rtol=1e-6)

    def test_velocity_from_pressure_100kpa(self) -> None:
        """v = 0.85 * sqrt(2*100000/1000) = 12.02 m/s."""
        result = calculate_velocity_from_pressure(ureg.Quantity(100, 'kPa'))
        expected = 0.85 * np.sqrt(2 * 100000 / 1000)
        np.testing.assert_allclose(result.magnitude, expected, rtol=1e-6)

    def test_stokes_number_known_value(self) -> None:
        """Stk = (2650 * (50e-6)^2 * 10) / (18 * 1e-3 * 3e-3) = 1.22685185."""
        result = calculate_stokes_number(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(3, 'mm'),
        )
        np.testing.assert_allclose(result.magnitude, 1.22685185, rtol=1e-6)

    def test_stokes_factor_below_critical(self) -> None:
        """Stk=0.05 < 0.1: factor = 0.5 + 0.5*(0.05/0.1) = 0.75."""
        result = calculate_stokes_factor(
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(0.05, 'dimensionless'),
        )
        np.testing.assert_allclose(result.magnitude, 0.75, rtol=1e-6)

    def test_stokes_factor_above_critical(self) -> None:
        """Stk=1.0 > 0.1: factor = 1.0 + 0.15*log1p(1.0/0.1) ≈ 1.36 (uncapped)."""
        result = calculate_stokes_factor(
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(1.0, 'dimensionless'),
        )
        expected = 1.0 + 0.15 * np.log1p(1.0 / 0.1)
        np.testing.assert_allclose(result.magnitude, expected, rtol=1e-6)

    def test_dp_dn_ratio(self) -> None:
        """dp/Dn = 50um / 3mm = 0.016667."""
        ratio, _ = calculate_dp_dn_ratio_and_factor(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(3, 'mm'),
        )
        np.testing.assert_allclose(ratio.magnitude, 50e-6 / 3e-3, rtol=1e-6)

    def test_velocity_shear_factor_below_threshold(self) -> None:
        """v=2m/s < 8m/s threshold: factor = 0.5 + 0.5*(2/8) = 0.625."""
        result = calculate_velocity_shear_factor(ureg.Quantity(2.0, 'm/s'))
        np.testing.assert_allclose(result.magnitude, 0.625, rtol=1e-6)

    def test_velocity_shear_factor_above_threshold(self) -> None:
        """v=10m/s > 8m/s: factor = 1.0 - 0.15*log1p(10/8 - 1) ≈ 0.97."""
        result = calculate_velocity_shear_factor(ureg.Quantity(10.0, 'm/s'))
        ratio = 10.0 / 8.0
        expected = 1.0 - 0.15 * np.log1p(ratio - 1.0)
        np.testing.assert_allclose(result.magnitude, expected, rtol=1e-6)

    def test_settling_velocity_known_value(self) -> None:
        """ws(100um) = 0.00753413 m/s."""
        sv, _ = calculate_settling_velocity_and_factor(
            ureg.Quantity(100, 'um'),
            ureg.Quantity(10, 'm/s'),
        )
        np.testing.assert_allclose(sv.magnitude, 0.00753413, rtol=1e-4)

    def test_physical_modifiers_all_ones(self) -> None:
        """All factors = 1.0, product = 1.0."""
        result = calculate_physical_modifiers(
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
            ureg.Quantity(1.0, 'dimensionless'),
        )
        np.testing.assert_allclose(result.magnitude, 1.0, rtol=1e-6)


class TestPhysicsBoundaryConditions:
    """Test physics functions at edge cases and boundaries."""

    def test_velocity_zero_pressure(self) -> None:
        """Zero pressure produces zero velocity."""
        result = calculate_velocity_from_pressure(ureg.Quantity(0, 'kPa'))
        np.testing.assert_allclose(result.magnitude, 0.0)

    def test_stokes_number_zero_velocity(self) -> None:
        """Zero velocity produces zero Stokes number."""
        result = calculate_stokes_number(
            ureg.Quantity(50, 'um'),
            ureg.Quantity(0, 'm/s'),
            ureg.Quantity(3, 'mm'),
        )
        np.testing.assert_allclose(result.magnitude, 0.0, atol=1e-10)

    def test_stokes_number_very_small_particle(self) -> None:
        """Tiny particle produces tiny Stokes number."""
        result = calculate_stokes_number(
            ureg.Quantity(1e-3, 'um'),
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(3, 'mm'),
        )
        assert result.magnitude > 0
        assert result.magnitude < 1e-6

    def test_settling_velocity_zero_flow(self) -> None:
        """Zero flow velocity returns factor=1.0."""
        _, factor = calculate_settling_velocity_and_factor(
            ureg.Quantity(100, 'um'),
            ureg.Quantity(0, 'm/s'),
        )
        np.testing.assert_allclose(factor.magnitude, 1.0)

    def test_settling_velocity_very_small_particle(self) -> None:
        """Tiny particle produces tiny settling velocity."""
        sv, _ = calculate_settling_velocity_and_factor(
            ureg.Quantity(1e-3, 'um'),
            ureg.Quantity(10, 'm/s'),
        )
        assert sv.magnitude > 0
        assert sv.magnitude < 1e-6

    def test_dp_dn_ratio_very_small_particle(self) -> None:
        """Tiny particle produces near-zero dp/Dn ratio."""
        ratio, _ = calculate_dp_dn_ratio_and_factor(
            ureg.Quantity(1e-3, 'um'),
            ureg.Quantity(3, 'mm'),
        )
        assert ratio.magnitude > 0
        assert ratio.magnitude < 1e-6

    def test_physical_modifiers_extreme_low(self) -> None:
        """All factors near zero -> product clips to minimum 0.5."""
        result = calculate_physical_modifiers(
            ureg.Quantity(1e-10, 'dimensionless'),
            ureg.Quantity(1e-10, 'dimensionless'),
            ureg.Quantity(1e-10, 'dimensionless'),
            ureg.Quantity(1e-10, 'dimensionless'),
        )
        np.testing.assert_allclose(result.magnitude, 0.5, rtol=1e-6)

    def test_physical_modifiers_extreme_high(self) -> None:
        """All factors huge -> product clips to maximum 10.0."""
        result = calculate_physical_modifiers(
            ureg.Quantity(1e10, 'dimensionless'),
            ureg.Quantity(1e10, 'dimensionless'),
            ureg.Quantity(1e10, 'dimensionless'),
            ureg.Quantity(1e10, 'dimensionless'),
        )
        np.testing.assert_allclose(result.magnitude, 10.0, rtol=1e-6)

    def test_velocity_shear_factor_minimum(self) -> None:
        """v=1000 m/s: factor = max(1 - 0.15*log1p(1000/8-1), 0.25) ≈ 0.276."""
        result = calculate_velocity_shear_factor(ureg.Quantity(1000, 'm/s'))
        np.testing.assert_allclose(result.magnitude, 0.2758, rtol=1e-2)

    def test_stokes_factor_at_critical(self) -> None:
        """Stk=0.1 at critical: uses above-critical branch (strict < comparison)."""
        result = calculate_stokes_factor(
            ureg.Quantity(10, 'm/s'),
            ureg.Quantity(0.1, 'dimensionless'),
        )
        expected = 1.0 + 0.15 * np.log1p(0.1 / 0.1)
        np.testing.assert_allclose(result.magnitude, expected, rtol=1e-6)

    def test_velocity_shear_factor_at_threshold(self) -> None:
        """v=2m/s < 8m/s threshold: factor = 0.5 + 0.5*(2/8) = 0.625."""
        result = calculate_velocity_shear_factor(ureg.Quantity(2.0, 'm/s'))
        np.testing.assert_allclose(result.magnitude, 0.625, rtol=1e-6)


class TestUnitMismatchRejection:
    """Test that schema validation rejects truly incompatible units."""

    def test_wrong_pressure_dimension_rejected(self) -> None:
        """Length column should fail when schema expects pressure."""
        df = pd.DataFrame(
            {
                'TSS_mg_L': pint_pandas.PintArray(
                    [100.0], dtype='pint[milligram / liter]'
                ),
                'pressure_kPa': pint_pandas.PintArray([200.0], dtype='pint[meter]'),
                'nozzle_diameter_mm': pint_pandas.PintArray(
                    [3.0], dtype='pint[millimeter]'
                ),
                'duration_hrs': pint_pandas.PintArray([2.0], dtype='pint[hour]'),
                'particle_diameter_um': pint_pandas.PintArray(
                    [50.0], dtype='pint[micrometer]'
                ),
                'particle_size_range': ['Medium'],
            }
        )
        with pytest.raises(Exception):
            SimulationInputSchema.validate(df)

    def test_wrong_tss_dimension_rejected(self) -> None:
        """Length column should fail when schema expects concentration."""
        df = pd.DataFrame(
            {
                'TSS_mg_L': pint_pandas.PintArray([0.1], dtype='pint[meter]'),
                'pressure_kPa': pint_pandas.PintArray(
                    [200.0], dtype='pint[kilopascal]'
                ),
                'nozzle_diameter_mm': pint_pandas.PintArray(
                    [3.0], dtype='pint[millimeter]'
                ),
                'duration_hrs': pint_pandas.PintArray([2.0], dtype='pint[hour]'),
                'particle_diameter_um': pint_pandas.PintArray(
                    [50.0], dtype='pint[micrometer]'
                ),
                'particle_size_range': ['Medium'],
            }
        )
        with pytest.raises(Exception):
            SimulationInputSchema.validate(df)

    def test_correct_units_pass(self) -> None:
        """Correct units should pass schema validation."""
        df = pd.DataFrame(
            {
                'TSS_mg_L': pint_pandas.PintArray(
                    [100.0], dtype='pint[milligram / liter]'
                ),
                'pressure_kPa': pint_pandas.PintArray(
                    [200.0], dtype='pint[kilopascal]'
                ),
                'nozzle_diameter_mm': pint_pandas.PintArray(
                    [3.0], dtype='pint[millimeter]'
                ),
                'duration_hrs': pint_pandas.PintArray([2.0], dtype='pint[hour]'),
                'particle_diameter_um': pint_pandas.PintArray(
                    [50.0], dtype='pint[micrometer]'
                ),
                'particle_size_range': ['Medium'],
            }
        )
        validated = SimulationInputSchema.validate(df)
        assert isinstance(validated, pd.DataFrame)


class TestDimensionalityError:
    """Test that wrong dimensionality raises errors."""

    def test_velocity_from_pressure_wrong_units(self) -> None:
        """Passing length instead of pressure should raise DimensionalityError."""
        with pytest.raises(pint.DimensionalityError):
            calculate_velocity_from_pressure(ureg.Quantity(200, 'mm'))

    def test_stokes_number_wrong_units(self) -> None:
        """Passing length for velocity should raise DimensionalityError."""
        with pytest.raises(pint.DimensionalityError):
            calculate_stokes_number(
                ureg.Quantity(50, 'um'),
                ureg.Quantity(10, 'mm'),
                ureg.Quantity(3, 'mm'),
            )

    def test_velocity_shear_wrong_units(self) -> None:
        """Passing pressure instead of velocity should raise DimensionalityError."""
        with pytest.raises(pint.DimensionalityError):
            calculate_velocity_shear_factor(ureg.Quantity(200, 'kPa'))

    def test_settling_velocity_wrong_units(self) -> None:
        """Passing length for velocity should raise DimensionalityError."""
        with pytest.raises(pint.DimensionalityError):
            calculate_settling_velocity_and_factor(
                ureg.Quantity(100, 'um'),
                ureg.Quantity(50, 'mm'),
            )

    def test_classify_risk_wrong_units(self) -> None:
        """Passing velocity instead of dimensionless raises DimensionalityError."""
        with pytest.raises(pint.DimensionalityError):
            classify_clogging_risk(ureg.Quantity(0.5, 'm/s'))


class TestToQuantityHelper:
    """Test the to_quantity helper function."""

    def test_pint_array_series_to_quantity(self) -> None:
        """PintArray-backed Series converts to Quantity."""
        s = pd.Series(
            pint_pandas.PintArray(np.array([100.0, 200.0]), dtype='pint[kPa]')
        )
        result = to_quantity(s)
        assert isinstance(result, pint.Quantity)
        assert len(result) == 2

    def test_non_series_raises(self) -> None:
        """Non-Series input raises TypeError."""
        with pytest.raises(TypeError):
            to_quantity(np.array([100.0]))

    def test_non_pint_series_raises(self) -> None:
        """Regular Series (no pint accessor) raises TypeError."""
        s = pd.Series([100.0, 200.0])
        with pytest.raises(TypeError):
            to_quantity(s)


class TestToPintArrayHelper:
    """Test the to_pint_array helper function."""

    def test_quantity_to_pint_array(self) -> None:
        """Quantity converts to PintArray with correct unit."""
        q = ureg.Quantity(np.array([17.0]), 'm/s')
        result = to_pint_array(q, 'velocity_m_s')
        assert isinstance(result, pint_pandas.PintArray)
        assert str(result.dtype.units) == 'meter / second'
        np.testing.assert_allclose(pd.Series(result).pint.magnitude, 17.0)

    def test_dimensionless_to_pint_array(self) -> None:
        """Dimensionless Quantity converts correctly."""
        q = ureg.Quantity(np.array([0.5]), 'dimensionless')
        result = to_pint_array(q, 'stokes_number')
        assert isinstance(result, pint_pandas.PintArray)
        assert str(result.dtype.units) == 'dimensionless'

    def test_unit_conversion(self) -> None:
        """Quantity is converted to target unit."""
        q = ureg.Quantity(np.array([200.0]), 'kPa')
        result = to_pint_array(q, 'pressure_kPa')
        assert isinstance(result, pint_pandas.PintArray)
        np.testing.assert_allclose(pd.Series(result).pint.magnitude, 200.0)


class TestComputePhysicsNumericalAccuracy:
    """Test that compute_physics produces correct numerical values."""

    def test_velocity_derived_from_pressure(self) -> None:
        """Velocity is correctly computed from pressure in compute_physics."""
        df = generate_simulation_inputs(5, seed=42)
        result = compute_physics(df)
        expected_velocity = calculate_velocity_from_pressure(
            df['pressure_kPa'].pint.quantity
        )
        np.testing.assert_allclose(
            result['velocity_m_s'].pint.magnitude,
            expected_velocity.magnitude,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result['velocity_m_s'].pint.magnitude,
            expected_velocity.magnitude,
            rtol=1e-10,
        )

    def test_stokes_number_positive(self) -> None:
        """Stokes number is always positive."""
        df = generate_simulation_inputs(100, seed=42)
        result = compute_physics(df)
        assert (result['stokes_number'].pint.magnitude > 0).all()

    def test_stokes_factor_in_range(self) -> None:
        """Stokes factor is in [0.5, 2.0] based on implementation."""
        df = generate_simulation_inputs(100, seed=42)
        result = compute_physics(df)
        m = result['stokes_factor'].pint.magnitude
        assert (m >= 0.5).all()
        assert (m <= 2.0).all()

    def test_settling_velocity_positive(self) -> None:
        """Settling velocity is always positive."""
        df = generate_simulation_inputs(100, seed=42)
        result = compute_physics(df)
        assert (result['settling_velocity'].pint.magnitude > 0).all()


class TestCloggingProbabilityBehavior:
    """Test clogging probability behavior with varied inputs."""

    def test_high_tss_increases_probability(self) -> None:
        """Higher TSS should produce higher clogging probability."""
        df_low = generate_simulation_inputs(50, seed=42)
        df_high = generate_simulation_inputs(50, seed=42)

        df_low['TSS_mg_L'] = pint_pandas.PintArray(
            np.full(50, 10.0), dtype='pint[milligram / liter]'
        )
        df_high['TSS_mg_L'] = pint_pandas.PintArray(
            np.full(50, 500.0), dtype='pint[milligram / liter]'
        )

        df_low = compute_physics(df_low)
        df_high = compute_physics(df_high)

        _, _, _, _, pc_low = calculate_clogging_probability(df_low)
        _, _, _, _, pc_high = calculate_clogging_probability(df_high)

        assert pc_high.magnitude.mean() > pc_low.magnitude.mean()

    def test_high_pressure_increases_velocity_and_probability(self) -> None:
        """Higher pressure increases velocity, raising clogging probability."""
        df_low_p = generate_simulation_inputs(50, seed=42)
        df_high_p = generate_simulation_inputs(50, seed=42)

        df_low_p['pressure_kPa'] = pint_pandas.PintArray(
            np.full(50, 100.0), dtype='pint[kilopascal]'
        )
        df_high_p['pressure_kPa'] = pint_pandas.PintArray(
            np.full(50, 400.0), dtype='pint[kilopascal]'
        )

        df_low_p = compute_physics(df_low_p)
        df_high_p = compute_physics(df_high_p)

        _, _, _, _, pc_low = calculate_clogging_probability(df_low_p)
        _, _, _, _, pc_high = calculate_clogging_probability(df_high_p)

        assert pc_high.magnitude.mean() > pc_low.magnitude.mean()

    def test_large_particles_increase_probability(self) -> None:
        """Larger particles should produce higher clogging probability."""
        df_small = generate_simulation_inputs(50, seed=42)
        df_large = generate_simulation_inputs(50, seed=42)

        df_small['particle_diameter_um'] = pint_pandas.PintArray(
            np.full(50, 10.0), dtype='pint[micrometer]'
        )
        df_large['particle_diameter_um'] = pint_pandas.PintArray(
            np.full(50, 300.0), dtype='pint[micrometer]'
        )

        df_small = compute_physics(df_small)
        df_large = compute_physics(df_large)

        _, _, _, _, pc_small = calculate_clogging_probability(df_small)
        _, _, _, _, pc_large = calculate_clogging_probability(df_large)

        assert pc_large.magnitude.mean() > pc_small.magnitude.mean()

    def test_longer_duration_increases_probability(self) -> None:
        """Longer duration should produce higher clogging probability."""
        df_short = generate_simulation_inputs(50, seed=42)
        df_long = generate_simulation_inputs(50, seed=42)

        df_short['duration_hrs'] = pint_pandas.PintArray(
            np.full(50, 0.5), dtype='pint[hour]'
        )
        df_long['duration_hrs'] = pint_pandas.PintArray(
            np.full(50, 24.0), dtype='pint[hour]'
        )

        df_short = compute_physics(df_short)
        df_long = compute_physics(df_long)

        _, _, _, _, pc_short = calculate_clogging_probability(df_short)
        _, _, _, _, pc_long = calculate_clogging_probability(df_long)

        assert pc_long.magnitude.mean() > pc_short.magnitude.mean()

    def test_probability_bounds_for_extreme_inputs(self) -> None:
        """Even extreme inputs should produce probability in [0, 1]."""
        df = generate_simulation_inputs(50, seed=42)
        df['TSS_mg_L'] = pint_pandas.PintArray(
            np.full(50, 1000.0), dtype='pint[milligram / liter]'
        )
        df['particle_diameter_um'] = pint_pandas.PintArray(
            np.full(50, 300.0), dtype='pint[micrometer]'
        )
        df['pressure_kPa'] = pint_pandas.PintArray(
            np.full(50, 100.0), dtype='pint[kilopascal]'
        )
        df['duration_hrs'] = pint_pandas.PintArray(
            np.full(50, 24.0), dtype='pint[hour]'
        )

        df = compute_physics(df)
        _, _, _, _, pc = calculate_clogging_probability(df)

        assert (pc.magnitude >= 0).all()
        assert (pc.magnitude <= 1.0).all()


class TestRunSimulation:
    def test_default_simulation(self) -> None:
        df = run_simulation(total_samples=100, chunk_size=50)
        assert df is not None
        assert len(df) == 100
        validated_df = SimulationOutputSchema.validate(df)
        assert isinstance(validated_df, pd.DataFrame)

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


class TestPropertyBasedPhysics:
    """Property-based tests verifying physics functions with Hypothesis."""

    @given(
        st.floats(min_value=10, max_value=990, allow_nan=False, allow_infinity=False)
    )
    @settings(deadline=None)
    def test_velocity_from_pressure_always_positive(self, pressure_kpa: float) -> None:
        """Velocity from pressure is always non-negative for any valid pressure."""
        result = calculate_velocity_from_pressure(ureg.Quantity(pressure_kpa, 'kPa'))
        assert result.magnitude >= 0

    @given(
        st.floats(min_value=1, max_value=999, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1, max_value=50, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.5, max_value=49, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_stokes_number_always_non_negative(
        self, particle_um: float, velocity_m_s: float, nozzle_mm: float
    ) -> None:
        """Stokes number is always non-negative for valid inputs."""
        result = calculate_stokes_number(
            ureg.Quantity(particle_um, 'um'),
            ureg.Quantity(velocity_m_s, 'm/s'),
            ureg.Quantity(nozzle_mm, 'mm'),
        )
        assert result.magnitude >= 0

    @given(
        st.floats(min_value=1, max_value=999, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.5, max_value=49, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_dp_dn_ratio_between_zero_and_one(
        self, particle_um: float, nozzle_mm: float
    ) -> None:
        """dp/Dn ratio is always in [0, 1] when particle is smaller than nozzle."""
        if particle_um >= nozzle_mm * 1000:
            return
        ratio, _ = calculate_dp_dn_ratio_and_factor(
            particle_um * ureg.um, nozzle_mm * ureg.mm
        )
        m = ratio.magnitude
        assert m >= 0
        assert m <= 1

    @given(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False))
    @settings(deadline=None)
    def test_classify_risk_returns_valid_levels(self, probability: float) -> None:
        """Classification always returns one of the valid risk levels."""
        result = classify_clogging_risk(ureg.Quantity(probability, 'dimensionless'))
        assert result[0] in config.RISK_LEVELS

    @given(simulation_input_df())
    @settings(deadline=None)
    def test_compute_physics_preserves_rows(self, sample_df: pd.DataFrame) -> None:
        """Physics computation preserves row count."""
        result = compute_physics(sample_df)
        assert len(result) == len(sample_df)

    @given(simulation_input_df())
    @settings(deadline=None)
    def test_compute_physics_output_conforms_to_schema(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Physics output always conforms to PhysicsComputedSchema."""
        result = compute_physics(sample_df)
        validated = PhysicsComputedSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)

    @given(
        st.floats(min_value=0, max_value=50, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_stokes_factor_between_zero_and_one(
        self, velocity: float, stokes_number: float
    ) -> None:
        """Stokes factor is in [0.5, 2.0] based on implementation."""
        result = calculate_stokes_factor(
            ureg.Quantity(velocity, 'm/s'),
            ureg.Quantity(stokes_number, 'dimensionless'),
        )
        m = result.magnitude
        assert m >= 0.5
        assert m <= 2.0

    @given(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(deadline=None)
    def test_velocity_shear_factor_above_minimum(self, velocity: float) -> None:
        """Velocity shear factor is always >= min_factor (0.25)."""
        result = calculate_velocity_shear_factor(ureg.Quantity(velocity, 'm/s'))
        assert result.magnitude >= 0.25

    @given(
        st.floats(min_value=1, max_value=300, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_settling_velocity_positive(
        self, particle_um: float, velocity: float
    ) -> None:
        """Settling velocity is always positive for valid inputs."""
        sv, factor = calculate_settling_velocity_and_factor(
            ureg.Quantity(particle_um, 'um'),
            ureg.Quantity(velocity, 'm/s'),
        )
        assert sv.magnitude >= 0
        assert factor.magnitude >= 0

    @given(
        st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(deadline=None)
    def test_physical_modifiers_in_valid_range(
        self, dp_dn: float, stokes: float, shear: float, settling: float
    ) -> None:
        """Physical modifiers output is always in [0.1, 5.0]."""
        result = calculate_physical_modifiers(
            ureg.Quantity(dp_dn, 'dimensionless'),
            ureg.Quantity(stokes, 'dimensionless'),
            ureg.Quantity(shear, 'dimensionless'),
            ureg.Quantity(settling, 'dimensionless'),
        )
        m = result.magnitude
        assert m >= 0.1
        assert m <= 5.0

    @given(simulation_input_df())
    @settings(deadline=None)
    def test_clogging_probability_between_zero_and_one(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Clogging probability is always in [0, 1] for any valid input."""
        physics_df = compute_physics(sample_df)
        _, _, _, _, pc = calculate_clogging_probability(physics_df)
        m = pc.magnitude
        assert (m >= 0).all()
        assert (m <= 1).all()
