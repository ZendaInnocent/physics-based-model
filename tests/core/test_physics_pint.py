import numpy as np
import pandas as pd
import pint_pandas

from nozzle_clogging.generation import generate_simulation_inputs
from nozzle_clogging.physics import (
    calculate_dp_dn_ratio_and_factor,
    calculate_settling_velocity_and_factor,
    calculate_stokes_factor,
    calculate_stokes_number,
    calculate_velocity_from_pressure,
    calculate_velocity_shear_factor,
)
from nozzle_clogging.schemas import PhysicsComputedSchema
from nozzle_clogging.simulation import compute_physics
from nozzle_clogging.units import ureg


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
