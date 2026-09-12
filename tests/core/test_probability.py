import numpy as np
import pandas as pd
import pint_pandas

from nozzle_clogging.generation import generate_simulation_inputs
from nozzle_clogging.probability import (
    calculate_clogging_probability,
    calculate_risk_proportions,
    classify_clogging_risk,
    run_calibration_sensitivity_sweep,
)
from nozzle_clogging.schemas import SimulationInputSchema
from nozzle_clogging.simulation import (
    compute_and_classify_clogging_probability,
    compute_physics,
)
from nozzle_clogging.units import ureg


def _make_physics_df() -> pd.DataFrame:
    """Create a minimal PhysicsComputedSchema DataFrame for testing."""
    df = generate_simulation_inputs(3, seed=42)
    return compute_physics(df)


def _make_full_output_df() -> pd.DataFrame:
    """Create a DataFrame with X and probability columns for testing."""
    df = generate_simulation_inputs(3, seed=42)
    physics_df = compute_physics(df)
    return compute_and_classify_clogging_probability(physics_df)


class TestCalculateCloggingProbability:
    def test_returns_five_quantities(self) -> None:
        df = _make_physics_df()
        result = calculate_clogging_probability(df)
        assert len(result) == 5
        vf, x_base, pf, x, pc = result
        assert all(isinstance(v, type(x)) for v in [vf, x_base, pf, x, pc])

    def test_probability_in_range(self) -> None:
        df = _make_physics_df()
        _, _, _, _, pc = calculate_clogging_probability(df)
        m = pc.magnitude
        assert np.all(m >= 0)
        assert np.all(m <= 1.0)

    def test_without_physical_constraints(self) -> None:
        df = _make_physics_df()
        _, _, pf_no, _, _ = calculate_clogging_probability(
            df,
            apply_physical_constraints=False,
        )
        np.testing.assert_allclose(pf_no.magnitude, 1.0)

    def test_output_schema_conformance(self) -> None:
        df = _make_physics_df()
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


class TestCalculateRiskProportions:
    def test_proportions_sum_to_one(self) -> None:
        df = _make_physics_df()
        props = calculate_risk_proportions(df)
        total = props['Low'] + props['Moderate'] + props['High']
        np.testing.assert_allclose(total, 1.0, atol=1e-9)

    def test_all_keys_present(self) -> None:
        df = _make_physics_df()
        props = calculate_risk_proportions(df)
        assert set(props.keys()) == {'Low', 'Moderate', 'High'}

    def test_proportions_nonnegative(self) -> None:
        df = _make_physics_df()
        props = calculate_risk_proportions(df)
        assert all(v >= 0 for v in props.values())


class TestRunCalibrationSensitivitySweep:
    def test_returns_dataframe(self) -> None:
        df = _make_full_output_df()
        result = run_calibration_sensitivity_sweep(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self) -> None:
        df = _make_full_output_df()
        result = run_calibration_sensitivity_sweep(df)
        expected_cols = {
            'logistic_scale',
            'centering_offset',
            'Low_prop',
            'Moderate_prop',
            'High_prop',
            'Moderate_gte_15pct',
        }
        assert set(result.columns) == expected_cols

    def test_row_count_matches_combinations(self) -> None:
        from nozzle_clogging import config

        df = _make_full_output_df()
        result = run_calibration_sensitivity_sweep(df)
        expected_rows = len(
            config.CALIBRATION_SENSITIVITY_VALUES['logistic_scale']
        ) * len(config.CALIBRATION_SENSITIVITY_VALUES['centering_offset'])
        assert len(result) == expected_rows

    def test_proportions_sum_to_one_across_rows(self) -> None:
        df = _make_full_output_df()
        result = run_calibration_sensitivity_sweep(df)
        sums = result['Low_prop'] + result['Moderate_prop'] + result['High_prop']
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)

    def test_finds_optimal_gamma_meeting_threshold(self) -> None:
        df = _make_full_output_df()
        result = run_calibration_sensitivity_sweep(df)
        meeting_threshold = result[result['Moderate_gte_15pct']]
        assert len(meeting_threshold) > 0, (
            'Expected at least one param combo to meet 15% Moderate threshold'
        )

    def test_custom_parameter_values(self) -> None:
        df = _make_full_output_df()
        result = run_calibration_sensitivity_sweep(
            df, logistic_scales=[1.0, 2.0], centering_offsets=[3.0]
        )
        assert len(result) == 2
        assert set(result['logistic_scale'].unique()) == {1.0, 2.0}
        assert set(result['centering_offset'].unique()) == {3.0}
