import numpy as np
import pandas as pd
import pint_pandas

from nozzle_clogging.generation import (
    compute_lognormal_params,
    generate_lhs_samples,
    generate_simulation_inputs,
    generate_vectorized_lognormal_particle_sizes,
)


class TestComputeLognormalParams:
    def test_symmetric_range(self) -> None:
        mu, sigma = compute_lognormal_params(10, 100)
        assert mu > 0
        assert sigma > 0
        expected_median = np.sqrt(10 * 100)
        assert np.isclose(np.exp(mu), expected_median)

    def test_narrow_range(self) -> None:
        mu, sigma = compute_lognormal_params(50, 60)
        assert sigma < compute_lognormal_params(10, 100)[1]

    def test_wide_range(self) -> None:
        mu, sigma = compute_lognormal_params(10, 300)
        assert sigma > compute_lognormal_params(10, 100)[1]


class TestGenerateVectorizedLognormalParticleSizes:
    def test_returns_correct_length(self) -> None:
        rng = np.random.default_rng(42)
        indices = np.array([0, 1, 2, 0, 1])
        ranges = {'Fine': (10, 50), 'Medium': (50, 150), 'Coarse': (150, 300)}
        result = generate_vectorized_lognormal_particle_sizes(indices, rng, ranges)
        assert len(result) == 5

    def test_all_positive(self) -> None:
        rng = np.random.default_rng(42)
        indices = np.array([0, 1, 2])
        ranges = {'Fine': (10, 50), 'Medium': (50, 150), 'Coarse': (150, 300)}
        result = generate_vectorized_lognormal_particle_sizes(indices, rng, ranges)
        assert np.all(result > 0)

    def test_coarse_larger_than_fine(self) -> None:
        rng = np.random.default_rng(42)
        n = 1000
        fine_idx = np.zeros(n, dtype=int)
        coarse_idx = np.full(n, 2, dtype=int)
        ranges = {'Fine': (10, 50), 'Medium': (50, 150), 'Coarse': (150, 300)}
        fine = generate_vectorized_lognormal_particle_sizes(fine_idx, rng, ranges)
        coarse = generate_vectorized_lognormal_particle_sizes(coarse_idx, rng, ranges)
        assert np.mean(coarse) > np.mean(fine)


class TestGenerateLHSSamples:
    def test_returns_six_arrays(self) -> None:
        result = generate_lhs_samples(10, seed=42)
        assert len(result) == 6

    def test_correct_lengths(self) -> None:
        n = 50
        result = generate_lhs_samples(n, seed=42)
        for arr in result:
            assert len(arr) == n

    def test_deterministic_with_seed(self) -> None:
        r1 = generate_lhs_samples(10, seed=42)
        r2 = generate_lhs_samples(10, seed=42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_tss_in_range(self) -> None:
        result = generate_lhs_samples(100, seed=42)
        tss = result[0]
        assert np.all(tss >= 10)
        assert np.all(tss <= 500)

    def test_pressure_in_range(self) -> None:
        result = generate_lhs_samples(100, seed=42)
        pressure = result[1]
        assert np.all(pressure >= 100)
        assert np.all(pressure <= 400)

    def test_particle_size_range_valid(self) -> None:
        result = generate_lhs_samples(100, seed=42)
        ps_range = result[5]
        valid = {'Fine', 'Medium', 'Coarse'}
        assert all(v in valid for v in ps_range)


class TestGenerateSimulationInputs:
    def test_returns_dataframe(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        expected_columns = {
            'TSS_mg_L',
            'pressure_kPa',
            'nozzle_diameter_mm',
            'duration_hrs',
            'particle_diameter_um',
            'particle_size_range',
        }
        assert set(df.columns) == expected_columns

    def test_correct_row_count(self) -> None:
        n = 25
        df = generate_simulation_inputs(n, seed=42)
        assert len(df) == n

    def test_numeric_columns_have_pint_dtype(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        pint_columns = [
            'TSS_mg_L',
            'pressure_kPa',
            'nozzle_diameter_mm',
            'duration_hrs',
            'particle_diameter_um',
        ]
        for col in pint_columns:
            assert isinstance(df[col].dtype, pint_pandas.PintType), (
                f'{col} should have pint dtype, got {df[col].dtype}'
            )

    def test_tss_has_correct_unit(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert str(df['TSS_mg_L'].dtype.units) == 'milligram / liter'

    def test_pressure_has_correct_unit(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert str(df['pressure_kPa'].dtype.units) == 'kilopascal'

    def test_nozzle_diameter_has_correct_unit(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert str(df['nozzle_diameter_mm'].dtype.units) == 'millimeter'

    def test_duration_has_correct_unit(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert str(df['duration_hrs'].dtype.units) == 'hour'

    def test_particle_diameter_has_correct_unit(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert str(df['particle_diameter_um'].dtype.units) == 'micrometer'

    def test_particle_size_range_is_string(self) -> None:
        df = generate_simulation_inputs(10, seed=42)
        assert pd.api.types.is_string_dtype(df['particle_size_range'])

    def test_deterministic_with_seed(self) -> None:
        df1 = generate_simulation_inputs(10, seed=42)
        df2 = generate_simulation_inputs(10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_tss_values_in_range(self) -> None:
        df = generate_simulation_inputs(100, seed=42)
        tss = df['TSS_mg_L'].pint.magnitude
        np.testing.assert_array_compare(lambda x, y: x >= y, tss, np.full_like(tss, 10))
        np.testing.assert_array_compare(
            lambda x, y: x <= y, tss, np.full_like(tss, 500)
        )

    def test_pressure_values_in_range(self) -> None:
        df = generate_simulation_inputs(100, seed=42)
        pressure = df['pressure_kPa'].pint.magnitude
        np.testing.assert_array_compare(
            lambda x, y: x >= y, pressure, np.full_like(pressure, 100)
        )
        np.testing.assert_array_compare(
            lambda x, y: x <= y, pressure, np.full_like(pressure, 400)
        )
