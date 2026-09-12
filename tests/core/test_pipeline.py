import pandas as pd
import pytest

from nozzle_clogging.pipeline import PipelineConfig, SimulationPipeline
from nozzle_clogging.schemas import (
    PhysicsComputedSchema,
    SimulationInputSchema,
    SimulationOutputSchema,
)


class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.n_samples == 20_000
        assert cfg.chunk_size == 2_000
        assert cfg.seed == 42

    def test_custom_values(self) -> None:
        cfg = PipelineConfig(n_samples=100, chunk_size=10, seed=7)
        assert cfg.n_samples == 100
        assert cfg.chunk_size == 10
        assert cfg.seed == 7


class TestSimulationPipelineRun:
    def test_run_produces_valid_output(self) -> None:
        pipeline = SimulationPipeline(PipelineConfig(n_samples=50, chunk_size=25))
        result = pipeline.run()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50
        validated = SimulationOutputSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)

    def test_run_matches_functional_api(self) -> None:
        from nozzle_clogging.simulation import run_simulation

        pipeline = SimulationPipeline(
            PipelineConfig(n_samples=100, chunk_size=50, seed=42)
        )
        result_pipeline = pipeline.run()
        result_functional = run_simulation(total_samples=100, chunk_size=50, seed=42)

        pd.testing.assert_frame_equal(result_pipeline, result_functional)

    def test_run_caches_output(self) -> None:
        pipeline = SimulationPipeline(PipelineConfig(n_samples=20, chunk_size=10))
        pipeline.run()
        assert pipeline.output is not None
        assert len(pipeline.output) == 20


class TestSimulationPipelineStages:
    def test_generate_returns_valid_schema(self) -> None:
        pipeline = SimulationPipeline()
        result = pipeline.generate(n_samples=10, seed=42)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        validated = SimulationInputSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)
        assert pipeline.inputs is not None

    def test_compute_physics_returns_valid_schema(self) -> None:
        pipeline = SimulationPipeline()
        inputs = pipeline.generate(n_samples=10, seed=42)
        result = pipeline.compute_physics(inputs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        validated = PhysicsComputedSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)
        assert pipeline.physics is not None

    def test_compute_probability_returns_valid_schema(self) -> None:
        pipeline = SimulationPipeline()
        inputs = pipeline.generate(n_samples=10, seed=42)
        physics = pipeline.compute_physics(inputs)
        result = pipeline.compute_probability(physics)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        validated = SimulationOutputSchema.validate(result)
        assert isinstance(validated, pd.DataFrame)
        assert pipeline.output is not None

    def test_stage_chaining(self) -> None:
        pipeline = SimulationPipeline()
        inputs = pipeline.generate(n_samples=10, seed=42)
        physics = pipeline.compute_physics()
        output = pipeline.compute_probability()

        pd.testing.assert_frame_equal(inputs, pipeline.inputs)
        pd.testing.assert_frame_equal(physics, pipeline.physics)
        pd.testing.assert_frame_equal(output, pipeline.output)

    def test_compute_physics_no_data_raises(self) -> None:
        pipeline = SimulationPipeline()
        with pytest.raises(ValueError, match='No input data'):
            pipeline.compute_physics()

    def test_compute_probability_no_data_raises(self) -> None:
        pipeline = SimulationPipeline()
        with pytest.raises(ValueError, match='No physics data'):
            pipeline.compute_probability()

    def test_partial_run_generate_only(self) -> None:
        pipeline = SimulationPipeline()
        _inputs = pipeline.generate(n_samples=10, seed=42)
        assert pipeline.inputs is not None
        assert pipeline.physics is None
        assert pipeline.output is None

    def test_partial_run_generate_and_physics(self) -> None:
        pipeline = SimulationPipeline()
        pipeline.generate(n_samples=10, seed=42)
        pipeline.compute_physics()
        assert pipeline.inputs is not None
        assert pipeline.physics is not None
        assert pipeline.output is None
