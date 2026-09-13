from dataclasses import dataclass

from pandera.typing import DataFrame

from nozzle_clogging import config
from nozzle_clogging.generation import generate_simulation_inputs
from nozzle_clogging.orchestration import (
    compute_and_classify_clogging_probability,
    compute_physics,
    run_batched,
)
from nozzle_clogging.schemas import (
    PhysicsComputedSchema,
    SimulationInputSchema,
    SimulationOutputSchema,
)


@dataclass
class PipelineConfig:
    """Configuration for the simulation pipeline."""

    n_samples: int = 20_000
    chunk_size: int = 2_000
    seed: int = config.RANDOM_SEED


class SimulationPipeline:
    """Configurable pipeline with convenience and power-user methods.

    Encapsulates the 3-stage simulation pipeline:
    1. Generate simulation inputs (Latin Hypercube sampling)
    2. Compute physics parameters (Stokes number, settling velocity, etc.)
    3. Compute clogging probability and risk classification

    Usage::

        # Common case - one liner
        results = SimulationPipeline(n_samples=50_000).run()

        # Power user - manual stage control
        pipeline = SimulationPipeline()
        df_inputs = pipeline.generate(n_samples=5_000)
        df_physics = pipeline.compute_physics(df_inputs)
        df_output = pipeline.compute_probability(df_physics)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._inputs: DataFrame[SimulationInputSchema] | None = None
        self._physics: DataFrame[PhysicsComputedSchema] | None = None
        self._output: DataFrame[SimulationOutputSchema] | None = None

    @property
    def inputs(self) -> DataFrame[SimulationInputSchema] | None:
        """Cached simulation inputs from the last generate() call."""
        return self._inputs

    @property
    def physics(self) -> DataFrame[PhysicsComputedSchema] | None:
        """Cached physics results from the last compute_physics() call."""
        return self._physics

    @property
    def output(self) -> DataFrame[SimulationOutputSchema] | None:
        """Cached output from the last compute_probability() call."""
        return self._output

    def generate(
        self,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> DataFrame[SimulationInputSchema]:
        """Generate simulation inputs via Latin Hypercube sampling.

        Args:
            n_samples: Number of samples. Defaults to PipelineConfig.n_samples.
            seed: Random seed. Defaults to PipelineConfig.seed.

        Returns:
            DataFrame adhering to :class:`SimulationInputSchema`.
        """
        n = n_samples or self._config.n_samples
        s = seed if seed is not None else self._config.seed
        self._inputs = generate_simulation_inputs(n, s)
        return self._inputs

    def compute_physics(
        self,
        df: DataFrame[SimulationInputSchema] | None = None,
    ) -> DataFrame[PhysicsComputedSchema]:
        """Compute physics parameters for simulation inputs.

        Args:
            df: Input DataFrame. If None, uses cached inputs from generate().

        Returns:
            DataFrame with physics columns, adhering to :class:`PhysicsComputedSchema`.

        Raises:
            ValueError: If no input DataFrame provided and generate() not called.
        """
        if df is None:
            if self._inputs is None:
                raise ValueError('No input data. Call generate() first or pass df.')
            df = self._inputs
        self._physics = compute_physics(df)
        return self._physics

    def compute_probability(
        self,
        df: DataFrame[PhysicsComputedSchema] | None = None,
    ) -> DataFrame[SimulationOutputSchema]:
        """Compute clogging probability and risk classification.

        Args:
            df: Physics DataFrame. If None, uses cached physics from compute_physics().

        Returns:
            DataFrame with probability columns, adhering to
            :class:`SimulationOutputSchema`.

        Raises:
            ValueError: If no input DataFrame provided and compute_physics() not called.
        """
        if df is None:
            if self._physics is None:
                raise ValueError(
                    'No physics data. Call compute_physics() first or pass df.'
                )
            df = self._physics
        self._output = compute_and_classify_clogging_probability(df)
        return self._output

    def run(
        self,
        n_samples: int | None = None,
        chunk_size: int | None = None,
        seed: int | None = None,
    ) -> DataFrame[SimulationOutputSchema]:
        """Run the full simulation pipeline.

        Args:
            n_samples: Total samples. Defaults to PipelineConfig.n_samples.
            chunk_size: Batch size. Defaults to PipelineConfig.chunk_size.
            seed: Random seed. Defaults to PipelineConfig.seed.

        Returns:
            Combined results DataFrame adhering to :class:`SimulationOutputSchema`.
        """
        n = n_samples or self._config.n_samples
        cs = chunk_size or self._config.chunk_size
        s = seed if seed is not None else self._config.seed

        def _gen(batch_n: int, batch_seed: int) -> DataFrame[SimulationInputSchema]:
            return generate_simulation_inputs(batch_n, batch_seed)

        self._output = run_batched(n, cs, s, _gen)
        return self._output
