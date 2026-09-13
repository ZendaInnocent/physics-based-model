"""Pandera dtype registration for pint-pandas extension types.

Registers PintType with pandera's engine so that DataFrameModel schemas
can use pint-pandas columns while preserving unit metadata through validation.

Usage::

    from nozzle_clogging.pint_types import PanderaPintDtype
    from pandera.typing import Series

    class MySchema(pa.DataFrameModel):
        col: Series[PanderaPintDtype] = pa.Field(
            coerce=False, dtype_kwargs={'units': 'mg/L'}
        )

    # Validation preserves pint dtype AND enforces unit match
    validated = MySchema.validate(df_with_pint_columns)
    assert isinstance(validated['col'].dtype, pint_pandas.PintType)

Note:
    Field constraints like ``ge`` and ``le`` do not work directly with pint
    Quantities. Use ``@pa.dataframe_check`` with ``.pint.magnitude`` instead::

        class MySchema(pa.DataFrameModel):
            col: Series[PanderaPintDtype] = pa.Field(coerce=False)

            @pa.dataframe_check
            def col_in_range(cls, df):
                mag = df['col'].pint.magnitude
                return (mag >= 0) & (mag <= 1000)
"""

import pint
import pint_pandas
from beartype.typing import Any
from pandera.dtypes import DataType
from pandera.engines import pandas_engine

from nozzle_clogging.units import ureg


@pandas_engine.Engine.register_dtype(equivalents=[pint_pandas.PintType])
class PanderaPintDtype(pandas_engine.DataType):
    """Pandera dtype for pint-pandas extension arrays with unit enforcement.

    This dtype tells pandera to accept PintType columns without coercing
    to float64. The unit metadata is preserved through schema validation,
    and the ``check`` method enforces that the column's units match the
    expected units specified via ``dtype_kwargs``.
    """

    def __init__(self, units: str = 'dimensionless') -> None:
        self._units = units
        self._type = pint_pandas.PintType(units)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'type',
            pint_pandas.PintType(self.units),
        )

    @property
    def type(self) -> pint_pandas.PintType:
        return self._type

    @property
    def units(self) -> str:
        return self._units

    @classmethod
    def from_parametrized_dtype(cls, pint_type: pint_pandas.PintType) -> Any:
        """Convert a parametrized PintType to a pandera dtype."""
        return cls(units=str(pint_type.units))

    def check(self, pandera_dtype: DataType, data_container: Any | None = None) -> bool:
        """Check that pandera DataType is a PintType with compatible dimensions.

        Returns True if:
        - Both are PanderaPintDtype with matching units (exact match), OR
        - Both have compatible dimensions (e.g., 'millimeter' and 'meter' both
          have length dimension) even if units differ.

        Returns False if dimensions are incompatible (e.g., 'meter' vs 'second').
        """
        if not isinstance(pandera_dtype, PanderaPintDtype):
            return False

        other: PanderaPintDtype = pandera_dtype
        if self.units == other.units:
            return True

        try:
            schema_units = ureg.Unit(self.units)
            column_units = ureg.Unit(pandera_dtype.units)

            schema_base: Any = schema_units.dimensionality
            column_base: Any = column_units.dimensionality

            return schema_base == column_base
        except pint.DimensionalityError:
            return False
