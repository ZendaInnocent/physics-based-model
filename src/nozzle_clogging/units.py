"""Unit definitions for the nozzle-clogging model.

All physical quantities are expressed with explicit units via pint-pandas.

The global UnitRegistry is set once and shared with pint_pandas to ensure
all pint operations use the same registry.
"""

from typing import Any

import pint
import pint_pandas
from pint import UnitRegistry

ureg: UnitRegistry[Any] = UnitRegistry()
Q_: type[pint.Quantity[Any]] = ureg.Quantity

pint_pandas.PintType.ureg = ureg
