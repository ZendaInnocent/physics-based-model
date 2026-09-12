import pint
import pint_pandas

# Create shared registry
ureg = pint.UnitRegistry()

# Set up default formatting
ureg.default_format = '~P'

# Register pint-pandas with our shared registry
pint_pandas.PintType.ureg = ureg

__all__ = ['ureg']