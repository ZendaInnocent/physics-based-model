from collections.abc import Sequence

import numpy as np
import pandas as pd


def broadcast_inputs(
    *arrays: np.ndarray | int | float | Sequence[float] | pd.Series,
) -> tuple[np.ndarray, ...]:
    """Safely broadcast multiple inputs to the same shape.

    Converts scalars to arrays and applies NumPy broadcasting so that
    all returned arrays share identical shapes.

    Parameters
    ----------
    *arrays : array-like
        Scalars, lists, or numpy arrays

    Returns
    -------
    tuple of numpy arrays
        Broadcasted arrays with identical shapes
    """

    _arrays = [np.asarray(a) for a in arrays]
    return np.broadcast_arrays(*_arrays)
