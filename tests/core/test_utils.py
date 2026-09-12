import numpy as np

from nozzle_clogging import utils


class TestBroadcastInputs:
    def test_broadcast_two_arrays(self) -> None:
        a = np.array([1, 2, 3])
        b = np.array([1.0, 1.0, 1.0])
        result = utils.broadcast_inputs(a, b)
        assert len(result) == 2

    def test_broadcast_scalars(self) -> None:
        result = utils.broadcast_inputs(1.0, 2.0)
        assert len(result) == 2
        assert result[0].shape == result[1].shape

    def test_broadcast_mixed(self) -> None:
        a = np.array([1, 2, 3])
        b = 1.0
        result = utils.broadcast_inputs(a, b)
        assert len(result) == 2
        assert result[1].shape == a.shape
