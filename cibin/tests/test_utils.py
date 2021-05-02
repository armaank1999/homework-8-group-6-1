"""
Test cibin.utils functions.

Unless otherwise noted, expected returns are calculated using the equivalent
functions in R.
"""

import pytest
import numpy as np
from ..utils import nchoosem


def test_nchoosem():
    """Test that nchoosem returns correct list."""
    n = 5
    m = 3
    Z_all = nchoosem(n, m)
    expected = [[1, 1, 1, 0, 0], [1, 1, 0, 1, 0],
                [1, 1, 0, 0, 1], [1, 0, 1, 1, 0],
                [1, 0, 1, 0, 1], [1, 0, 0, 1, 1],
                [0, 1, 1, 1, 0], [0, 1, 1, 0, 1],
                [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]]
    assert Z_all == expected
