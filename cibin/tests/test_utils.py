"""
Test cibin.utils functions.

Unless otherwise noted, expected returns are calculated using the equivalent
functions in R.
"""

import pytest
import numpy as np
from ..utils import *


def test_nchoosem():
    """Test that nchoosem returns correct list."""
    n = 5
    m = 3
    Z = nchoosem(n, m)
    expected = [[1, 1, 1, 0, 0], [1, 1, 0, 1, 0],
                [1, 1, 0, 0, 1], [1, 0, 1, 1, 0],
                [1, 0, 1, 0, 1], [1, 0, 0, 1, 1],
                [0, 1, 1, 1, 0], [0, 1, 1, 0, 1],
                [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]]
    assert Z == expected


def test_combs():
    """Test that rows of list have correct number of ones."""
    n = 5
    m = 3
    nperm = 10
    Z = combs(n, m, nperm)
    for x in Z:
        assert sum(x) == m


def test_pval_one_lower():
    """Test that pval_one_lower returns correct p-value."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    N11 = 2
    N = [N11, N10, N01, n-(N11+N10+N01)]
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)
    pval = pval_one_lower(n, m, N, Z_all, tau_obs)
    expected = 0.23684210526315788
    assert pval == expected


def test_pval_two():
    """Test that pval_two returns correct p-value."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    N11 = 2
    N = [N11, N10, N01, n-(N11+N10+N01)]
    Z_all = nchoosem(n, m)
    tau_obs = n11/m - n01/(n-m)
    pval = pval_two(n, m, N, Z_all, tau_obs)
    expected = 0.47368421052631576
    assert pval == expected
