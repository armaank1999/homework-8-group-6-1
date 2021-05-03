"""
Test cibin.utils functions.

Unless otherwise noted, expected returns are calculated using the equivalent
functions in R.
"""

import pytest
import numpy as np
from ..utils import *


def test_nchoosem():
    """Test nchoosem returns correct list."""
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
    """Test rows of list have correct number of ones."""
    n = 5
    m = 3
    nperm = 10
    Z = combs(n, m, nperm)
    for x in Z:
        assert sum(x) == m


def test_pval_one_lower():
    """Test pval_one_lower returns correct p-value."""
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
    """Test pval_two returns correct p-value."""
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


def test_check_compatible():
    """Check check_compatible returns correct list of booleans."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    N11 = [5, 6, 7]
    N10 = [6, 7, 8]
    N01 = [5, 7, 9]
    compatible = check_compatible(n11, n10, n01, n00, N11, N10, N01)
    expected = [True, True, False]
    assert compatible == expected


def test_tau_lower_N11_oneside():
    """Test tau_lower_N11_oneside returns correct tau_min and N_accept."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N11 = 10
    Z_all = nchoosem(n, m)
    alpha = 0.05
    N11_oneside = tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all, alpha)
    expected_N11_oneside = (-0.15, [10, 0, 3, 7])
    assert N11_oneside == expected_N11_oneside


def test_tau_lower_oneside():
    """Test tau_lower_oneside returns correct tau_lower and tau_upper."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    alpha = 0.05
    nperm = 6000
    lower_oneside = tau_lower_oneside(n11, n10, n01, n00, alpha, nperm)
    expected_lower_oneside = (-0.15, 0.6, [5, 1, 4, 10])
    assert lower_oneside == expected_lower_oneside


def test_tau_lower_N11_twoside():
    """Test tau_lower_N11_twoside returns the correct taus and N_accepts."""
    n11 = 6
    n10 = 4
    n01 = 4
    n00 = 6
    n = n11+n10+n01+n00
    m = n11+n10
    N11 = 10
    Z_all = nchoosem(n, m)
    alpha = 0.05
    N11_twoside = tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha)
    expected_N11_twoside = (-0.2, 0.2, [10, 0, 4, 6], [10, 4, 0, 6], 11)
    assert N11_twoside == expected_N11_twoside
