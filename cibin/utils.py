"""
Utilities and helper functions.

These functions can be used to construct 2-sided 1 - alpha confidence
bounds for the average treatment effect in a randomized experiment with binary
outcomes and two treatments.
"""

from itertools import combinations
from math import comb, floor
import numpy as np


def nchoosem(n, m):
    """
    Exact re-randomization matrix for small n choose m.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects with treatment

    Returns:
    Z: list
        re-randomization matrix
    """
    c = comb(n, m)
    trt = combinations(np.arange(1, n+1), m)
    Z = [[None]*n for i in np.arange(c)]
    for i in np.arange(c):
        co = next(trt)
        Z[i] = [1 if j in co else 0 for j in np.arange(1, n+1)]
    return Z


def combs(n, m, nperm):
    """
    Sample from re-randomization matrix.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects with treatment
    nperm: int
        number of permutations

    Returns
    -------
    Z: list
        sample from re-randomization matrix
    """
    trt = [[None]*m for i in np.arange(nperm)]
    Z = [[None]*n for i in np.arange(nperm)]
    for i in np.arange(nperm):
        trt[i] = np.random.choice(n, m, replace=False).tolist()
    for i in np.arange(nperm):
        Z[i] = [1 if j in trt[i] else 0 for j in np.arange(n)]
    return Z


def pval_one_lower(n, m, N, Z_all, tau_obs):
    """
    Calculate p-value for method I.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects with treatment
    N: list
        potential table
    Z_all: list
        re-randomization or sample of re-randomization matrix
    tau_obs: float
        observed value of tau

    Returns:
    --------
    pl: float
        p-value
    """
    n_Z_all = len(Z_all)
    dat = [[None]*2 for i in np.arange(n)]
    if N[0] > 0:
        for i in np.arange(N[0]):
            dat[i] = [1]*2
    if N[1] > 0:
        for i in np.arange(N[0], N[0]+N[1]):
            dat[i][0] = 1
            dat[i][1] = 0
    if N[2] > 0:
        for i in np.arange(N[0]+N[1], N[0]+N[1]+N[2]):
            print(i)
            dat[i][0] = 0
            dat[i][1] = 1
    if N[3] > 0:
        for i in np.arange(N[0]+N[1]+N[2], sum(N)):
            dat[i] = [0]*2
    x = [i[0]/m for i in dat]
    y = [i[1]/(n-m) for i in dat]
    a = []
    b = []
    for i in np.arange(len(Z_all)):
        a.append(sum([x[j]*Z_all[i][j] for j in np.arange(len(x))]))
        b.append(sum([(1 - Z_all[i][j])*y[j] for j in np.arange(len(y))]))
    tau_hat = [a[i] - b[i] for i in np.arange(len(a))]
    if isinstance(tau_obs, list):
        count = sum([1 if round(tau_hat[i], 15) >= round(tau_obs[i], 15) else 0
                     for i in np.arange(len(tau_hat))])
    else:
        count = sum([1 if round(tau_hat[i], 15) >= round(tau_obs, 15) else 0
                     for i in np.arange(len(tau_hat))])
    pl = count/n_Z_all
    return pl


def pval_two(n, m, N, Z_all, tau_obs):
    """
    Calculate p-value for method 3.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects with treatment
    N: list
        potential table
    Z_all: list
        re-randomization or sample of re-randomization matrix
    tau_obs: float
        observed value of tau

    Returns:
    --------
    pl: float
        p-value
    """
    n_Z_all = len(Z_all)
    dat = [[None]*2 for i in np.arange(n)]
    if N[0] > 0:
        for i in np.arange(N[0]):
            dat[i] = [1]*2
    if N[1] > 0:
        for i in np.arange(N[0], N[0]+N[1]):
            dat[i][0] = 1
            dat[i][1] = 0
    if N[2] > 0:
        for i in np.arange(N[0]+N[1], N[0]+N[1]+N[2]):
            dat[i][0] = 0
            dat[i][1] = 1
    if N[3] > 0:
        for i in np.arange(N[0]+N[1]+N[2], sum(N)):
            dat[int(i)] = [0]*2
    x = [i[0]/m for i in dat]
    y = [i[1]/(n-m) for i in dat]
    a = []
    b = []
    for i in np.arange(len(Z_all)):
        a.append(sum([x[j]*Z_all[i][j] for j in np.arange(len(x))]))
        b.append(sum([(1 - Z_all[i][j])*y[j] for j in np.arange(len(y))]))
    tau_hat = [a[i] - b[i] for i in np.arange(len(a))]
    tau_N = (N[1] - N[2])/n
    count = sum([1 if round(abs(tau_hat[i] - tau_N), 14) >=
                 round(abs(tau_obs - tau_N), 14) else 0
                 for i in np.arange(len(tau_hat))])
    pd = count/n_Z_all
    return pd
