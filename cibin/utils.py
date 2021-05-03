"""
Utilities and helper functions.

These functions can be used to construct 2-sided 1 - alpha confidence
bounds for the average treatment effect in a randomized experiment with binary
outcomes and two treatments.
"""

import numpy as np
from scipy.stats import hypergeom
from itertools import combinations
from math import comb, floor


def nchoosem(n, m):
    """
    Exact re-randomization matrix for small n choose m.

    Parameters
    ----------
    n: int
        total number of subjects
    m: int
        number of subjects under treatment

    Returns
    -------
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
        number of subjects under treatment
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
        number of subjects under treatment
    N: list
        potential table
    Z_all: list
        re-randomization or sample of re-randomization matrix
    tau_obs: float
        observed value of tau

    Returns
    -------
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
            dat[i] = [0]*2
    x = [i[0]/m for i in dat]
    y = [i[1]/(n-m) for i in dat]
    a = []
    b = []
    for i in np.arange(len(Z_all)):
        a.append(sum([x[j]*Z_all[i][j] for j in np.arange(len(x))]))
        b.append(sum([(1-Z_all[i][j])*y[j] for j in np.arange(len(y))]))
    tau_hat = [a[i]-b[i] for i in np.arange(len(a))]
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
        number of subjects under treatment
    N: list
        potential table
    Z_all: list
        re-randomization or sample of re-randomization matrix
    tau_obs: float
        observed value of tau

    Returns
    -------
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
        b.append(sum([(1-Z_all[i][j])*y[j] for j in np.arange(len(y))]))
    tau_hat = [a[i]-b[i] for i in np.arange(len(a))]
    tau_N = (N[1]-N[2])/n
    count = sum([1 if round(abs(tau_hat[i]-tau_N), 14) >=
                 round(abs(tau_obs-tau_N), 14) else 0
                 for i in np.arange(len(tau_hat))])
    pd = count/n_Z_all
    return pd


def check_compatible(n11, n10, n01, n00, N11, N10, N01):
    """
    Check that observed table is compatible with potential table.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    N11: list of integers
        potential number of subjects under treatment that experienced
        outcome 1
    N10: list of integers
        potential number of subjects under treatment that experienced
        outcome 0
    N01: list of integers
        potential number of subjects under treatment that experienced
        outcome 0

    Returns
    -------
    compact: list
        booleans indicating compatibility of inputs
    """
    n = n11+n10+n01+n00
    n_t = len(N10)
    lefts = [[] for i in np.arange(len(N10))]
    rights = [[] for i in np.arange(len(N10))]
    for i in np.arange(len(N10)):
        lefts[i].append(0)
        rights[i].append(N11[i])
        lefts[i].append(n11 - N10[i])
        rights[i].append(n11)
        lefts[i].append(N11[i]-n01)
        rights[i].append(N11[i]+N01[i]-n01)
        lefts[i].append(N11[i]+N01[i]-n10-n01)
        rights[i].append(n-N10[i]-n01-n10)
    left = [max(x) for x in lefts]
    right = [min(x) for x in rights]
    compact = [left[i] <= right[i] for i in np.arange(len(left))]
    return compact


def tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Calculate tau_min and N_accept for method I.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    N11: int
        potential number of subjects under treatment that experienced
        outcome 1
    Z_all: list
        re-randomization or sample of re-randomization matrix
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_min: float
        minimum tau value of accepted potential tables
    N_accept:
        accepted potential table
    """
    n = n11+n10+n01+n00
    m = n11+n10
    N01 = 0
    N10 = 0
    tau_obs = n11/m - n01/(n-m)
    M = [0]*(n-N11+1)
    while (N10 <= (n-N11-N01)) & (N01 <= (n-N11)):
        pl = pval_one_lower(n, m, [N11, N10, N01, n-(N11 + N10 + N01)],
                            Z_all, tau_obs)
        if pl >= alpha:
            M[N01] = N10
            N01 += 1
        else:
            N10 += 1
    if N01 <= (n - N11):
        for i in np.arange(N01, (n-N11+1)):
            M[i] = n+1
    N11_vec0 = [N11]*(n-N11+1)
    N10_vec0 = M
    N01_vec0 = np.arange((n-N11+1)).tolist()
    N11_vec = []
    N10_vec = []
    N01_vec = []
    for i in np.arange(len(N11_vec0)):
        if N10_vec0[i] <= (n - N11_vec0[i] - N01_vec0[i]):
            N10_vec.extend(np.arange(N10_vec0[i], n-N11_vec0[i]-N01_vec0[i]+1)
                           .tolist())
            N11_vec.extend([N11_vec0[i]]*(n-N11_vec0[i]-N01_vec0[i]-N10_vec0[i]
                                          + 1))
            N01_vec.extend([N01_vec0[i]]*(n-N11_vec0[i]-N01_vec0[i]-N10_vec0[i]
                                          + 1))
    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)
    if sum(compat) > 0:
        N10_vec_compat = [N10_vec[i] for i in np.arange(len(N10_vec)) if
                          compat[i]]
        N01_vec_compat = [N01_vec[i] for i in np.arange(len(N10_vec)) if
                          compat[i]]
        tau_min = min([N10_vec_compat[i]-N01_vec_compat[i] for i in
                       np.arange(len(N10_vec_compat))])/n
        accept_pos = [i for i in np.arange(len(N10_vec_compat)) if
                      N10_vec_compat[i]-N01_vec_compat[i] == n*tau_min]
        accept_pos = accept_pos[0]
        N_accept = [N11, N10_vec_compat[accept_pos],
                    N01_vec_compat[accept_pos],
                    n-(N11+N10_vec_compat[accept_pos] +
                       N01_vec_compat[accept_pos])]
    else:
        tau_min = (n11 + n00)/n
        N_accept = float('NaN')
    return (tau_min, N_accept)


def tau_lower_oneside(n11, n10, n01, n00, alpha, nperm):
    """
    Calculate tau_lower, tau_upper, and N_accept for method I.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    alpha: float
        1 - confidence level
    nperm: int
        maximum desired number of permutations

    Returns
    -------
    tau_lower: float
        left-side tau for one-sided confidence interval
    tau_upper: float
        right-side tau for one-sided confidence interval
    N_accept:
        accepted potential table for one-sided confidence interval
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if comb(n, m) <= nperm:
        Z_all = nchoosem(n, m)
    else:
        Z_all = combs(n, m, nperm)
    tau_min = (n11+n00)/n
    N_accept = float('NaN')
    for N11 in np.arange(n11+n01+1):
        tau_min_N11 = tau_lower_N11_oneside(n11, n10, n01, n00, N11, Z_all,
                                            alpha)
        if tau_min_N11[0] < tau_min:
            N_accept = tau_min_N11[1]
        tau_min = min(tau_min, tau_min_N11[0])
    tau_lower = tau_min
    tau_upper = (n11+n00)/n
    return (tau_lower, tau_upper, N_accept)


def tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Calculate tau_min and N_accept for method 3.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    N11: int
        potential number of subjects under treatment that experienced
        outcome 1
    Z_all: list
        re-randomization or sample of re-randomization matrix
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_min: float
        minimum tau value of accepted potential tables
    tau_max: float
        minimum tau value of accepted potential tables
    N_accept_min: list
        minimum accepted potential table
    N_accept_max: list
        maximum accepted potential table
    rand_test_num: int
        number of tests run
    """
    n = n11 + n10 + n01 + n00
    m = n11 + n10
    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n*n11/m - n*n01/(n-m)
    N10 = 0
    N01_vec0 = [i for i in np.arange(n-N11+1) if i >= (-ntau_obs)]
    N01 = min(N01_vec0)
    M = [float('NaN')]*len(N01_vec0)
    rand_test_num = 0
    while (N10 <= (n-N11-N01)) & (N01 <= (n-N11)):
        if N10 <= (N01+ntau_obs):
            pl = pval_two(n, m, [N11, N10, N01, n-(N11 + N10 + N01)], Z_all,
                          tau_obs)
            rand_test_num += 1
            if pl >= alpha:
                for i in np.arange(len(M)):
                    if N01_vec0[i] == N01:
                        M[i] = N10
                N01 += 1
            else:
                N10 += 1
        else:
            for i in np.arange(len(M)):
                if N01_vec0[i] == N01:
                    M[i] = N10
            N01 += 1
    if N01 <= (n-N11):
        for i in np.arange(len(M)):
            if N01_vec0[i] >= N01:
                M[i] = floor(N01_vec0[i]+ntau_obs)+1
    N11_vec0 = [N11]*len(N01_vec0)
    N10_vec0 = M
    N11_vec = []
    N10_vec = []
    N01_vec = []
    for i in np.arange(len(N11_vec0)):
        N10_upper = min((n-N11_vec0[i]-N01_vec0[i]), floor(N01_vec0[i] +
                                                           ntau_obs))
        if N10_vec0[i] <= N10_upper:
            N10_vec.extend(np.arange(N10_vec0[i], N10_upper+1).tolist())
            N11_vec.extend([N11_vec0[i]]*(N10_upper-N10_vec0[i]+1))
            N01_vec.extend([N01_vec0[i]]*(N10_upper-N10_vec0[i]+1))
    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)
    if sum(compat) > 0:
        N10_vec_compat = [N10_vec[i] for i in np.arange(len(N10_vec)) if
                          compat[i]]
        N01_vec_compat = [N01_vec[i] for i in np.arange(len(N10_vec)) if
                          compat[i]]
        tau_min = min([N10_vec_compat[i]-N01_vec_compat[i] for i in
                       np.arange(len(N10_vec_compat))])/n
        accept_pos = [i for i in np.arange(len(N10_vec_compat)) if
                      N10_vec_compat[i]-N01_vec_compat[i] == n*tau_min]
        accept_pos = accept_pos[0]
        N_accept_min = [N11, N10_vec_compat[accept_pos],
                        N01_vec_compat[accept_pos],
                        n-(N11+N10_vec_compat[accept_pos] +
                           N01_vec_compat[accept_pos])]

        tau_max = max([N10_vec_compat[i] - N01_vec_compat[i] for i in
                       np.arange(len(N10_vec_compat))])/n
        accept_pos = [i for i in np.arange(len(N10_vec_compat)) if
                      N10_vec_compat[i]-N01_vec_compat[i] == n*tau_max]
        accept_pos = accept_pos[0]
        N_accept_max = [N11, N10_vec_compat[accept_pos],
                        N01_vec_compat[accept_pos],
                        n-(N11+N10_vec_compat[accept_pos] +
                           N01_vec_compat[accept_pos])]
    else:
        tau_min = float('inf')
        N_accept_min = float('NaN')
        tau_max = float('-inf')
        N_accept_max = float('NaN')
    return (tau_min, tau_max, N_accept_min, N_accept_max, rand_test_num)


def tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all):
    """
    Calculate taus and N_accepts for method 3.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    alpha: float
        1 - confidence level
    Z_all: list
        re-randomization or sample of re-randomization matrix

    Returns
    -------
    tau_lower: float
        left-side tau for two-sided confidence interval
    N_accept_lower: list
        left-side accepted potential table for two-sided confidence interval
    tau_upper: float
        right-side tau for two-sided confidence interval
    N_accept_upper: list
        right-side accepted potential table for two-sided confidence interval
    rand_test_total: int
        number of tests run
    """
    n = n11+n10+n01+n00
    m = n11+n10
    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n*n11/m - n*n01/(n-m)
    tau_min = float('inf')
    tau_max = float('-inf')
    N_accept_min = float('NaN')
    N_accept_max = float('NaN')
    rand_test_total = 0
    for N11 in np.arange(int(min(n11+n01, n+ntau_obs))+1):
        tau_min_N11 = tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all,
                                            alpha)
        rand_test_total = rand_test_total + tau_min_N11[4]
        if tau_min_N11[0] < tau_min:
            N_accept_min = tau_min_N11[2]
        if tau_min_N11[1] > tau_max:
            N_accept_max = tau_min_N11[3]
        tau_min = min(tau_min, tau_min_N11[0])
        tau_max = max(tau_max, tau_min_N11[1])
    tau_lower = tau_min
    tau_upper = tau_max
    N_accept_lower = N_accept_min
    N_accept_upper = N_accept_max
    return (tau_lower, N_accept_lower, tau_upper, N_accept_upper,
            rand_test_total)


def tau_twoside_less_treated(n11, n10, n01, n00, alpha, nperm):
    """
    Calculate taus and N_accepts for method 3.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    alpha: float
        1 - confidence level
    nperm: int
        maximum desired number of permutations

    Returns
    -------
    tau_lower: float
        left-side tau for two-sided confidence interval
    tau_upper: float
        right-side tau for two-sided confidence interval
    N_accept_lower: list
        left-side accepted potential table for two-sided confidence interval
    N_accept_upper: list
        right-side accepted potential table for two-sided confidence interval
    rand_test_total: int
        number of tests run
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if comb(n, m) <= nperm:
        Z_all = nchoosem(n, m)
    else:
        Z_all = combs(n, m, nperm)
    ci_lower = tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all)
    ci_upper = tau_twoside_lower(n10, n11, n00, n01, alpha, Z_all)
    rand_test_total = ci_lower[4] + ci_upper[4]
    tau_lower = min(ci_lower[0], -1*ci_upper[2])
    tau_upper = max(ci_lower[2], -1*ci_upper[0])
    if tau_lower == ci_lower[0]:
        N_accept_lower = ci_lower[1]
    else:
        N_accept_lower = ci_upper[3][::-1]
    if tau_upper == -1*ci_upper[0]:
        N_accept_upper = ci_upper[1][::-1]
    else:
        N_accept_upper = ci_lower[3]
    return (tau_lower, tau_upper, N_accept_lower, N_accept_upper,
            rand_test_total)


def tau_twoside(n11, n10, n01, n00, alpha, nperm):
    """
    Calculate taus and N_accepts for method 3.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    alpha: float
        1 - confidence level
    nperm: int
        maximum desired number of permutations

    Returns
    -------
    tau_lower: float
        left-side tau for two-sided confidence interval
    tau_upper: float
        right-side tau for two-sided confidence interval
    N_accept_lower: list
        left-side accepted potential table for two-sided confidence interval
    N_accept_upper: list
        right-side accepted potential table for two-sided confidence interval
    rand_test_total: int
        number of tests run
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if m > (n/2):
        ci = tau_twoside_less_treated(n01, n00, n11, n10, alpha, nperm)
        tau_lower = -ci[1]
        tau_upper = -ci[0]
        N_accept_lower = ci[2]
        N_accept_upper = ci[3]
        rand_test_total = ci[4]
    else:
        ci = tau_twoside_less_treated(n11, n10, n01, n00, alpha, nperm)
        tau_lower = ci[0]
        tau_upper = ci[1]
        N_accept_lower = ci[2]
        N_accept_upper = ci[3]
        rand_test_total = ci[4]
    return (tau_lower, tau_upper, N_accept_lower, N_accept_upper,
            rand_test_total)


def exact_CI_odd(N, n, x, alpha):
    """
    Calculate exact CI for odd sample size.

    Parameters
    ----------
    N: int
        population size
    n: int (odd)
        sample size
    x: int
        number of good items in the sample
    alpha:
        1 - confidence level

    Returns
    -------
    lower: int
        lower bound of confidence interval
    upper: int
        upper bound of confidence interval
    """
    def ind(x, a, b):
        return (x >= a)*(x <= b)

    def lci(x, n, alpha):
        if isinstance(x, int):
            xs = [x]
        else:
            xs = x
        kk = np.arange(0, len(xs)).tolist()
        for i in kk:
            if xs[i] < 0.5:
                kk[i] = 0
            else:
                aa = np.arange(0, N+1).tolist()
                bb = [x+1 for x in aa]
                bb[1:(N+1)] = hypergeom.cdf(xs[i]-1, N, [x-1 for x in
                                                         aa[1:(N+1)]], n)
                cc = []
                cc.append(aa)
                cc.append(bb)
                inds = [i >= (1-alpha) for i in cc[1]]
                dd = [[], []]
                dd[0] = [cc[0][i] for i in np.arange(len(inds)) if inds[i]]
                dd[1] = [cc[1][i] for i in np.arange(len(inds)) if inds[i]]
                if len(dd[0])*len(dd) == 2:
                    kk[i] = dd[1][0]
                else:
                    kk[i] = max(dd[0])
        if isinstance(x, int):
            return kk[0]
        else:
            return kk

    def uci(x, n, alpha):
        if isinstance(x, int):
            xs = [x]
        else:
            xs = x
        lcis = lci([n-i for i in xs], n, alpha)
        upper = [N - i for i in lcis]
        if isinstance(x, int):
            return upper[0]
        else:
            return upper
    xx = np.arange(n+1)
    lcin1 = lci(xx, n, alpha/2)
    ucin1 = uci(xx, n, alpha/2)
    lcin2 = lci(xx, n, alpha)
    ucin2 = uci(xx, n, alpha)
    lciw = lcin1
    uciw = ucin1
    xvalue = int(floor(n/2)+1)
    while xvalue > 0.5:
        al = lcin2[xvalue-1]-lciw[xvalue-1]+1
        au = int(uciw[xvalue-1] - ucin2[xvalue-1]+1)
        if al*au > 1:
            gg = [[], [], [], []]
            for i in np.arange(al):
                gg[0].append([lciw[xvalue-1]+i]*au)
                gg[1].append(list(np.arange(ucin2[xvalue-1],
                                            uciw[xvalue-1]+1)))
            ff = [[], [], [], []]
            ff[0] = list(np.concatenate(gg[0]))
            ff[1] = list(np.concatenate(gg[1]))
            ff[2] = [ff[1][i] - ff[0][i] for i in np.arange(len(ff[0]))]
            for ii in np.arange(len(ff[0])):
                lciw[xvalue-1] = ff[0][ii]
                uciw[xvalue-1] = ff[1][ii]
                lciw[n+1-xvalue] = N - uciw[xvalue-1]
                uciw[n+1-xvalue] = N - lciw[xvalue-1]

                def cpci(M):
                    kk = list(np.arange(len(M)))
                    for i in kk:
                        xx = list(np.arange(n+1))
                        indp = xx
                        uu = 0
                        while (uu < n + 0.5):
                            indp[uu] = (ind(M[i], lciw[uu], uciw[uu]) *
                                        hypergeom.pmf(uu, N, M[i], n))
                            uu += 1
                        kk[i] = sum(indp)
                    return kk
                M = np.arange(N+1)
                ff[3].append(min(cpci(M)))
            ff = np.array(ff).T.tolist()
            ff[:] = [x for x in ff if x[3] >= (1-alpha)]
            if len(ff[0])*len(ff) > 4:
                ff = sorted(ff, key=lambda x: x[2])
                lciw[xvalue-1] = ff[0][0]
                uciw[xvalue-1] = ff[0][1]
            else:
                lciw[xvalue-1] = ff[0][0]
                uciw[xvalue-1] = ff[0][1]
            lciw[n+1-xvalue] = N - uciw[xvalue-1]
            uciw[n+1-xvalue] = N - lciw[xvalue-1]
        xvalue = xvalue - 1

    def cpcig(M, lcin, ucin):
        kk = list(np.arange(len(M)))
        for i in kk:
            xx = list(np.arange(n+1))
            indp = xx
            uu = 0
            while (uu < n + 0.5):
                indp[uu] = ((ind(M[i], lcin[uu], ucin[uu]) *
                             hypergeom.pmf(uu, N, M[i], n)))
                uu += 1
            kk[i] = sum(indp)
        return kk
    lower = int([lciw[i] for i in xx if i == x][0])
    upper = int([uciw[i] for i in xx if i == x][0])
    return (lower, upper)


def exact_CI_even(N, n, x, alpha):
    """
    Calculate exact CI for even sample size.

    Parameters
    ----------
    N: int
        population size
    n: int (even)
        sample size
    x: int
        number of good items in the sample
    alpha:
        1 - confidence level

    Returns
    -------
    lower: int
        lower bound of confidence interval
    upper: int
        upper bound of confidence interval
    """
    def ind(x, a, b):
        return (x >= a)*(x <= b)

    def lci(x, n, alpha):
        if isinstance(x, int):
            xs = [x]
        else:
            xs = x
        kk = np.arange(0, len(xs)).tolist()
        for i in kk:
            if xs[i] < 0.5:
                kk[i] = 0
            else:
                aa = np.arange(0, N+1).tolist()
                bb = [x+1 for x in aa]
                bb[1:(N+1)] = hypergeom.cdf(xs[i]-1, N, [x-1 for x in
                                                         aa[1:(N+1)]], n)
                cc = []
                cc.append(aa)
                cc.append(bb)
                inds = [i >= (1-alpha) for i in cc[1]]
                dd = [[], []]
                dd[0] = [cc[0][i] for i in np.arange(len(inds)) if inds[i]]
                dd[1] = [cc[1][i] for i in np.arange(len(inds)) if inds[i]]
                if len(dd[0])*len(dd) == 2:
                    kk[i] = dd[1][0]
                else:
                    kk[i] = max(dd[0])
        if isinstance(x, int):
            return kk[0]
        else:
            return kk

    def uci(x, n, alpha):
        if isinstance(x, int):
            xs = [x]
        else:
            xs = x
        lcis = lci([n-i for i in xs], n, alpha)
        upper = [N - i for i in lcis]
        if isinstance(x, int):
            return upper[0]
        else:
            return upper
    xx = np.arange(n+1)
    lcin1 = lci(xx, n, alpha/2)
    ucin1 = uci(xx, n, alpha/2)
    lcin2 = lci(xx, n, alpha)
    ucin2 = uci(xx, n, alpha)
    lciw = lcin1
    uciw = ucin1
    xvalue = int((n/2)+1)
    aa = np.arange(lciw[xvalue-1], floor(N/2)+1)
    ii = 1
    while ii < (len(aa) + 0.5):
        lciw[xvalue-1] = aa[ii - 1]
        uciw[xvalue-1] = N - aa[ii - 1]

        def cpci(M):
            kk = list(np.arange(len(M)))
            for i in kk:
                xx = list(np.arange(n+1))
                indp = xx
                uu = 0
                while (uu < n+0.5):
                    indp[uu] = (ind(M[i], lciw[uu], uciw[uu]) *
                                hypergeom.pmf(uu, N, M[i], n))
                    uu += 1
                kk[i] = sum(indp)
            return kk
        M = list(np.arange(N+1))
        bb = min(cpci(M))
        if (bb >= 1-alpha):
            ii1 = ii
            ii += 1
        else:
            ii = len(aa) + 1
    lciw[xvalue-1] = aa[ii1-1]
    uciw[xvalue-1] = N - lciw[xvalue-1]
    xvalue = int(n/2)
    while xvalue > 0.5:
        al = lcin2[xvalue-1] - lciw[xvalue-1]+1
        au = int(uciw[xvalue-1] - ucin2[xvalue-1]+1)
        if al*au > 1:
            gg = [[], [], [], []]
            for i in np.arange(al):
                gg[0].append([lciw[xvalue-1]+i]*au)
                gg[1].append(list(np.arange(ucin2[xvalue-1],
                                            uciw[xvalue-1]+1)))
            ff = [[], [], [], []]
            ff[0] = list(np.concatenate(gg[0]))
            ff[1] = list(np.concatenate(gg[1]))
            ff[2] = [ff[1][i] - ff[0][i] for i in np.arange(len(ff[0]))]
            for ii in np.arange(len(ff[0])):
                lciw[xvalue-1] = ff[0][ii]
                uciw[xvalue-1] = ff[1][ii]
                lciw[n+1-xvalue] = N - uciw[xvalue-1]
                uciw[n+1-xvalue] = N - lciw[xvalue-1]

                def cpci(M):
                    kk = list(np.arange(len(M)))
                    for i in kk:
                        xx = list(np.arange(n+1))
                        indp = xx
                        uu = 0
                        while (uu < n + 0.5):
                            indp[uu] = (ind(M[i], lciw[uu], uciw[uu]) *
                                        hypergeom.pmf(uu, N, M[i], n))
                            uu += 1
                        kk[i] = sum(indp)
                    return kk
                M = np.arange(N+1)
                ff[3].append(min(cpci(M)))
            ff = np.array(ff).T.tolist()
            ff[:] = [x for x in ff if x[3] >= (1-alpha)]
            if len(ff[0])*len(ff) > 4:
                ff = sorted(ff, key=lambda x: x[2])
                lciw[xvalue-1] = ff[0][0]
                uciw[xvalue-1] = ff[0][1]
            else:
                lciw[xvalue-1] = ff[0][0]
                uciw[xvalue-1] = ff[0][1]
            lciw[n+1-xvalue] = N - uciw[xvalue-1]
            uciw[n+1-xvalue] = N - lciw[xvalue-1]
        xvalue = int(xvalue - 1)
    lower = [lciw[i] for i in xx if i == x][0]
    upper = [uciw[i] for i in xx if i == x][0]
    return (lower, upper)


def exact_CI(N, n, x, alpha):
    """
    Calculate exact CI for even or odd sample size.

    Parameters
    ----------
    N: int
        population size
    n: int (even)
        sample size
    x: int
        number of good items in the sample
    alpha:
        1 - confidence level

    Returns
    -------
    lower: int
        lower bound of confidence interval
    upper: int
        upper bound of confidence interval
    """
    if n % 2 == 1:
        ci = exact_CI_odd(N, n, x, alpha)
    else:
        ci = exact_CI_even(N, n, x, alpha)
    lower = int(ci[0])
    upper = int(ci[1])
    return (lower, upper)


def combin_exact_CI(n11, n10, n01, n00, alpha):
    """
    Calculate exact CI from observed table.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_lower: float
        left-side tau for one-sided confidence interval
    tau_upper: float
        right-side tau for one-sided confidence interval
    """
    n = n11+n10+n01+n00
    m = n11+n10
    ci_1plus = exact_CI(N=n, n=m, x=n11, alpha=alpha/2)
    ci_plus1 = exact_CI(N=n, n=(n-m), x=n01, alpha=alpha/2)
    tau_upper = (ci_1plus[1]-ci_plus1[0])/n
    tau_lower = (ci_1plus[0]-ci_plus1[1])/n
    return (tau_lower, tau_upper)


def N_plus1_exact_CI(n11, n10, n01, n00, alpha):
    """
    Calculate optimal exact CI from observed table.

    Parameters
    ----------
    n11: int
        number of subjects under treatment that experienced outcome 1
    n10: int
        number of subjects under treatment that experienced outcome 0
    n01: int
        number of subjects under control that experienced outcome 1
    n00: int
        number of subjects under control that experienced outcome 0
    alpha: float
        1 - confidence level

    Returns
    -------
    tau_lower: float
        left-side tau for one-sided confidence interval
    tau_upper: float
        right-side tau for one-sided confidence interval
    """
    n = n11+n10+n01+n00
    m = n11+n10
    tau_min = float('inf')
    tau_max = float('-inf')
    ci_plus1 = exact_CI(N=n, n=(n-m), x=n01, alpha=alpha)
    for N_plus1 in np.arange(ci_plus1[0], ci_plus1[1]+1):
        for N11 in np.arange(N_plus1+1):
            N01 = N_plus1-N11
            for N10 in np.arange(n-N_plus1+1):
                N00 = n-N_plus1-N10
                if check_compatible(n11, n10, n01, n00, [N11], [N10],
                                    [N01])[0]:
                    tau = (N10-N01)/n
                    tau_min = min(tau, tau_min)
                    tau_max = max(tau, tau_max)
    upper = tau_max
    lower = tau_min
    return (lower, upper)
