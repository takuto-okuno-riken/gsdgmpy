# -*- coding: utf-8 -*-
##
# Calculate partial cross correlation matrix (simple version)
# returns 3D matrix (node x node x max_lag*2+1)
# input:
#  X         multivariate time series matrix (node x time series)
#  max_lag   time lag number (default: 2)

from __future__ import print_function, division

import timeit
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


def in_node_calc(i, node_num, sig_len, max_lag, ulen, x, model):
    if model == 'ridge':
        lr = Ridge(fit_intercept=True, alpha=1e-9, solver='cholesky')  # this is much faster
    else:
        lr = LinearRegression(fit_intercept=True)

    xi = x[i, :].transpose()
    pccmj = np.zeros((node_num, max_lag*2+1))
    for j in range(i, node_num):
        if ulen[i] == 1 and ulen[j] == 1:
            continue

        idx = np.arange(node_num)
        if j > i:
            idx = np.delete(idx, j)
        idx = np.delete(idx, i)
        xtj = x[idx, :].transpose()
        xj = x[j, :].transpose()

        lr.fit(xtj, xi)
        pred = lr.predict(xtj)
        r1 = (xi - pred)
        lr.fit(xtj, xj)
        pred = lr.predict(xtj)
        r2 = (xj - pred)

        #            r1n = r1 - np.mean(r1)  # this may not be necessary
        #            r2n = r2 - np.mean(r2)  # this may not be necessary
        cc = scipy.signal.correlate(r1, r2, mode='full')
        cc /= np.linalg.norm(r1, ord=2) * np.linalg.norm(r2, ord=2)
        pccmj[j, :] = cc[(sig_len - 1 - max_lag):(sig_len + max_lag)]
    return i, pccmj


def calc(x, max_lag=2, model='ridge', concurrent_type='thread'):
    node_num = x.shape[0]
    sig_len = x.shape[1]
    pccm = np.zeros((node_num, node_num, max_lag*2+1), dtype=x.dtype)
    tic = timeit.default_timer()
    print('start to calc pccm ... ', end='')

    # check all same value or not
    ulen = np.zeros(node_num)
    for i in range(node_num):
        ulen[i] = np.unique(x[i, :]).size

    futures = []
    if concurrent_type == 'thread':
        with ThreadPoolExecutor() as executor:
            for i in range(node_num):
                futures.append(executor.submit(in_node_calc, i, node_num, sig_len, max_lag, ulen, x, model))
    else:
        with ProcessPoolExecutor() as executor:
            for i in range(node_num):
                futures.append(executor.submit(in_node_calc, i, node_num, sig_len, max_lag, ulen, x, model))
    for i in range(len(futures)):
        a = futures[i].result()
        pccm[a[0], :, :] = a[1]

    e = np.ones((node_num, node_num), dtype=x.dtype)
    mask = np.tril(e, k=-1)
    for p in range(-max_lag, max_lag):
        b = pccm[:, :, max_lag - p]
        b = b.transpose() * mask
        pccm[:, :, max_lag + p] += b
    toc = timeit.default_timer()
    print('done t='+format(toc-tic, '3f'))
    return pccm


def plot(x, max_lag=2):
    ccm = calc(x=x, max_lag=max_lag)
    plt.matshow(ccm[:, :, max_lag], vmin=-1, vmax=1)
    plt.colorbar()
    plt.xlabel('Source Nodes')
    plt.ylabel('Target Nodes')
    plt.show(block=False)
    plt.pause(1)
    return ccm
