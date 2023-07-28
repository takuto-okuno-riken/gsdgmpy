# -*- coding: utf-8 -*-
##
# Caluclate MTESS, MTESS statistical properties, Node MTESS and Node MTESS statistical properties
# returns MTESS matrix (cell number x cell number)(MTS), MTESS statistical property matrix (cell number x cell number x 7)(MTSp),
#   Node MTESS (cell number x cell number x node)(nMTS) and Node MTESS statistical properties (cell number x cell number x node x 7)(nMTSp).
#   Data in the middle of calculation, such as mean (Means), standard deviation (Stds), auto-correlation (AC), partial auto-correlation (PAC),
#   correlation matrix (FCs), partial correlation matrix (PCs), cross-correlation matrix (CCs), partial cross-correlation matrix (PCCs) and multivariate kurtosis (mKT).
# input:
#  cx               cells of multivariate time series matrix {(node x time series)} x cell number (time series length can be different)
#  mtrange          mtess range [min, max] of time series for normalized mean and std dev (default: min and max of input CX)
#  ac_lags          time lags for Auto-Correlation function (default: 15)
#  cc_lags          time lags for Cross-Correlation function (default: 4)
#  pcc_lags         time lags for Partial Cross-Correlation function (default: 2)
#  cxnames          CX signals names used for cache filename (default: {})
#  cache_path       cache file path (default: 'results/cache')

from __future__ import print_function, division

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import measures


def calc(cx, mtrange=np.nan, ac_lags=5, pac_lags=13, cc_lags=2, pcc_lags=4, cxnames=[], cache_path='results'+os.sep+'cache'):
    clen = len(cx)
    node_num = cx[0].shape[0]
    # check data file. node num should be same.
    for i in range(1, clen):
        if cx[i].shape[0] != node_num:
            print('Error : input time series should have same node number.')
            return

    # find time series range
    if len(mtrange) == 1 and np.isnan(mtrange):
        minv = np.min(cx[0])
        maxv = np.max(cx[0])
        for i in range(2,clen):
            v = np.min(cx[i])
            if v < minv:
                minv = v
            v = np.max(cx[i])
            if v > maxv:
                maxv = v
        mtrange = [minv, maxv]
    trange = mtrange[1] - mtrange[0]  # should be positive

    if len(cxnames) > 0:
        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)

    # calc statistical properties
    means = np.zeros((clen, node_num))
    stds = np.zeros((clen, node_num))
    acs = np.zeros((clen, node_num, ac_lags+1))
    pacs = np.zeros((clen, node_num, pac_lags+1))
    cms = np.zeros((clen, node_num, node_num))
    pcms = np.zeros((clen, node_num, node_num))
    ccms = np.zeros((clen, node_num, node_num, 2*cc_lags+1))
    pccms = np.zeros((clen, node_num, node_num, 2*pcc_lags+1))
    mkts = np.zeros(clen)
    for nn in range(clen):
        x = cx[nn]
        cachef = ''
        if len(cxnames) > 0:
            cachef = cache_path + os.sep + 'mtess-' + cxnames[nn] + '-' + str(x.shape[0])+'x'+str(x.shape[1])\
                     + 'a'+str(ac_lags)+'c'+str(cc_lags)+'p'+str(pcc_lags)+'.mat'
        if len(cachef) > 0 and os.path.isfile(cachef):
            print('load cache of '+cachef)
            dic = sio.loadmat(cachef)
            xm = dic.get('xm')
            xsd = dic.get('xsd')
            xac = dic.get('xac')
            xpac = dic.get('xpac')
            xcc = dic.get('xcc')
            xpcc = dic.get('xpcc')
            xmkt = dic.get('xmkt')
        else:
            xm = np.mean(x, axis=1)
            xsd = np.std(x, axis=1)
            xac = measures.ac.calc(x=x, max_lag=ac_lags)
            xpac = measures.pac.calc(x=x, max_lag=pac_lags)
            xcc = measures.ccm.calc(x=x, max_lag=cc_lags)
            xpcc = measures.pccm.calc(x=x, max_lag=pcc_lags)
            xmsw, xmkt = measures.mskewkurt.calc(x=x)
            if len(cachef) > 0:
                print('save cache of ' + cachef)
                sio.savemat(cachef, {'xm': xm, 'xsd': xsd, 'xac': xac, 'xpac': xpac, 'xcc': xcc, 'xpcc': xpcc, 'xmkt': xmkt})  # compatible with matlab version
        means[nn, :] = xm
        stds[nn, :] = xsd
        acs[nn, :, :] = xac
        pacs[nn, :, :] = xpac
        cms[nn, :, :] = xcc[:, :, cc_lags]
        pcms[nn, :, :] = xpcc[:, :, pcc_lags]
        ccms[nn, :, :, :] = xcc
        pccms[nn, :, :, :] = xpcc
        mkts[nn] = xmkt

    # calc MTESS
    item_num = 8
    ne = np.empty((node_num, node_num))
    ne[:] = np.nan
    nanx = np.tril(ne, 0)
    nanxcc = np.repeat(nanx[:, :, np.newaxis], cc_lags*2, axis=2)
    nanxpcc = np.repeat(nanx[:, :, np.newaxis], pcc_lags*2, axis=2)
    mtsp = np.empty((clen, clen, item_num))
    nmtsp = np.empty((clen, clen, node_num, item_num))
    mtsp[:] = np.nan
    nmtsp[:] = np.nan
    ccidx = np.r_[0:cc_lags, (cc_lags + 1):(2 * cc_lags + 1)]
    pccidx = np.r_[0:pcc_lags, (pcc_lags + 1):(2 * pcc_lags + 1)]
    for i in range(clen):
        b = np.empty((clen, item_num))
        nb = np.empty((clen, node_num, item_num))
        b[:] = np.nan
        nb[:] = np.nan
        for j in range(i+1, clen):
            c = np.empty(item_num)
            nc = np.empty((node_num, item_num))
            c[:] = np.nan
            nc[:] = np.nan
            # calc std dev difference
            ds = stds[i, :] - stds[j, :]
            d = 5 * (1 - np.abs(ds) / (trange / 4))
            d[d < 0] = 0
            c[0] = np.nanmean(d)
            nc[:, 0] = d

            # calc AC/PAC simirality
            a1 = acs[i, :, 1:ac_lags]
            a2 = acs[j, :, 1:ac_lags]
            c[1] = 5 * measures.cos_sim(a1, a2)
            for k in range(node_num):
                nc[k, 1] = 5 * measures.cos_sim(a1[k, :], a2[k, :])
            a1 = pacs[i, :, 1:pac_lags]
            a2 = pacs[j, :, 1:pac_lags]
            c[2] = 5 * measures.cos_sim(a1, a2)
            for k in range(node_num):
                nc[k, 2] = 5 * measures.cos_sim(a1[k, :], a2[k, :])

            # calc zero-lag covariance similarity
            cm1 = cms[i, :, :] + nanx
            cm2 = cms[j, :, :] + nanx
            c[3] = 5 * measures.cos_sim(cm1, cm2)
            for k in range(node_num):
                nc[k, 3] = 5 * measures.cos_sim(np.r_[cm1[k, :], np.transpose(cm1[:, k])],
                                                np.r_[cm2[k, :], np.transpose(cm2[:, k])])

            # calc zero-lag partial covariance similarity
            pcm1 = pcms[i, :, :] + nanx
            pcm2 = pcms[j, :, :] + nanx
            c[4] = 5 * measures.cos_sim(pcm1, pcm2)
            for k in range(node_num):
                nc[k, 4] = 5 * measures.cos_sim(np.r_[pcm1[k, :], np.transpose(pcm1[:, k])],
                                                np.r_[pcm2[k, :], np.transpose(pcm2[:, k])])

            # calc cross-covariance simirality
            ccm1 = ccms[i, :, :, :]
            ccm1 = ccm1[:, :, ccidx] + nanxcc
            ccm2 = ccms[j, :, :, :]
            ccm2 = ccm2[:, :, ccidx] + nanxcc
            c[5] = 5 * measures.cos_sim(ccm1, ccm2)
            for k in range(node_num):
                r1 = np.r_[ccm1[k, :, :], ccm1[:, k, :]]
                r2 = np.r_[ccm2[k, :, :], ccm2[:, k, :]]
                nc[k, 5] = 5 * measures.cos_sim(r1, r2)

            # calc partial cross-covariance simirality
            pccm1 = pccms[i, :, :, :]
            pccm1 = pccm1[:, :, pccidx] + nanxpcc
            pccm2 = pccms[j, :, :, :]
            pccm2 = pccm2[:, :, pccidx] + nanxpcc
            c[6] = 5 * measures.cos_sim(pccm1, pccm2)
            for k in range(node_num):
                r1 = np.r_[pccm1[k, :, :], pccm1[:, k, :]]
                r2 = np.r_[pccm2[k, :, :], pccm2[:, k, :]]
                nc[k, 6] = 5 * measures.cos_sim(r1, r2)

            # calc multivariate kurtosis difference
            mkt = node_num * (node_num + 2) / 2    # 2 is empirically defined.
            ds = 5 * (1 - abs(mkts[i] - mkts[j]) / mkt)    # normalize
            if ds < 0:
                ds = 0
            c[7] = ds
            nc[:, 7] = ds

            b[j, :] = c
            nb[j, :, :] = nc
        mtsp[i, :, :] = b
        nmtsp[i, :, :, :] = nb

    # calc MTESS & Node MTESS
    mtsp[mtsp < 0] = 0
    nmtsp[nmtsp < 0] = 0
    mtsp[mtsp > 5] = 5  # this may happen because of decimal point calculation
    nmtsp[nmtsp > 5] = 5  # this may happen because of decimal point calculation
    mts = np.nanmean(mtsp, axis=2)
    nmts = np.nanmean(nmtsp, axis=3)

    return mts, mtsp, nmts, nmtsp, means, stds, acs, pacs, cms, pcms, ccms, pccms, mkts


def plot(mts, outname):
    plt.matshow(mts, vmin=0, vmax=5)
    plt.title('MTESS - '+outname)
    plt.colorbar()
    plt.xlabel('Cell number')
    plt.ylabel('Cell number')
    plt.show(block=False)
    plt.pause(1)


def plot_bar3d(mts, outname):
    vals = np.arange(mts.shape[0])
    X, Y = np.meshgrid(vals, vals)
    y = X.flatten()
    x = Y.flatten()
    z = np.zeros_like(x)
    dx = np.repeat(a=0.8, repeats=len(x))
    dy = np.repeat(a=0.8, repeats=len(y))
    mts[np.isnan(mts)] = 0
    dz = mts.reshape((mts.shape[0]*mts.shape[1]))

    cm = plt.get_cmap('jet')  # color map

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz,
             color=cm(dz / 5), alpha=0.5)
    ax.set_xlabel('Cell number')
    ax.set_ylabel('Cell number')
    ax.set_zlabel('MTESS')
    ax.set_title('MTESS - '+outname)
    plt.show(block=False)
    plt.pause(1)


def plot_all_mat(mts, mtsp, outname):
    sz = mts.shape[0]
    s = np.empty((sz, sz, 5))
    mts[np.isnan(mts)] = 0
    mtsp[np.isnan(mtsp)] = 0
    s[:, :, 0] = np.transpose(mtsp[:, :, 0]) + mts
    s[:, :, 1] = np.transpose(mtsp[:, :, 2]) + mtsp[:, :, 1]
    s[:, :, 2] = np.transpose(mtsp[:, :, 4]) + mtsp[:, :, 3]
    s[:, :, 3] = np.transpose(mtsp[:, :, 6]) + mtsp[:, :, 5]
    s[:, :, 4] = mtsp[:, :, 7]

    prop2 = ['MTESS/SD', 'AC/PAC', 'CM/PCM', 'CCM/PCCM', 'mKT/-']
    for i in range(5):
        # show MTESS prop matrix
        plt.matshow(s[:, :, i], vmin=0, vmax=5)
        plt.title(prop2[i]+' - '+outname)
        plt.colorbar()
        plt.xlabel('Cell number')
        plt.ylabel('Cell number')
        plt.show(block=False)
        plt.pause(1)


# pm      Node MTESS plot (data x node num)
def plot_node(nm, outname, dnames=[]):
    x = np.arange(nm.shape[1]) + 1
    plt.figure()
    plt.plot(x, nm.transpose(), marker='o', fillstyle='none', linestyle=':')
    plt.title('Node MTESS - ' + outname)
    plt.ylim(0, 5)
    plt.xlabel('node number')
    plt.ylabel('MTESS')
    plt.legend(dnames, loc='lower right')
    plt.show(block=False)
    plt.pause(1)


# pm      MTESS statistical properties matrix (cell number x 8)
def plot_radar(pm, outname, dnames=[]):
    rgrids = np.arange(6)
    labels = ['SD', 'AC', 'PAC', 'CM', 'PCM', 'CCM', 'PCCM', 'mKT']
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(1, 1, 1, polar=True)
    for i in range(pm.shape[0]):
        rv = np.concatenate([pm[i, :], [pm[i, 0]]])
        ax.plot(angles, rv)
#        ax.fill(angles, rv, alpha=0.2)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_rgrids([])
    ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for grid_value in rgrids:
        grid_values = [grid_value] * (len(labels)+1)
        ax.plot(angles, grid_values, color="gray",  linewidth=0.5)

    for t in rgrids:
        ax.text(x=0, y=t, s=t)

    ax.set_rlim([min(rgrids), max(rgrids)])
    ax.set_title('MTESS radar chart - '+outname)
    if len(dnames) == 0:
        dnames = np.arange(pm.shape[0])
    plt.legend(dnames, loc='upper left', bbox_to_anchor=(1, 1))
    plt.show(block=False)
    plt.pause(1)
