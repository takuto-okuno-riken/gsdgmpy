# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from utils.parse_mtess_options import ParseOptions
from utils.convert_sigmd import SigmoidConverter
import measures


# -------------------------------------------------------------------------
# matrix calculation
def save_mat_file(opt, mts, mtsp, nmts, nmtsp, CXnames, savename):
    out_path = opt.outpath + os.sep
    if opt.format == 1:  # mat all
        f_name = out_path + savename + '_mtess.mat'
        print('output mat file : ' + f_name)
        sio.savemat(f_name, {'MTS': mts, 'MTSp': mtsp, 'nMTS': nmts, 'nMTSp': nmtsp})
    else:
        # output result MTESS matrix csv file
        f_name = out_path + savename + '_mtess'
        print('output csv files : ' + f_name)
        np.savetxt(f_name + '.csv', mts, delimiter=',')

        # output result MTESS statistical property matrix csv file
        props = ['M', 'SD', 'AC', 'CM', 'PCM', 'CCM', 'PCCM']
        for k in range(len(props)):
            np.savetxt(f_name+'_'+props[k]+'.csv', mtsp[:, :, k], delimiter=',')

        # output result node MTESS matrix csv file
        for k in range(nmts.shape[2]):
            np.savetxt(f_name+'_node'+str(k+1)+'.csv', nmts[:, :, k], delimiter=',')


def get_group_range(cx):
    a = cx[0]
    for i in range(1, len(cx)):
        a = np.concatenate([a, cx[i]])
    r = {
        'min': np.min(a),
        'max': np.max(a),
        'm': np.nanmean(a),
        's': np.nanstd(a)}
    return r


# -------------------------------------------------------------------------
# main
if __name__ == '__main__':
    options = ParseOptions()
    opt = options.parse()

    if type(opt.outpath) is list:
        opt.outpath = opt.outpath[0]  # replaced by string
    if type(opt.range) is list:
        opt.range = opt.range[0]  # replaced by string
    if type(opt.showdend) is list:
        opt.showdend = opt.showdend[0]  # replaced by string
    if type(opt.cachepath) is list:
        opt.cachepath = opt.cachepath[0]  # replaced by string

    # read time-series and control files
    savename = ''
    CXnames = []
    CX = []
    for i in range(len(opt.in_files)):
        if not os.path.isfile(opt.in_files[i]):
            print('bad file name. ignore : ' + opt.in_files[i])
            continue

        print('loading signals : ' + opt.in_files[i])
        name = os.path.splitext(os.path.basename(opt.in_files[i]))[0]
        if '.csv' in opt.in_files[i]:
            csv_input = pd.read_csv(opt.in_files[i], header=None)
            CX.append(np.float32(csv_input.values))
            CXnames.append(name)
        elif '.mat' in opt.in_files[i]:
            dic = sio.loadmat(opt.in_files[i])
            if dic.get('CX') is not None:
                cx = dic.get('CX').flatten()
                if dic.get('multiple') is not None:
                    mlt = np.float32(dic.get('multiple'))  # smaller memory
                    for j in range(len(cx)):
                        x = cx[j] / mlt
                        CX.append(x)
                else:
                    for j in range(len(cx)):
                        CX.append(np.float32(cx[j]))
                if dic.get('names') is not None:
                    names = dic.get('names').flatten()
                    for j in range(len(names)):
                        s = str(names[j])
                        s = s.replace('[', '')  # remove some useless chars
                        s = s.replace(']', '')  # remove some useless chars
                        s = s.replace("'", "")  # remove some useless chars
                        CXnames.append(s)
                else:
                    for j in range(len(CX)):
                        CXnames.append(name + '-' + str(j + 1))

            elif dic.get('X') is not None:
                CX.append(np.float32(dic['X']))
                CXnames.append(name)

        if len(savename) == 0:
            savename = name

    if len(CX) == 0:
        print('no input files. exit script.')
        sys.exit()

    # convert input & exogenous signals
    if opt.transform == 1:
        conv = SigmoidConverter()
        for i in range(len(CX)):
            CX[i], sig, c, max_si, min_si = conv.to_sigmoid_signal(x=CX[i], centroid=opt.transopt)

    # show input signals
    if opt.showinsig:
        for i in range(len(CX)):
            plt.figure()
            plt.plot(CX[i].transpose(), linewidth=0.3)
            plt.title('Input time-series : ' + CXnames[i])
            plt.xlabel('Time frames')
            plt.ylabel('Signal value')
            plt.show(block=False)

    # show raster of input signals
    if opt.showinras:
        for i in range(len(CX)):
            fig, ax = plt.subplots(figsize=(6, 5))
            img = ax.matshow(CX[i], aspect="auto")
            fig.colorbar(img, ax=ax)
            plt.title('Raster plot of input time-series : ' + CXnames[i])
            plt.xlabel('Time frames')
            plt.ylabel('Node number')
            plt.show(block=False)

    plt.pause(1)

    # -------------------------------------------------------------------------
    # set range
    gr = get_group_range(CX)

    mtrange = np.nan
    if opt.range == 'auto':
        mtrange = [gr['m'] - gr['s'] * 3, gr['m'] + gr['s'] * 3]
    elif ':' in opt.range:
        sp = opt.range.split(':')
        if sp[0] == 'sigma':  # <num> sigma of the whole group
            n = float(sp[1])
            mtrange = [gr['m'] - gr['s'] * n, gr['m'] + gr['s'] * n]
        elif sp[0] == 'full':  # <num> * full min & max range of the whole group
            n = (float(sp[1]) - 1) / 2
            r = gr['max'] - gr['min']
            mtrange = [gr['min'] - r*n, gr['max'] + r*n]
        else:  # force [<num>, <num>] range
            mtrange = [float(sp[0]), float(sp[1])]
    else:
        print('bad range option. stop operation.')
        sys.exit()

    # calc MTESS
    cnames = []
    if opt.cache:
        cnames = CXnames
    mts, mtsp, nmts, nmtsp, means, stds, acs, cms, pcms, ccms, pccms = \
        measures.mtess.calc(cx=CX, mtrange=mtrange, n_dft=opt.ndft, cc_lags=opt.cclag, pcc_lags=opt.pcclag,
                            cxnames=cnames, cache_path=opt.cachepath)

    # output result matrix files
    save_mat_file(opt, mts, mtsp, nmts, nmtsp, CXnames, savename)

    # show all matrix
    if opt.showmat:
        measures.mtess.plot_bar3d(mts, savename)
        measures.mtess.plot_all_mat(mts, mtsp, savename)

    # show 1 vs. others signals
    if opt.showsig:
        node_num = CX[0].shape[0]
        for i in range(1, len(CX)):
            fig = plt.figure()
            for j in range(node_num):
                ax1 = fig.add_subplot(node_num, 1, j+1)
                ax1.plot(CX[0][j, :], linewidth=0.3)
                ax1.plot(CX[i][j, :], linewidth=0.3)
                ax1.set_ylim(mtrange[0], mtrange[1])
            plt.show(block=False)
            fig.suptitle('Node signals : '+CXnames[0]+' vs. '+CXnames[i])
            plt.legend([CXnames[0], CXnames[i]], loc='lower right', ncol=2)
            plt.pause(1)

    # show 1 vs. others MTESS statistical properties
    if opt.showprop:
        measures.mtess.plot_radar(mtsp[0, 1:7, :], savename, CXnames[1:7])

    # show 1 vs. others node MTESS
    if opt.shownode:
        measures.mtess.plot_node(nmts[0, 1:7, :], savename, CXnames[1:7])

    # show dendrogram
    if opt.showdend:
        mts[np.isnan(mts)] = 0  # mts might have nan
        mts = mts + mts.transpose()
        e = np.eye(mts.shape[0])
        mask = np.where(e == 1, 0, 1)
        x = (5 - mts) * mask
        condensed_dist_matrix = squareform(x)
        linkage_result = linkage(condensed_dist_matrix, method=opt.showdend, metric='euclidean')
        plt.figure()
        dendrogram(linkage_result, labels=np.arange(mts.shape[0]))
        plt.title('Hierarchical clustering based on MTESS')
        plt.ylabel('MTESS distance')
        plt.xlabel('Cell number')

    plt.pause(1)
    input("Press Enter to exit...")
