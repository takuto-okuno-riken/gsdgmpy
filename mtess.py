# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.parse_mtess_options import ParseMtessOptions
from utils.convert_sigmd import SigmoidConverter
import measures
import models

# -------------------------------------------------------------------------
# matrix calculation

def save_mat_file(opt, mts, mtsp, nmts, nmtsp, CXnames, savename):
    out_path = opt.outpath + os.sep
    if opt.format == 1:  # mat all
        f_name = out_path + savename + '_mtess.mat'
        print('saving matrix : ' + f_name)
        sio.savemat(f_name, {'MTS': mts, 'MTSp': mtsp, 'nMTS': nmts, 'nMTSp': nmtsp})
    else:
        # output result MTESS matrix csv file
        f_name = out_path + savename + '_mtess'
        print('saving matrix : ' + f_name)
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
    options = ParseMtessOptions()
    opt = options.parse()

    if type(opt.outpath) is list:
        opt.outpath = opt.outpath[0]  # replaced by string

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
            CX.append(csv_input.values)
            CXnames.append(name)
        elif '.mat' in opt.in_files[i]:
            dic = sio.loadmat(opt.in_files[i])
            cx = dic.get('CX').flatten()
            names = dic.get('names').flatten()
            if cx is not None and len(cx[:]) > 0:
                for j in range(len(cx[:])):
                    CX.append(cx[j])
                    s = str(names[j])
                    s = s.replace('[', '')  # remove some useless chars
                    s = s.replace(']', '')  # remove some useless chars
                    s = s.replace("'", "")  # remove some useless chars
                    CXnames.append(s)
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
    elif ':' in opt.range[0]:
        sp = opt.range[0].split(':')
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
    mts, mtsp, nmts, nmtsp, means, stds, acs, cms, pcms, ccms, pccms = \
        measures.mtess.calc(cx=CX, mtrange=mtrange, n_dft=opt.ndft, cc_lags=opt.cclag, pcc_lags=opt.pcclag)

    # output result matrix files
    save_mat_file(opt, mts, mtsp, nmts, nmtsp, CXnames, savename)

    # show all matrix
    if opt.showmat:
        measures.mtess.plot_bar3d(mts, savename)
        measures.mtess.plot_all_mat(mts, mtsp, savename)

    # show 1 vs. others MTESS statistical properties
    if opt.showprop:
        measures.mtess.plot_radar(mtsp[0, 1:7, :], savename, CXnames[1:7])

    # show 1 vs. others node MTESS
    if opt.shownode:
        measures.mtess.plot_node(nmts[0, 1:7, :], savename, CXnames[1:7])

    plt.pause(1)
    input("Press Enter to exit...")
