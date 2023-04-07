# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import urllib.request

from utils.parse_gsdgm_options import ParseOptions
from utils.convert_sigmd import SigmoidConverter

import models
import surrogate


# -------------------------------------------------------------------------
# matrix calculation
def save_result_files(opt, y, outname):
    out_path_f = opt.outpath + os.sep + outname
    if opt.format == 0:  # csv each
        for i in range(y.shape[2]):
            f_name = out_path_f + '_' + str(i+1) + '.csv'
            print('output csv file : ' + f_name)
            np.savetxt(f_name, y[:, :, i], delimiter=',')

    elif opt.format == 1:  # mat each
        for i in range(y.shape[2]):
            f_name = out_path_f + '_' + str(i+1) + '.mat'
            print('output mat file : ' + f_name)
            sio.savemat(f_name, {'X': y[:, :, i]})

    elif opt.format == 2:  # mat all
        f_name = out_path_f + '_all.mat'
        print('output mat file : ' + f_name)
        names = np.empty((y.shape[2]), dtype=object)
        cx = np.empty((y.shape[2]), dtype=object)
        for i in range(y.shape[2]):
            cx[i] = y[:, :, i]
            names[i] = outname+'_'+str(i+1)
        sio.savemat(f_name, {'CX': cx, 'names': names})


def save_model_file(opt, net, gr, outname):
    f_name = opt.outpath + os.sep + outname + '.mat'
    print('output group surrogate model file : ' + f_name)
    # convert to dictionary
    dic = dict()
    dic['nodeNum'] = net.node_num
    dic['sigLen'] = net.sig_len
    dic['exNum'] = net.ex_num
    dic['lags'] = net.lags
    dic['cxM'] = net.cx_m
    dic['cxCov'] = net.cx_cov
    bvec = np.empty(net.node_num, dtype=object)
    rvec = np.empty(net.node_num, dtype=object)
    for k in range(net.node_num):
        b = np.concatenate([net.lr_objs[k].coef_.flatten(), [net.lr_objs[k].intercept_]])
        bvec[k] = b
        rvec[k] = net.residuals[k]  # list to nested array (cell)
    dic['bvec'] = bvec
    dic['rvec'] = rvec
    sio.savemat(f_name, {'net': dic, 'gRange': gr})


def url2cache_string(url):
    u = url.split('?')
    u = u[0].replace('http://', '')
    u = u.replace('https://', '')
    u = u.split('/')
    b = u[0].replace('.', '_')
    for j in range(1, len(u)):
        b = b+'-'+u[j]
    return b


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


def get_group_range_dic(dic):
    if type(dic) is np.ndarray:
        r = {
            'min': dic['min'][0, 0][0, 0],
            'max': dic['max'][0, 0][0, 0],
            'm': dic['m'][0, 0][0, 0],
            's': dic['s'][0, 0][0, 0]}
    else:  # h5py
        r = {
            'min': dic['min'][0, 0],
            'max': dic['max'][0, 0],
            'm': dic['m'][0, 0],
            's': dic['s'][0, 0]}
    return r

# -------------------------------------------------------------------------
# main
if __name__ == '__main__':
    options = ParseOptions()
    opt = options.parse()

    if type(opt.outpath) is list:
        opt.outpath = opt.outpath[0]  # replaced by string
    if type(opt.noise) is list:
        opt.noise = opt.noise[0]  # replaced by string
    if type(opt.range) is list:
        opt.range = opt.range[0]  # replaced by string

    # read time-series or model file
    CX = []
    CXnames = []
    mat_net = None
    savename = ''
    for i in range(len(opt.in_files)):
        infile = opt.in_files[i]

        # check url or file
        if 'http://' in infile or 'https://' in infile:
            # make download cache directory
            if not os.path.exists('data/cache'):
                os.mkdir('data/cache')
            # make cache file string
            url = opt.in_files[i]
            infile = 'data/cache/'+url2cache_string(url)
            if not os.path.isfile(infile):
                print('downloading '+url+' ...')
                urllib.request.urlretrieve(url, infile)
                print('save cache file : '+infile)

        flist = []
        if os.path.isfile(infile):
            flist.append(infile)
        else:
            flist = os.listdir(infile)
        if len(flist) == 0:
            print('file is not found. ignoring : ' + infile)
            continue
        for k in range(len(flist)):
            # init data
            infile = flist[k]

            # load multivariate time-series csv or mat file
            name = os.path.splitext(os.path.basename(infile))[0]
            if '.mat' in infile:
                try:
                    dic = sio.loadmat(infile)
                except NotImplementedError:  # -v3.7
                    dic = h5py.File(infile, 'r')

                if dic.get('CX') is not None:
                    # training mode
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

                elif dic.get('net') is not None:
                    # surrogate data mode
                    mat_net = dic['net']
                    if dic.get('gRange') is not None:
                        mat_range = dic['gRange']
                else:
                    print('file does not contain "X" matrix or "CX" cell. ignoring : '+flist)

            elif '.csv' in infile:
                # training mode
                csv_input = pd.read_csv(infile, header=None)
                CX.append(np.float32(csv_input.values))
                CXnames.append(name)

        if len(savename) == 0:
            savename = name

    # check each multivariate time-series
    for i in range(len(CX)):
        x = CX[i]
        # convert input & exogenous signals
        if opt.transform == 1:
            conv = SigmoidConverter()
            x, sig, c, max_si, min_si = conv.to_sigmoid_signal(x=x, centroid=opt.transopt)
            CX[i] = x

        # show input signals
        if opt.showinsig:
            plt.figure()
            plt.plot(x.transpose(), linewidth=0.3)
            plt.title('Input time-series : ' + CXnames[i])
            plt.xlabel('Time frames')
            plt.ylabel('Signal value')
            plt.show(block=False)

        # show raster of input signals
        if opt.showinras:
            fig, ax = plt.subplots(figsize=(6, 5))
            img = ax.matshow(x, aspect="auto")
            fig.colorbar(img, ax=ax)
            plt.title('Raster plot of input time-series : ' + CXnames[i])
            plt.xlabel('Time frames')
            plt.ylabel('Node number')
            plt.show(block=False)

        plt.pause(1)

    # ------------------------------------------------------------------------------
    # training mode
    net = None
    if len(CX) > 0:
        gr = get_group_range(CX)

        # generate model data
        if opt.var:
            ntype = 'var'
            net = models.MultivariateVARNetwork()
            net.init_with_cell(CX, lags=opt.lag)
            save_model_file(opt, net, gr, savename+'_gsm_'+ntype)

    # ------------------------------------------------------------------------------
    # surrogate data mode
    if mat_net is not None:
        # currently, we only support var network
        ntype = 'var'
        net = models.MultivariateVARNetwork()
        net.init_with_matnet(mat_net)
        gr = get_group_range_dic(mat_range)

    if net is not None:
        if opt.siglen > 0:
            sig_len = opt.siglen
        else:
            sig_len = net.sig_len
        # generate dummy input time-series
        x = np.random.multivariate_normal(mean=net.cx_m, cov=net.cx_cov, size=sig_len).transpose()

        # set range
        yrange = np.nan
        if opt.range == 'auto':  # 4.2 sigma of the whole group
            yrange = [gr['m'] - gr['s'] * 4.2, gr['m'] + gr['s'] * 4.2]
        elif opt.range == 'none':  # no range limit
            yrange = []
        elif ':' in opt.range:
            sp = opt.range.split(':')
            if sp[0] == 'sigma':  # <num> sigma of the whole group
                n = float(sp[1])
                yrange = [gr['m'] - gr['s'] * n, gr['m'] + gr['s'] * n]
            elif sp[0] == 'full':  # <num> * full min & max range of the whole group
                n = (float(sp[1]) - 1) / 2
                r = gr['max'] - gr['min']
                yrange = [gr['min'] - r * n, gr['max'] + r * n]
            else:  # force [<num>, <num>] range
                yrange = [float(sp[0]), float(sp[1])]
        else:
            print('bad range option. stop operation.')
            sys.exit()

        if ntype == 'var':
            y = surrogate.multivariate_var(x, net, surr_num=opt.surrnum, dist=opt.noise, y_range=yrange)
            save_result_files(opt, y, savename + '_gsd_' + ntype)

        # show surrogate signals
        if opt.showsig:
            for i in range(y.shape[2]):
                x = y[:, :, i]
                plt.figure()
                plt.plot(x.transpose(), linewidth=0.3)
                plt.title('Group Surrogate Data : ' + savename+'-'+ntype+'-'+str(i+1))
                plt.xlabel('Time frames')
                plt.ylabel('Signal value')
                plt.show(block=False)

        # show raster of input signals
        if opt.showras:
            for i in range(y.shape[2]):
                x = y[:, :, i]
                fig, ax = plt.subplots(figsize=(6, 5))
                img = ax.matshow(x, aspect="auto")
                fig.colorbar(img, ax=ax)
                plt.title('Raster plot of Group Surrogate Data : ' + savename+'-'+ntype+'-'+str(i+1))
                plt.xlabel('Time frames')
                plt.ylabel('Node number')
                plt.show(block=False)
    plt.pause(1)
