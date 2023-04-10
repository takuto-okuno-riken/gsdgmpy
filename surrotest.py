# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.parse_surrotest_options import ParseOptions
import measures


# -------------------------------------------------------------------------
# matrix calculation
def save_mat_file(opt, p, rank, savename):
    out_path = opt.outpath + os.sep + savename
    if opt.format == 1:  # mat all
        f_name = out_path + '.mat'
        print('output mat file : ' + f_name)
        sio.savemat(f_name, {'P': p, 'Rank': rank})
    else:
        # output result matrix csv files
        f_name = out_path + '_pval.csv'
        print('output csv files : ' + f_name)
        np.savetxt(f_name, p, delimiter=',')

        f_name = out_path + '_rank.csv'
        print('output csv files : ' + f_name)
        np.savetxt(f_name, rank, delimiter=',')



# -------------------------------------------------------------------------
# main
if __name__ == '__main__':
    options = ParseOptions()
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

            elif dic.get('X') is not None:
                CX.append(np.float32(dic['X']))
                CXnames.append(name)

        else:
                for j in range(len(CX)):
                    CXnames.append(name+'-'+str(j+1))
        if len(savename) == 0:
            savename = name

    if len(CX) < 2:
        print('original and surrogate files are required. please specify time-series files.')
        options.parser.print_help()
        sys.exit()

    req_num = 39
    if opt.side == 1 or opt.side == 3:
        req_num = 19
    if len(CX) < req_num + 1:
        print('please specify more than '+str(req_num)+' surrogate time-series.')
        options.parser.print_help()
        sys.exit()

    # show input signals
    if opt.showsig:
        plt.figure()
        plt.plot(CX[0].transpose(), linewidth=0.3)
        plt.title('Input time-series : ' + CXnames[0])
        plt.xlabel('Time frames')
        plt.ylabel('Signal value')
        plt.show(block=False)

    plt.pause(1)

    # -------------------------------------------------------------------------
    node_num = CX[0].shape[0]
    if opt.l:
        # output result matrix files
        params = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        h, p, t, rank = measures.surro_test.calc(CX[0], CX[1:len(CX)], func=measures.stat_linear, params=params, side=opt.side)
        print('significantly not linear ('+str(np.sum(h))+' / '+str(node_num)+')')
        save_mat_file(opt, p, rank, savename+'_linear_test')
        if opt.showrank:
            measures.surro_test.plot(p, t, rank, 'Linear')

    if opt.g:
        # output result matrix files
        h, p, t, rank = measures.surro_test.calc(CX[0], CX[1:len(CX)], func=measures.stat_gaussian, params=[], side=opt.side)
        print('significantly not gaussian distribution ('+str(np.sum(h))+' / '+str(node_num)+')')
        save_mat_file(opt, p, rank, savename+'_gaussian_test')
        if opt.showrank:
            measures.surro_test.plot(p, t, rank, 'gaussian distribution')

    if opt.i:
        # output result matrix files
        h, p, t, rank = measures.surro_test.calc(CX[0], CX[1:len(CX)], func=measures.stat_iid, params=[], side=opt.side)
        print('significantly not I.I.D. ('+str(np.sum(h))+' / '+str(node_num)+')')
        save_mat_file(opt, p, rank, savename+'_iid_test')
        if opt.showrank:
            measures.surro_test.plot(p, t, rank, 'I.I.D.')

    plt.pause(1)
    if opt.showrank:
        input("Press Enter to exit...")
