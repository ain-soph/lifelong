#!/usr/bin/env python3

from trojanplot import *

import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    _fig = plt.figure(figsize=[10, 2.5])
    color_list = [google_color['red'], google_color['blue'], google_color['yellow'], google_color['green'],
                  ting_color['purple']]
    for i, mode in enumerate(['clean', 'robust']):
        _ax = _fig.add_subplot(1, 2, i + 1)
        fig = Figure(mode, fig=_fig, ax=_ax)
        fig.set_axis_label('x', 'Step')
        fig.set_axis_label('y', 'Acc')
        fig.set_axis_lim('x', lim=[0, 160], piece=8, margin=[1, 1],
                         _format='%d')
        if mode == 'clean':
            fig.set_axis_lim('y', lim=[20, 80], piece=5, margin=[0.0, 2.0],
                             _format='%d')
        elif mode == 'robust':
            fig.set_axis_lim('y', lim=[0, 50], piece=5, margin=[0.0, 2.0],
                             _format='%d')
        fig.set_title()
        for j, eps in enumerate([0, 2, 4, 6, 8]):
            fname_list = glob.glob(f'./result/eps/eps{eps}_{mode}*.csv')
            x_list = []
            y_list = []
            for fname in fname_list:
                df = pd.read_csv(fname, usecols=['Step', 'Value'])
                x: np.ndarray = df['Step']
                y: np.ndarray = df['Value']
                # y = fig.avg_smooth(y, window=20)
                # y = fig.avg_smooth(y, window=10)
                x_list.append(x)
                y_list.append(y)
            x_list = np.concatenate(x_list)
            y_list = np.concatenate(y_list)
            fig.curve(x_list, y_list, color=color_list[j], label=f'eps {eps}')
        if mode == 'clean':
            fig.set_legend()
        else:
            fig.ax.get_legend().remove()
            fig.save(path='./result/eps.pdf')
