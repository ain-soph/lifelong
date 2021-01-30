#!/usr/bin/env python3

from trojanplot import *

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    _fig = plt.figure(figsize=[10, 2.5])
    color_list = [google_color['red'], google_color['blue'], google_color['yellow'], google_color['green']]
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
        for j, eps in enumerate([2, 4, 6]):
            fname = f'./result/eps{eps}_{mode}.csv'
            x = np.arange(160) + 1
            y = np.random.random(160) * 4
            if os.path.exists(fname):
                df = pd.read_csv(fname, usecols=['Step', 'Value'])
                x = df['Step']
                y = df['Value']
            y = fig.avg_smooth(y, window=20)
            y = fig.avg_smooth(y, window=10)
            fig.curve(x, y, color=color_list[j], label=f'eps {eps}')
        if mode == 'clean':
            fig.set_legend()
        else:
            fig.ax.get_legend().remove()
            fig.save(path='./result/eps.pdf')
