#!/usr/bin/env python3

from trojanplot import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    _fig = plt.figure(figsize=[10, 2.5])
    color_list = [google_color['red'], google_color['blue']]
    for i, mode in enumerate(['clean', 'adv']):
        _ax = _fig.add_subplot(1, 2, i + 1)
        fig = Figure(mode, fig=_fig, ax=_ax)
        fig.set_axis_label('x', 'Step')
        fig.set_axis_label('y', 'Acc')
        fig.set_axis_lim('x', lim=[0, 160], piece=8, margin=[1, 1],
                         _format='%d')
        fig.set_axis_lim('y', lim=[0, 100], piece=5, margin=[0.0, 5.0],
                         _format='%d')
        fig.set_title()
        for j, method in enumerate(['robust_agem', 'robust_agem_cluster']):
            df = pd.read_csv(f'./result/{method} {mode}.csv', usecols=['Step', 'Value'])
            fig.curve(df['Step'], df['Value'], color=color_list[j], label=method)
        fig.set_legend()
    fig.save(folder_path='./result/', ext='.pdf')
