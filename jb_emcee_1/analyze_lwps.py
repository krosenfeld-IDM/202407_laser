"""
References:
- https://github.com/krosenfeld-IDM/mcv_sia_timing_v_coverage/blob/d56456cd27c7ac8ca3af288bc1c1f8c33f515fa6/workflow_ri_dynamics/experiment_01/analyze_wavelet.py#L2
= https://github.com/PyWavelets/pywt/blob/main/demo/cwt_analysis.py
"""
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import pandas as pd
# from ew_measles import data

MAX_PERIOD = 7*52 # in weeks

# helper functions

def pad_data(x):
    """
    Pad data to the next power of 2
    """
    nx = len(x) # number of samples
    nx2 = (2**np.ceil(np.log(nx)/np.log(2))).astype(int) # next power of 2
    x2 = np.zeros(nx2, dtype=x.dtype) # pad to next power of 2
    offset = (nx2-nx)//2 # offset
    x2[offset:(offset+nx)] = x # copy
    return x2

def coi_mask(b, T, min_period, max_period):
    """
    Cone of influence mask
    """
    coi = np.tile((np.ptp(T)/2 - np.abs(T - np.mean(T))) / np.sqrt(2), (b.shape[0], 1))
    s = np.tile(np.linspace(min_period, max_period, b.shape[0]), (b.shape[1], 1)).T
    return s >= coi

def log_transform(x, debug=1):
    """
    Log transform for case data
    """ 
    # add one and take log
    x = np.log(x+1)
    # set mean=0 and std=1
    m = np.mean(x)
    s = np.std(x)
    x = (x - m)/s
    return x

if __name__ == "__main__":

    data = pd.read_csv(os.path.join('simulation_output.csv')).sort_values(['Node', 'Timestep'])

    cities = pd.read_csv(os.path.join('Assets', 'cities.csv'))
    london_node = cities[cities['Name'] == 'London']['ID'].values[0]

    cases = data[data['Node'] == london_node]['New_Infections'].to_numpy()
    max_ind = len(cases) // 7
    cases = cases[:max_ind*7].reshape((max_ind, 7)).sum(axis=1)


    # transform case data
    log_cases = pad_data(log_transform(cases))

    # setup and execute wavelet transform
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#morlet-wavelet
    # wavelet = pywt.ContinuousWavelet('morl')
    # wavelet = 'cmor1.5-1.0'
    # wavelet = 'cmor2-3'
    wavelet = pywt.ContinuousWavelet('cmor2-1')

    dt = 1
    widths = np.logspace(np.log10(1), np.log10(MAX_PERIOD), int(MAX_PERIOD))
    [cwt, frequencies] = pywt.cwt(log_cases, widths, wavelet, dt)

    # Number of time steps in padded time series
    nt = len(cases)
    # trim matrix
    offset = (cwt.shape[1] - nt) // 2
    cwt = cwt[:, offset:offset + nt]
    # take norm
    cwt2 = np.real(cwt * np.conj(cwt))

    # Figure 1b, 1c: local and global wavelet power spectrum

    # x = 1900 + data.reports
    x = np.arange(len(cases))
    y = 1 / frequencies / 52
    ylim = [4, 0.5]
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # 2:1 width ratio

    # Subplot 1: Plot local wavelet power spectrum (LWPS) 
    ax  = plt.subplot(gs[0])
    ax.contourf(x, y, cwt2)
    coi = coi_mask(cwt2, x, ylim[1], ylim[0])
    ax.contour(x, y, coi, levels=[0.9], colors='w')
    ax.set_ylim(ylim)
    ax.set_yscale('log')# ax.set_yticks(yticks);
    ax.set_ylabel('Period (years)')
    ax.set_xlabel('Year')
    ylim = ax.get_ylim()

    # subplot 2: global wavelet spectrum
    # ?: I suspect that this average global power so high because we are only looking at vaccinated era
    ax = plt.subplot(gs[1])
    ax.plot(cwt2.mean(axis=1), y)
    ax.vlines(0, *ylim, color=3*[0.7])
    ax.set_ylim(ylim)
    ax.set_yscale('log')
    ax.set_xlabel('Global Power')

    plt.savefig('lwps.png')

    result_dict = {'max_period':y[np.argmax(cwt2.mean(axis=1))]}

    # save result_dict to json
    import json
    with open('result_lwps.json', 'w') as f:
        json.dump(result_dict, f, indent=4)