import pywt
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statsmodels.api as sm


MAX_PERIOD = 7*52 # in weeks
ROOT_DIR = Path(__file__).resolve().parent

input_root="."
if os.getenv( "INPUT_ROOT" ):
    input_root=os.getenv( "INPUT_ROOT" )
sys.path.append( input_root )
sys.path.append( "Assets" )

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

def calc_Ws(cases):
    # transform case data
    log_cases = pad_data(log_transform(cases))

    # setup and execute wavelet transform
    # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#morlet-wavelet
    wavelet = pywt.ContinuousWavelet('cmor2-1')

    dt = 1 # 1 week
    widths = np.logspace(np.log10(1), np.log10(MAX_PERIOD), int(MAX_PERIOD))
    [cwt, frequencies] = pywt.cwt(log_cases, widths, wavelet, dt)

    # Number of time steps in padded time series
    nt = len(cases)
    # trim matrix
    offset = (cwt.shape[1] - nt) // 2
    cwt = cwt[:, offset:offset + nt]

    return cwt, frequencies

if __name__ == "__main__":

    # change directory to the parent of this file
    # os.chdir(ROOT_DIR)

    # load the cities information
    # cities = pd.read_csv( os.path.join('Assets','cities.csv'))
    distances = np.load(os.path.join('Assets', 'engwaldist.npy'))
    from engwaldata import data  # noqa: E402, I001

    # list files in output directory that end in .npy
    files = [f for f in os.listdir("outputs") if f.endswith(".npy")]
    # load sim output
    sim_output = np.load(os.path.join("outputs", files[0]))

    # identify which locations are within 30km of London
    ref_city = "London"
    j = data.placenames.index(ref_city)
    london_cities = {}
    for placename in data.placenames:
        i = data.placenames.index(placename)
        if i == j:
            continue
        if distances[j, i] < 30:
            london_cities.update({placename: distances[j, i]})
    print(f"Found {len(london_cities)} cities within 30km of {ref_city}.")
    london_node = j

    # identify which locations are within 30km of Manchester
    ref_city = "Manchester"
    j = data.placenames.index(ref_city)
    manchester_cities = {}
    for placename in data.placenames:
        i = data.placenames.index(placename)
        if i == j:
            continue
        if distances[j, i] < 30:
            manchester_cities.update({placename: distances[j, i]})
    print(f"Found {len(manchester_cities)} cities within 30km of {ref_city}.")
    manchester_node = j

    ref_city = "London"
    ref_cwt, _ = calc_Ws(sim_output[:, 1, data.placenames.index(ref_city)].flatten())
    x = []; y = [];
    for city, distance in london_cities.items():
        cwt, frequencies = calc_Ws(sim_output[:, 1, data.placenames.index(city)].flatten())
        
        diff = np.conjugate(ref_cwt)*cwt
        ind = np.where(np.logical_and(frequencies < 1/(1.5 * 52), frequencies > 1 / (3 * 52)))
        diff = diff[ind[0], :]

        # and by time
        ind = np.where(data.reports > 50)[0]
        diff = diff[:, ind]

        x.append(distance)
        y.append(np.angle(np.mean(diff)))

    london_x = np.array(x); london_y = np.array(y)

    ref_city = "Manchester"
    ref_cwt, _ = calc_Ws(sim_output[:, 1, data.placenames.index(ref_city)].flatten())
    x = []; y = [];
    for city, distance in manchester_cities.items():
        cwt, frequencies = calc_Ws(sim_output[:, 1, data.placenames.index(city)].flatten())
        
        diff = np.conjugate(ref_cwt)*cwt
        ind = np.where(np.logical_and(frequencies < 1/(1.5 * 52), frequencies > 1 / (3 * 52)))
        diff = diff[ind[0], :]

        # and by time
        ind = np.where(data.reports > 50)[0]
        diff = diff[:, ind]

        x.append(distance)
        y.append(np.angle(np.mean(diff)))

    manchester_x = np.array(x); manchester_y = np.array(y)

    def estimate_slope(x,y):
        X = sm.add_constant(x[:, np.newaxis])
        model = sm.OLS(y, X)
        results = model.fit()
        return results.params, results.bse

    result_dict = dict()

    plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(1, 2)

    ax0 = plt.subplot(gs[0])
    sns.regplot(x=london_x, y=180/np.pi*london_y, ax=ax0)
    ax0.set_xlabel("Distance from London (km)")
    ax0.set_ylabel("Phase diff from London")

    ind = np.isfinite(london_y)
    london_x = london_x[ind]
    london_y = london_y[ind]
    if ind.sum() > 2:
        p,pe = estimate_slope(london_x, 180/np.pi*london_y)
    else:
        p = [-np.inf, -np.inf]
        pe = [np.inf, np.inf]
    result_dict['London_m'] = (p[1],pe[1])
    result_dict['London_b'] = (p[0],pe[0])

    ax1 = plt.subplot(gs[1])
    sns.regplot(x=manchester_x, y=180/np.pi*manchester_y, ax=ax1)
    ax1.set_xlabel("Distance from Manchester (km)")
    ax1.set_ylabel("Phase diff from Manchester")
    plt.savefig("phase_diff.png")

    ind = np.isfinite(manchester_y)
    manchester_x = manchester_x[ind]
    manchester_y = manchester_y[ind]
    if ind.sum() > 2:
        p,pe = estimate_slope(manchester_x, 180/np.pi*manchester_y)
    else:
        p = [-np.inf, -np.inf]
        pe = [np.inf, np.inf]
    result_dict['Manchester_m'] = (p[1],pe[1])
    result_dict['Manchester_b'] = (p[0],pe[0])

    # save result_dict to json
    import json
    with open('result_waves.json', 'w') as f:
        json.dump(result_dict, f, indent=4)


