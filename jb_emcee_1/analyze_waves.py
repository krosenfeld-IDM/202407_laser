import pywt
import os
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statsmodels.api as sm


MAX_PERIOD = 7*52 # in weeks
ROOT_DIR = Path(__file__).resolve().parent

print(ROOT_DIR)

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
    # load the cities information
    cities = pd.read_csv( os.path.join('Assets','cities.csv'))
    # load sim output
    sim_output = pd.read_csv( "simulation_output.csv").sort_values(['Node', 'Timestep']).reset_index(drop=True)

    def find_nearby_cities(ref_city='London'):
        ref_node = cities[cities['Name']==ref_city]
        earth_radius = 6371 # km
        nearby_cities = {}
        for _, node in cities.iterrows():
            d = earth_radius*np.pi/180*np.sqrt((node['Latitude'] - ref_node['Latitude'])**2 + \
                np.cos(np.pi/180*ref_node['Latitude'])**2*(node['Longitude'] - ref_node['Longitude'])**2)
            if (d.values[0] < 30) & (d.values[0] > 0):
                nearby_cities[node['ID']] = d.values[0]
        return nearby_cities, ref_node

    london_cities, london_node = find_nearby_cities('London')
    manchester_cities, manchester_node = find_nearby_cities('Manchester')

    ref_city = "London"
    ref_cwt, _ = calc_Ws(sim_output[sim_output['Node'] == london_node['ID'].values[0]]['New_Infections'].values)
    x = []; y = [];
    for city, distance in london_cities.items():
        cwt, frequencies = calc_Ws(sim_output[sim_output['Node'] == city]['New_Infections'].values)
        
        diff = np.conjugate(ref_cwt)*cwt
        ind = np.where(np.logical_and(frequencies < 1/(1.5 * 52), frequencies > 1 / (3 * 52)))
        diff = diff[ind[0], :]

        # # and by time
        # ind = np.where(data.reports > 50)[0]
        # diff = diff[:, ind]

        x.append(distance)
        y.append(np.angle(np.mean(diff)))

    london_x = np.array(x); london_y = np.array(y)

    ref_city = "London"
    ref_cwt, _ = calc_Ws(sim_output[sim_output['Node'] == manchester_node['ID'].values[0]]['New_Infections'].values)
    x = []; y = [];
    for city, distance in manchester_cities.items():
        cwt, frequencies = calc_Ws(sim_output[sim_output['Node'] == city]['New_Infections'].values)
        
        diff = np.conjugate(ref_cwt)*cwt
        ind = np.where(np.logical_and(frequencies < 1/(1.5 * 52), frequencies > 1 / (3 * 52)))
        diff = diff[ind[0], :]

        # # and by time
        # ind = np.where(data.reports > 50)[0]
        # diff = diff[:, ind]

        x.append(distance)
        y.append(np.angle(np.mean(diff)))

    manchester_x = np.array(x); manchester_y = np.array(y)

    def estimate_slope(x,y):
        X = sm.add_constant(x[:, np.newaxis])
        model = sm.OLS(y, X)
        results = model.fit()
        return results.params[1], results.bse[1]


    result_dict = dict()

    plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(1, 2)

    ax0 = plt.subplot(gs[0])
    sns.regplot(x=london_x, y=180/np.pi*london_y, ax=ax0)
    ax0.set_xlabel("Distance from London (km)")
    ax0.set_ylabel("Phase diff from London")

    m,me = estimate_slope(london_x, 180/np.pi*london_y)
    result_dict['London'] = (m,me)

    ax1 = plt.subplot(gs[1])
    sns.regplot(x=manchester_x, y=180/np.pi*manchester_y, ax=ax1)
    ax1.set_xlabel("Distance from Manchester (km)")
    ax1.set_ylabel("Phase diff from Manchester")
    plt.savefig("phase_diff.png")

    m,me = estimate_slope(manchester_x, 180/np.pi*manchester_y)

    result_dict['Manchester'] = (m,me)

    # save result_dict to json
    import json
    with open('result_waves.json', 'w') as f:
        json.dump(result_dict, f, indent=4)


