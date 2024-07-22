"""
Calculate log prob
"""

import json
import numpy as np

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

if __name__ == "__main__":

    # https://deepnote.com/workspace/idm-903a8509-b110-4d4d-93be-a752fefd0d6b/project/EandW-data-summary-1b4f4161-7540-4a73-944f-f2180078f64b/notebook/london-phase-e5d6f1d3ebc643fa8d95eb0c3229dc25
    mu = np.array([-1.10, -3.6, -0.76, -4.8])
    cov = np.diag([0.15, 2.9, 0.18, 3.2])**2

    with open("result_waves.json", "r") as file:
        results_wave = json.load(file)
    x = np.array([v[0] for v in results_wave.values() ]).flatten()
    x[~np.isfinite(x)] = -np.inf

    result = log_prob(x=x, mu=mu, cov=cov)

    # write result to json
    with open("result.json", "w") as file:
        json.dump({"result": result}, file)
