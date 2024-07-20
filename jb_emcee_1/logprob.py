"""
Calculate log prob
"""

import json
import numpy as np

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

if __name__ == "__main__":
    
    # # load walker.json
    # with open("walker.json", "r") as file:
    #     iteration_dict = json.load(file)
    # walker = iteration_dict['walker']
    # print(f"walker: {walker}")

    # with open("parameters.json", "r") as file:
    #     parameters_dict = json.load(file)
    # parameters = parameters_dict[str(walker)]
    # print(f"parameters: {parameters}")

    # https://deepnote.com/workspace/idm-903a8509-b110-4d4d-93be-a752fefd0d6b/project/EandW-data-summary-1b4f4161-7540-4a73-944f-f2180078f64b/notebook/london-phase-e5d6f1d3ebc643fa8d95eb0c3229dc25
    mu = np.array([-1.10, -0.76])
    cov = np.array([[0.15**2, 0.0], [0.0, 0.18**2]])

    with open("result_waves.json", "r") as file:
        results_wave = json.load(file)
    x = np.array([v[0] if np.isfinite(v[0]) else -999 for v in results_wave.values() ])
    result = log_prob(x=x, mu=mu, cov=cov)

    # write result to json
    with open("result.json", "w") as file:
        json.dump({"result": result}, file)
