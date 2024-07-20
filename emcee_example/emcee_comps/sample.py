"""
Single sample
"""
import json
import numpy as np

def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

if __name__ == "__main__":
    
    # load iteraion.json
    with open("iteration.json", "r") as file:
        iteration_dict = json.load(file)
    iteration = iteration_dict['iteration']
    print(f"iteration: {iteration}")

    with open("parameters.json", "r") as file:
        parameters_dict = json.load(file)
    parameters = parameters_dict[str(iteration)]
    print(f"parameters: {parameters}")

    means = np.load("Assets/means.npy")
    cov = np.load("Assets/cov.npy")

    result = log_prob(x=parameters, mu=means, cov=cov)

    # write result to json
    with open("result.json", "w") as file:
        json.dump({"result": result}, file)
