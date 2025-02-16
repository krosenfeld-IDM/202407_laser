"""
https://emcee.readthedocs.io/en/stable/tutorials/monitor/
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner

# change working directory to the root of this file
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

class Pool():
    @staticmethod
    def map(func, iterable):
        return map(func, iterable)

if __name__ == "__main__":
    # number of dimenstions to the problem
    ndim = 5

    np.random.seed(42)
    means = np.random.rand(ndim)

    cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    nwalkers = 32
    p0 = np.random.rand(nwalkers, ndim)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "tutorial.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=[means, cov], backend=backend, pool=Pool()
    )

    max_n = 100

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(p0, iterations=max_n, progress=True):

        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
        
    
    samples = sampler.get_chain(flat=True)

    fig = corner.corner(
        samples
    );

    plt.savefig('parallel_corner.png')