import os
import sys
import json
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Type, Union, TYPE_CHECKING

from idmtools.assets import AssetCollection, Asset
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.builders import SimulationBuilder
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from idmtools.entities.command_task import CommandTask
from idmtools_platform_comps.utils.scheduling import add_schedule_config

from idmtools.core.platform_factory import Platform
from idmtools.entities import IAnalyzer
from idmtools.analysis.platform_anaylsis import PlatformAnalysis
from idmtools.analysis.download_analyzer import DownloadAnalyzer
from idmtools.analysis.analyze_manager import AnalyzeManager
from idmtools.core import ItemType

from typing import Dict, Any, Union
from idmtools.entities.iworkflow_item import IWorkflowItem
from idmtools.entities.simulation import Simulation
from argparse import ArgumentParser
import os
import json
from collections import OrderedDict

import subprocess

# change workding directory to the parent of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use for parallel execution on COMPS
class COMPSPool():
    @staticmethod
    def launch(p):
        pass
    @staticmethod
    def map(func, p):
        results = launch_COMPS(p) # p is a nwalker x ndim array
        # results is a list of nwalker elements
        return iter(results)
    
@dataclass
class ParamCommandTask(CommandTask):
    configfile_argument: Optional[str] = field(default="--config")

    def __init__(self, command):
        self.config = dict()
        CommandTask.__init__(self, command)

    def set_parameter(self, param_name, value):
        self.config[param_name] = value

    def gather_transient_assets(self) -> AssetCollection:
        """
        Gathers transient assets, primarily the settings.py file.

        Returns:
            AssetCollection: Transient assets.
        """
        # create a json string out of the dict self.config

        self.transient_assets.add_or_replace_asset(Asset(filename="iteration.json", content=json.dumps(self.config)))
        return CommandTask.gather_transient_assets(self)    


def update_parameter_callback(
    simulation,
    iteration
):
    """This function updates the parmeter values for each individual simulation."""
    simulation.task.set_parameter("iteration", iteration)

    ret_tags_dict = {
        "iteration": iteration
    }
    return ret_tags_dict


def launch_COMPS(p):

    nwalkers, ndim = p.shape
    
    #write out a json file where key is the row index of p and the value is the row
    with open("parameters.json", "w") as f:
        json.dump({str(i): p[i].tolist() for i in range(nwalkers)}, f)

    # Create a platform to run the workitem
    platform = Platform("CALCULON", priority="Normal")

    # create commandline input for the task
    cmdline = "singularity exec ./Assets/emcee_example_0.0.1_b5ddf11.sif bash run.sh"

    command = CommandLine(cmdline)
    task = ParamCommandTask(command=command)

    # Add our image
    task.common_assets.add_assets(AssetCollection.from_id_file("sif.id"))
    task.common_assets.add_directory("Assets")

    # Add analysis scripts
    task.transient_assets.add_or_replace_asset(Asset(filename="parameters.json"))
    task.transient_assets.add_or_replace_asset(Asset(filename="run.sh"))
    task.transient_assets.add_or_replace_asset(Asset(filename="sample.py"))

    ts = TemplatedSimulations(base_task=task)

    sb = SimulationBuilder()
    sb.add_multiple_parameter_sweep_definition(
        update_parameter_callback,
        iteration=np.arange(nwalkers).tolist(),
    )
    ts.add_builder(sb)
    num_threads = 1
    add_schedule_config(
        ts,
        command=cmdline,
        NumNodes=1,
        num_cores=num_threads,
        node_group_name="idm_abcd",
        Environment={"OMP_NUM_THREADS": str(num_threads)},
    )
    experiment = Experiment.from_template(ts, name=os.path.split(sys.argv[0])[1])
    experiment.run(wait_until_done=True, scheduling=True)

    if experiment.succeeded:
        experiment.to_id_file("experiment.id")

    # run download_analysis.py
    # Run a simple shell command
    result = subprocess.run(['python', 'download_analysis.py'], 
                            capture_output=True, text=True)

    with open("analyzer.id", "r") as file:
        analyzer_id = file.read()
    with open(os.path.join("output_emcee", analyzer_id, "data_brick.json"), "r") as file:
        data_dict = json.load(file)

    ind = np.array([int(k['iteration']) for k in data_dict.values()])
    ind = np.sort(ind)
    print(ind)
    result = np.array([data_dict[k]['result'] for k in data_dict.keys()])
    return result.tolist()

if __name__ == "__main__":
    
   # number of dimensions to the problem
    ndim = 5

    np.random.seed(42)
    means = np.random.rand(ndim)

    cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)

    # write means as .np file
    np.save("Assets/means.npy", means)
    # write cov as .np file
    np.save("Assets/cov.npy", cov)

    nwalkers = 256
    p0 = np.random.rand(nwalkers, ndim)

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "tutorial.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, print, args=[means, cov], backend=backend, pool=COMPSPool(),
    )

    max_n = 10

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # state = sampler.run_mcmc(p0, 100)
    # sampler.reset()

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

    fig = corner.corner(samples)

    plt.savefig('monitor_corner.png')
