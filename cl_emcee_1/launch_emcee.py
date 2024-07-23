import os
import json
import emcee
import corner
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

from idmtools.assets import AssetCollection, Asset
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.builders import SimulationBuilder
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from idmtools.entities.command_task import CommandTask
from idmtools_platform_comps.utils.scheduling import add_schedule_config


# use for parallel execution on COMPS
@dataclass
class COMPSPool:
    step: int = -1

    def check_limits(self, params):
        """Add boundaries"""
        blim = [0, 0]
        ulim = [0, np.inf]
        return np.all(np.greater(params, blim)) and np.all(np.less(params, ulim))

    def map(self, func, p):
        results = self.launch(p)  # p is a nwalker x ndim array
        # results is a list of nwalker elements
        for i, params in enumerate(p):
            if not self.check_limits(params):
                results[i] = -np.inf
        return iter(results)

    def launch(self, p):
        self.step += 1
        return launch_COMPS(p, step=self.step)


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

        self.transient_assets.add_or_replace_asset(
            Asset(filename="walker.json", content=json.dumps(self.config))
        )
        self.transient_assets.add_or_replace_asset(
            Asset(filename="settings.py", content=self.process_settings())
        )
        return CommandTask.gather_transient_assets(self)

    def process_settings(self, file_path="settings.py"):
        """
        Processes the settings by replacing values based on the current configuration.

        Returns:
            str: Processed settings as a string.
        """

        # walker ID
        walker = self.config["walker"]

        # load the base settngs.py file
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        with open(file_path, "r") as file:
            base_settings = file.readlines()

        with open("parameters.json", "r") as file:
            parameters_dict = json.load(file)
        parameters = parameters_dict[str(walker)]

        processed_settings = []
        param_config = ["migration_fraction", "base_infectivity"]
        for line in base_settings:
            if "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                if key in param_config:
                    value = str(parameters[param_config.index(key)])
                    processed_settings.append(f"{key} = {value} # SET IN SCRIPT")
                else:
                    processed_settings.append(f"{key} = {value}")
            else:
                processed_settings.append(line)  # Handle lines without '='

        return "\n".join(processed_settings)


def update_parameter_callback(simulation, walker):
    """This function updates the parmeter values for each individual simulation."""
    simulation.task.set_parameter("walker", walker)

    ret_tags_dict = {"walker": walker}
    return ret_tags_dict


def launch_COMPS(p, step=0):
    nwalkers, _ = p.shape  # nwalkers x ndim

    # write out a json file where key is the row index of p and the value is the row
    with open("parameters.json", "w") as f:
        json.dump({str(i): p[i].tolist() for i in range(nwalkers)}, f)

    # Create a platform to run the workitem
    with Platform("CALCULON", priority="Normal") as platform:
        # create commandline input for the task
        cmdline = (
            "singularity exec ./Assets/krosenfeld_idm_laser_0.1.0_c4f4c89.sif bash run.sh"
        )

        command = CommandLine(cmdline)
        task = ParamCommandTask(command=command)

        # Add our image
        task.common_assets.add_assets(AssetCollection.from_id_file("sif.id"))
        task.common_assets.add_directory("Assets")

        # Add analysis scripts
        task.transient_assets.add_or_replace_asset(Asset(filename="settings.py"))
        task.transient_assets.add_or_replace_asset(Asset(filename="run.sh"))
        task.transient_assets.add_or_replace_asset(Asset(filename="analyze_waves.py"))
        task.transient_assets.add_or_replace_asset(Asset(filename="analyze_lwps.py"))
        task.transient_assets.add_or_replace_asset(Asset(filename="logprob.py"))

        ts = TemplatedSimulations(base_task=task)

        sb = SimulationBuilder()
        sb.add_multiple_parameter_sweep_definition(
            update_parameter_callback,
            walker=np.arange(nwalkers).tolist(),
        )
        ts.add_builder(sb)
        num_threads = 10
        add_schedule_config(
            ts,
            command=cmdline,
            NumNodes=1,
            num_cores=num_threads,
            node_group_name="idm_abcd",
            Environment={"OMP_NUM_THREADS": str(num_threads)},
        )
        experiment = Experiment.from_template(ts, name=f"emcee_jb_1_{step}")
        experiment.run(wait_until_done=True, scheduling=True)

    if experiment.succeeded:
        experiment.to_id_file("experiment.id")

    # run download_analysis.py
    # Run a simple shell command
    result = subprocess.run(
        ["python", "download_analysis.py"], capture_output=True, text=True
    )

    with open("analyzer.id", "r") as file:
        analyzer_id = file.read()
    with open(
        os.path.join("output_emcee", analyzer_id, "data_brick.json"), "r"
    ) as file:
        data_dict = json.load(file)

    ind = np.array([int(k["walker"]) for k in data_dict.values()])
    ind = np.sort(ind)
    print(ind)
    result = np.array([data_dict[k]["result"] for k in data_dict.keys()])
    return result.tolist()


if __name__ == "__main__":
    # change workding directory to the parent of this file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # number of dimensions to the problem
    ndim = 2 # migration_fraction, base_infectivity

    nwalkers = 64
    p0 = np.random.rand(nwalkers, ndim) + np.array([0, 1.5])[np.newaxis, :]

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = "emcee.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        print,
        backend=backend,
        pool=COMPSPool(),
    )

    max_n = 6

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
        if sampler.iteration % 1:
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

    plt.savefig("monitor_corner.png")
