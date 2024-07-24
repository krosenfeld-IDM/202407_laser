import os
import sys
import numpy as np
from functools import partial
from idmtools.assets import AssetCollection, Asset
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.builders import SimulationBuilder
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from script_task import PyConfiguredSingularityTask as PCST
from idmtools_platform_comps.utils.scheduling import add_schedule_config


def update_parameter_callback(simulation, **kwargs):
    for k,v in kwargs.items():
        simulation.task.set_parameter(k, v)
    return kwargs
    # simulation.task.set_parameter("iteration", iteration)
    # return {"iteration": iteration}

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    os.chdir(here)

    # Create a platform to run the workitem
    platform = Platform("CALCULON", priority="Normal")

    # create commandline input for the task
    cmdline = "singularity exec ./Assets/krosenfeld_cl_idm_laser_0.0.1_07e7a44.sif bash run.sh"

    command = CommandLine(cmdline)
    task = PCST(command=command)

    # Add our image
    task.common_assets.add_assets(AssetCollection.from_id_file("laser.id"))
    task.common_assets.add_directory("Assets")

    # Add simulation scripts
    task.transient_assets.add_or_replace_asset(Asset(filename="run_simulation.py"))

    # Add analysis scripts
    task.transient_assets.add_or_replace_asset(Asset(filename="analyze_waves.py"))
    task.transient_assets.add_or_replace_asset(Asset(filename="analyze_lwps.py"))

    ts = TemplatedSimulations(base_task=task)

    sb = SimulationBuilder()
    sb.add_multiple_parameter_sweep_definition(
        lambda simulation, iteration: update_parameter_callback(simulation, iteration=iteration),
        iteration=np.arange(2).tolist()
    )

    ts.add_builder(sb)
    num_threads = 8
    add_schedule_config(
        ts,
        command=cmdline,
        NumNodes=1,
        num_cores=num_threads,
        node_group_name="idm_48cores",
        Environment={"NUMBA_NUM_THREADS": str(num_threads),
                     "PYTHONPATH": "."},
    )
    experiment = Experiment.from_template(ts, name=os.path.split(sys.argv[0])[1])
    experiment.run(wait_until_done=True, scheduling=True)
    if experiment.succeeded:
        experiment.to_id_file("experiment.id")
