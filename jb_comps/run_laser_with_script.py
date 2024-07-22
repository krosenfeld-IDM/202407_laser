import os
import sys
import numpy as np
from idmtools.assets import AssetCollection, Asset
from idmtools.core.platform_factory import Platform
from idmtools.entities import CommandLine
from idmtools.builders import SimulationBuilder
from idmtools.entities.experiment import Experiment
from idmtools.entities.templated_simulation import TemplatedSimulations
from script_task import PyConfiguredSingularityTask as PCST
from idmtools_platform_comps.utils.scheduling import add_schedule_config


def update_parameter_callback(
    simulation,
    base_infectivity,
    seasonal_multiplier,
    migration_fraction,
    iteration
):
    simulation.task.set_parameter("base_infectivity", base_infectivity)
    simulation.task.set_parameter("seasonal_multiplier", seasonal_multiplier)
    simulation.task.set_parameter("migration_fraction", migration_fraction)

    ret_tags_dict = {
        "base_infectivity": base_infectivity,
        "seasonal_multiplier": seasonal_multiplier,
        "migration_fraction": migration_fraction,
        "iteration": iteration
    }
    return ret_tags_dict


if __name__ == "__main__":
    here = os.path.dirname(__file__)

    # Create a platform to run the workitem
    platform = Platform("CALCULON", priority="Normal", node_group = 'idm_abcd')

    # create commandline input for the task
    cmdline = "singularity exec ./Assets/krosenfeld_idm_laser_0.1.0_c4f4c89.sif bash run.sh"

    command = CommandLine(cmdline)
    task = PCST(command=command)

    # Add our image
    task.common_assets.add_assets(AssetCollection.from_id_file("laser.id"))
    task.common_assets.add_directory("Assets")

    # Add analysis scripts
    task.transient_assets.add_or_replace_asset(Asset(filename="analyze_waves.py"))
    task.transient_assets.add_or_replace_asset(Asset(filename="analyze_lwps.py"))

    ts = TemplatedSimulations(base_task=task)

    sb = SimulationBuilder()
    sb.add_multiple_parameter_sweep_definition(
        update_parameter_callback,
        base_infectivity=[3.9, 4.0, 4.1],
        seasonal_multiplier=[0.55, 0.60, 0.65],
        migration_fraction=[0.004, 0.01, 0.04, 0.1],
        iteration=np.arange(20).tolist(),
    )
    # sb.add_multiple_parameter_sweep_definition(
    #     update_parameter_callback,
    #     base_infectivity=[4.0],
    #     seasonal_multiplier=[0.6],
    #     migration_fraction=[0.04],
    #     iteration=np.arange(2).tolist(),
    # )
    ts.add_builder(sb)
    num_threads = 24
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
