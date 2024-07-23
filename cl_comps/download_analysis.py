"""
https://github.com/krosenfeld-IDM/mcv_sia_timing_v_coverage/blob/main/workflow_sia_delay/analyzers/analyzers.py
"""

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


class ThisAnalyzer(IAnalyzer):
    def __init__(self):
        super().__init__(filenames=["result_waves.json", "result_lwps.json"])

    def map(
        self, data: Dict[str, Any], item: Union["IWorkflowItem", "Simulation"]
    ) -> Any:
        data_dict = {}
        data_dict.update(data[self.filenames[0]])
        data_dict.update(data[self.filenames[1]])
        # # check results
        # for k,v in data_dict.items():
        #     for ie, e in enumerate(v):
        #         if not isinstance(e, (int, float)):
        #             data_dict[k][ie] = -999
        data_dict["tags"] = item.tags

        return data_dict

    def reduce(self, all_data: Dict[Union["IWorkflowItem", "Simulation"], Any]) -> Any:
        output_dict = OrderedDict()

        for ix, (s, v) in enumerate(all_data.items()):
            entry = {}
            tags = v.pop("tags")
            entry.update(tags)
            entry.update(v)
            output_dict[str(s.uid)] = entry

        # write data
        with open("data_brick.json", "w") as fp:
            json.dump(output_dict, fp)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_analyzer", default=1, type=int)
    parser.add_argument("--run_download", default=1, type=int)
    parser.add_argument("--run_figures", default=0, type=int)
    args = parser.parse_args()

    analyzers_args = {}
    ana_name = os.path.splitext(os.path.basename(__file__))[0]
    exp_name = os.path.split(os.path.dirname(__file__))[-1]
    output_dir = "output_" + ana_name
    print(ana_name)

    if args.run_analyzer:
        platform = Platform("SLURM")

        experiment_ids = []
        with open("experiment.id", "r") as fid:
            for line in fid.readlines():
                experiment_ids += [line.split(":")[0]]

        # Setup analyzers
        analysis = PlatformAnalysis(
            platform=platform,
            experiment_ids=experiment_ids,
            analyzers=[ThisAnalyzer],
            analyzers_args=[analyzers_args],
            analysis_name="SSMT download laser analysis",
        )
        analysis.analyze(check_status=True)

        # Save work item id to file
        wi = analysis.get_work_item()
        with open(f"{ana_name}_ID", "w") as fd:
            fd.write("{}".format(wi.uid))
        print(wi)

    if args.run_download:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with Platform("SLURM") as platform:
            with open(f"{ana_name}_ID", "r") as fid:
                workitem_id = fid.readlines()[0]
            analyzers = [
                DownloadAnalyzer(filenames=["data_brick.json"], output_path=output_dir)
            ]
            manager = AnalyzeManager(
                ids=[(workitem_id, ItemType.WORKFLOW_ITEM)], analyzers=analyzers
            )
            manager.analyze()