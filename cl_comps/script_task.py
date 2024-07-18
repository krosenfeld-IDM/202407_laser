"""

"""
import os
from typing import Optional, Type, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from idmtools.entities.command_task import CommandTask
from idmtools.assets import AssetCollection, Asset
from idmtools.entities.iworkflow_item import IWorkflowItem
from idmtools.entities.simulation import Simulation

if TYPE_CHECKING:  # pragma: no cover
    from idmtools.entities.iplatform import IPlatform


@dataclass
class PyConfiguredSingularityTask(CommandTask):

    configfile_argument: Optional[str] = field(default="--config")

    def __init__(self, command, script_name="run.sh"):
        self.script_name = script_name
        self.script = self._load_script(script_name)
        self.config = dict()
        # self.base_settings = self._load_settings("settings.py")        
        CommandTask.__init__(self, command)


    @staticmethod
    def _load_script(script_name):
        # read in the text file in self.script_name as a string
        with open(script_name, 'r') as file:
            return file.read()

    def _load_settings(self, file_path):
        """
        Loads settings from a given file path.

        Args:
            file_path (str): Path to the settings file.

        Returns:
            list: List of settings lines.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        with open(file_path, 'r') as file:
            settings = file.readlines()
        return settings

    def _process_settings(self):
        """
        Processes the settings by replacing values based on the current configuration.

        Returns:
            str: Processed settings as a string.
        """
        processed_settings = []

        for line in self.base_settings:
            if '=' in line:
                key, value = map(str.strip, line.split('=', 1))
                if key in self.config:
                    value = str(self.config[key])
                    processed_settings.append(f"{key} = {value} # SET IN SCRIPT")
                else:
                    processed_settings.append(f"{key} = {value}")
            else:
                processed_settings.append(line)  # Handle lines without '='

        return "\n".join(processed_settings)


    def set_parameter(self, name: str, value: any) -> dict:
        """
        Sets a parameter in the configuration.

        Args:
            name (str): Parameter name.
            value (any): Parameter value.

        Returns:
            dict: Updated configuration.
        """
        self.config[name] = value
        return {name: value}        

    def gather_transient_assets(self) -> AssetCollection:
        """
        Gathers transient assets, primarily the settings.py file.

        Returns:
            AssetCollection: Transient assets.
        """
        # self.transient_assets.add_or_replace_asset(Asset(filename="settings.py", content=self._process_settings()))
        self.transient_assets.add_or_replace_asset(Asset(filename=self.script_name))
        return CommandTask.gather_transient_assets(self)    
   