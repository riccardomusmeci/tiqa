import os
from pathlib import Path
from typing import Dict, Optional, Union

import yaml


class TiqaConfiguration:
    """Configuration class for tiqa."""

    def _merge(self, config_dir: Union[str, Path], main_config: Dict) -> Dict:
        """Merge multiple yml files.

        Args:
            config_dir (Union[str, Path]): path to directory containing yml files
            main_config (Dict): main configuration dict

        Raises:
            Exception: if there is an error loading a config

        Returns:
            Dict: configuration dict
        """
        for k in main_config:
            if main_config[k] is not None:
                try:
                    main_config[k] = self._load_config(
                        config_path=os.path.join(config_dir, k, f"{main_config[k]}.yaml")
                    )
                except Exception as e:
                    raise e
                    # raise (f"Error loading config {main_config[k]}: {e}")
        return main_config

    @classmethod
    def load_experiment(cls, config_dir: Union[str, Path], experiment: str) -> Dict:
        """Load an experiment configuration.

        Args:
            config_dir (Union[str, Path]): where the config folders are located
            experiment_name (str): main yaml file name (no yaml at the end is needed)

        Returns:
            Dict: experiment configuration
        """

        main_config = cls()._load_config(config_path=os.path.join(config_dir, f"{experiment}.yaml"))
        config = cls()._merge(config_dir=config_dir, main_config=main_config)
        return config

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load a single yml file.

        Args:
            config_path (Union[str, Path]): path to yml file

        Raises:
            FileExistsError: if config file does not exist

        Returns:
            Dict: yml dict
        """
        if os.path.exists(path=config_path) is False:
            raise FileExistsError(f"Config file {config_path} does not exist.")

        with open(config_path) as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                quit()
        return params

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> Dict:
        """Load a single yml file.

        Args:
            config_path (Union[str, Path]): path to yml file

        Returns:
            Dict: yml dict
        """
        config = cls()._load_config(config_path=config_path)
        return config

    @classmethod
    def save(cls, config: Dict, output_path: Union[str, Path]) -> None:
        """Save a single yml file.

        Args:
            config (Dict): configuration dict
            output_path (Union[str, Path]): path to yml file
        """
        with open(output_path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
