import logging
import typing as t
from pathlib import Path
from types import MappingProxyType

import yaml


class Config:
    def __init__(self, config_path: str | Path | None = None, config: dict | None = None):
        if config_path is not None:
            self.config = yaml.safe_load(open(config_path))
        elif config is not None:
            self.config = config
        else:
            logging.warning("No config file or config dictionary provided. Using default config.")
            self.parent_path = Path().parent.absolute()
            self.config = yaml.safe_load(open(f"{self.parent_path}/config.yaml"))

    def _load_config(self, config_path: str | Path) -> dict:
        try:
            return yaml.safe_load(open(config_path))
        except FileNotFoundError as e:
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg) from e

    def __getitem__(self, key: str) -> dict:
        return self.config[key]

    def __setitem__(self, key: str, value: dict) -> None:
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.config

    def __repr__(self) -> str:
        return f"Config({self.config})"


class Configurable:
    """Base class for configurable objects"""

    _config_key: str
    _default_config: MappingProxyType

    def __init__(self, conf: Config) -> None:
        super().__init__()
        self.config = conf
        self._set_config()

    def _set_config(self) -> None:
        """Set the config for the object"""
        if self._config_key not in self.config:
            self.config[self._config_key] = dict(self._default_config)
        else:
            for key, value in self._default_config.items():
                if key not in self.config[self._config_key]:
                    self.config[self._config_key][key] = value

    def _merge_default_kwargs(self, defaults: dict, config_key: str = "kwargs"):
        """Merge kwargs from config with default kwargs. This doesn't handle nested kwargs"""
        if config_key not in self.config[self._config_key]:
            self.config[self._config_key][config_key] = dict(defaults)
        else:
            for key, value in defaults.items():
                if key not in self.config[self._config_key][config_key]:
                    self.config[self._config_key][config_key][key] = value

    @t.overload
    def add_config(self, config: dict) -> None: ...

    @t.overload
    def add_config(self, config: str, value: str | int | float | bool | dict) -> None: ...

    def add_config(self, config: dict | str, value: str | int | float | bool | dict | None = None) -> None:
        """Add a config to the object"""
        if isinstance(config, dict):
            self.config[self._config_key].update(config)
        elif isinstance(config, str):
            if not isinstance(value, str | int | float | bool | dict):
                msg = f"value must be either a string, int, float, bool or dict, not {type(value)}"
                raise TypeError(msg)
            self.config[self._config_key][config] = value
        else:
            msg = f"config must be either a dict or a string, not {type(config)}"
            raise TypeError(msg)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config[self._config_key]})"
