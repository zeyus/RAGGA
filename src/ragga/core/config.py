import logging
import typing as t
from pathlib import Path
from types import MappingProxyType

import yaml
from pydantic.v1.utils import deep_update


def tuple_constructor(loader, node):
    # Load the sequence of values from the YAML node
    values = loader.construct_sequence(node)
    # Return a tuple constructed from the sequence
    return tuple(values)

# Register the constructor with PyYAML
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)

class Config:
    def __init__(self, config_path: str | Path | None = None, config: dict | None = None):
        if config_path is not None:
            self.config = yaml.load(open(config_path), Loader=yaml.SafeLoader)
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

    def merge(self, config: dict[str, t.Any]) -> None:
        self.config = deep_update(self.config, config)

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


    @property
    def key(self) -> str:
        return self._config_key

    def _set_config(self) -> None:
        """Set the config for the object"""
        if self._config_key not in self.config:
            self.config[self._config_key] = dict(self._default_config)
        else:
            for key, value in self._default_config.items():
                if key not in self.config[self._config_key]:
                    self.config[self._config_key][key] = value

    def _merge_default_kwargs(self, defaults: dict, config_key: str = "kwargs"):
        """Merge kwargs from config with default kwargs. This doesn't handle nested kwargs/dicts"""
        if config_key not in self.config[self._config_key]:
            self.config[self._config_key][config_key] = dict(defaults)
        else:
            for key, value in defaults.items():
                if key not in self.config[self._config_key][config_key]:
                    self.config[self._config_key][config_key][key] = value

    @t.overload
    def add_config(self, config: dict) -> None:
        """Set the config dictionary for the object"""
        ...

    @t.overload
    def add_config(self, config: str, value: str | int | float | bool | dict) -> None:
        """Set a config key to the specified value for the object"""
        ...

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
