import os
import numpy as np
import subprocess
import sys
import json
from types import SimpleNamespace
from typing import Callable, Optional, Any

from sympy import MatrixSymbol
from sympy import MutableDenseNDimArray as ZArray

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.logger_config import logger


def get_main_directory():
    """
    Determine the main project directory.
    """
    # 1. Environment variable overrides everything
    env_dir = os.getenv("ZOOMY_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    # 2. Detect Git root if available
    try:
        git_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return git_root
    except Exception:
        pass

    # 3. Handle Jupyter notebooks vs scripts
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip and hasattr(ip, "run_line_magic"):
            notebook_dir = ip.run_line_magic("pwd", "")
            if notebook_dir:
                return notebook_dir
    except Exception:
        pass

    # 4. Fallback: use the directory of the current script
    if hasattr(sys.modules["__main__"], "__file__"):
        return os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__))

    # 5. Ultimate fallback: CWD
    return os.getcwd()


class Zstruct(SimpleNamespace):
    """
    Dynamic structure that behaves like a Namespace but supports
    iteration and index access (based on insertion order).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        # Fallback to integer indexing
        return self.values()[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            setattr(self, key, value)
        elif isinstance(key, int):
            # Access by index (relying on __dict__ order)
            keys = list(self.__dict__.keys())
            if 0 <= key < len(keys):
                setattr(self, keys[key], value)
            else:
                raise IndexError(
                    f"Index {key} out of range for Zstruct of length {len(keys)}"
                )
        else:
            raise TypeError("Zstruct indices must be integers or strings")

    def length(self):
        return len(self.__dict__)

    def get_list(self, recursive: bool = True):
        if recursive:
            output = []
            for item in self.values():
                if hasattr(item, "get_list"):
                    output.append(item.get_list(recursive=True))
                else:
                    output.append(item)
            return output
        else:
            return self.values()

    def as_dict(self, recursive: bool = True):
        if recursive:
            output = {}
            for key, value in self.__dict__.items():
                if hasattr(value, "as_dict"):
                    output[key] = value.as_dict(recursive=True)
                else:
                    output[key] = value
            return output
        else:
            return self.__dict__

    def items(self, recursive: bool = False):
        return self.as_dict(recursive=recursive).items()

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def contains(self, key):
        return hasattr(self, key)

    def update(self, zstruct, recursive: bool = True):
        """
        Update the current Zstruct with another Zstruct or dictionary.
        """
        if not hasattr(zstruct, "as_dict") and not isinstance(zstruct, dict):
            raise TypeError("Input must be a Zstruct or a dictionary.")

        # Normalize input to dict
        data = (
            zstruct.as_dict(recursive=False) if hasattr(zstruct, "as_dict") else zstruct
        )

        if recursive:
            for key, value in data.items():
                if hasattr(self, key):
                    current_value = getattr(self, key)
                    if isinstance(current_value, Zstruct) and isinstance(
                        value, Zstruct
                    ):
                        current_value.update(value, recursive=True)
                    else:
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        else:
            for key, value in data.items():
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise TypeError("Input must be a dictionary.")

        data = d.copy()
        for k, v in data.items():
            if isinstance(v, dict):
                data[k] = Zstruct.from_dict(v)
        return cls(**data)


class Settings(Zstruct):
    """
    Settings class for the application.
    """

    def __init__(self, **kwargs):
        # Default initialization logic
        if "output" not in kwargs:
            logger.warning("No 'output' Zstruct found in Settings. Defaulting.")
            kwargs["output"] = Zstruct(
                directory="output",
                filename="simulation",
                snapshots=2,
                clean_directory=True,
            )
        elif not isinstance(kwargs["output"], Zstruct):
            # Try to convert dict to Zstruct if passed as dict
            if isinstance(kwargs["output"], dict):
                kwargs["output"] = Zstruct(**kwargs["output"])

        output = kwargs["output"]

        # Enforce defaults on output struct
        defaults = {
            "directory": "output",
            "filename": "simulation",
            "clean_directory": False,
            "snapshots": 2,
        }

        for key, val in defaults.items():
            if not output.contains(key):
                logger.warning(
                    f"No '{key}' attribute found in output Zstruct. Default: '{val}'"
                )
                setattr(output, key, val)

        super().__init__(**kwargs)

    @classmethod
    def default(cls):
        return cls(
            output=Zstruct(
                directory="output",
                filename="simulation",
                snapshots=2,
                clean_directory=False,
            )
        )

    @classmethod
    def from_json(cls, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)

        if "output" in data:
            data["output"] = Zstruct(**data["output"])

        return cls(**data)

    def write_json(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(
                get_main_directory(), self.output.directory, "settings.json"
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        settings_dict = self.as_dict()

        with open(filepath, "w") as f:
            json.dump(settings_dict, f, indent=4)
        print(f"Settings exported to {filepath}")

