import os
import numpy as np
import subprocess
import sys
import json
from types import SimpleNamespace
from typing import Callable, Optional, Any

import sympy as sp
from sympy import MutableDenseNDimArray, tensorcontraction, tensorproduct

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.logger_config import logger


class ZArray(MutableDenseNDimArray):
    """
    Enhanced SymPy Array that supports matrix multiplication (@)
    and vector algebra (+, -) with lists and Zstructs.
    """

    def __new__(cls, iterable, *args, **kwargs):
        # 1. Normalize Input to Nested List
        if isinstance(iterable, (ZArray, sp.NDimArray, sp.MatrixBase)):
            data = iterable.tolist()
        elif hasattr(iterable, "values") and callable(iterable.values):  # Zstruct
            data = list(iterable.values())
        elif isinstance(iterable, (list, tuple)):
            data = iterable
        else:
            raise TypeError(
                f"ZArray constructor does not support type: {type(iterable)}"
            )

        # 2. Helper to flatten and determine shape
        # We do this manually because SymPy's inference can be fragile with mixed types
        flat_list = []

        def flatten_and_get_shape(item, current_depth=0, shape_acc=None):
            if shape_acc is None:
                shape_acc = []

            if isinstance(item, (list, tuple)):
                if current_depth >= len(shape_acc):
                    shape_acc.append(len(item))
                elif shape_acc[current_depth] != len(item):
                    # Ragged array support is limited in SymPy, warn or assume user handles it
                    pass

                for sub in item:
                    flatten_and_get_shape(sub, current_depth + 1, shape_acc)
            else:
                flat_list.append(item)
            return shape_acc

        inferred_shape = tuple(flatten_and_get_shape(data))

        # 3. Determine Final Shape (Args > Kwargs > Inferred)
        final_shape = inferred_shape
        if args:
            final_shape = args[0]
        elif "shape" in kwargs:
            final_shape = kwargs["shape"]

        # 4. Construct (Flat Data + Explicit Shape is safest)
        return super().__new__(cls, flat_list, final_shape, **kwargs)

    def _to_array(self, other):
        """Helper to cast Zstruct, list, or scalar to a SymPy-compatible Array."""
        if hasattr(other, "values"):  # Specifically catches Zstruct
            return sp.Array(list(other.values()))
        if isinstance(other, (list, tuple)):
            return sp.Array(other)
        if hasattr(other, "tolist"):  # Catch numpy arrays or other ZArrays
            return sp.Array(other.tolist())
        return other

    def __matmul__(self, other):
        """Matrix Multiplication (@ operator)."""
        other_arr = self._to_array(other)
        if isinstance(other_arr, sp.NDimArray):
            # Matrix-Vector Projection (Rank 2 @ Rank 1) -> Vector
            if self.rank() == 2 and other_arr.rank() == 1:
                return ZArray(tensorcontraction(tensorproduct(self, other_arr), (1, 2)))
            # Dot Product (Rank 1 @ Rank 1) -> Scalar
            if self.rank() == 1 and other_arr.rank() == 1:
                return tensorcontraction(tensorproduct(self, other_arr), (0, 1))
            # Matrix-Matrix Multiplication (Rank 2 @ Rank 2) -> Matrix
            if self.rank() == 2 and other_arr.rank() == 2:
                return ZArray(tensorcontraction(tensorproduct(self, other_arr), (1, 2)))
        return NotImplemented

    def __sub__(self, other):
        return ZArray(super().__add__(-1 * self._to_array(other)))

    def __add__(self, other):
        return ZArray(super().__add__(self._to_array(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return ZArray(sp.Array(self._to_array(other)) + (-1 * self))

    def tomatrix(self):
        """Converts rank-2 array to SymPy Matrix (needed for .T or .eigenvals)."""
        if self.rank() != 2:
            raise ValueError("Only rank-2 ZArrays can be converted to Matrix.")
        return sp.Matrix(self.tolist())

    @property
    def flat(self):
        return list(self)


def get_main_directory():
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
        self._symbolic_name = None

    def __repr__(self):
        clean_dict = self._filter_dict()
        args = ", ".join(f"{k}={v!r}" for k, v in clean_dict.items())
        return f"Zstruct({args})"

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return self.values()[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            setattr(self, key, value)
        elif isinstance(key, int):
            keys = self.keys()
            if 0 <= key < len(keys):
                setattr(self, keys[key], value)
            else:
                raise IndexError(f"Index {key} out of range")
        else:
            raise TypeError("Indices must be integers or strings")

    def _filter_dict(self):
        """Returns internal dict excluding private/hidden attributes."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def length(self):
        return len(self._filter_dict())

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
        data = self._filter_dict()
        if recursive:
            output = {}
            for key, value in data.items():
                if hasattr(value, "as_dict"):
                    output[key] = value.as_dict(recursive=True)
                else:
                    output[key] = value
            return output
        else:
            return data

    def items(self, recursive: bool = False):
        return self.as_dict(recursive=recursive).items()

    def keys(self):
        return list(self._filter_dict().keys())

    def values(self):
        return list(self._filter_dict().values())

    def contains(self, key):
        return hasattr(self, key)

    def update(self, zstruct, recursive: bool = True):
        if not hasattr(zstruct, "as_dict") and not isinstance(zstruct, dict):
            raise TypeError("Input must be a Zstruct or a dictionary.")
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
    def __init__(self, **kwargs):
        if "output" not in kwargs:
            logger.warning("No 'output' Zstruct found in Settings. Defaulting.")
            kwargs["output"] = Zstruct(
                directory="output",
                filename="simulation",
                snapshots=2,
                clean_directory=True,
            )
        elif not isinstance(kwargs["output"], Zstruct):
            if isinstance(kwargs["output"], dict):
                kwargs["output"] = Zstruct(**kwargs["output"])

        output = kwargs["output"]
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
