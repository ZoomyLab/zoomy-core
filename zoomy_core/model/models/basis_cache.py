"""
Hash-based cache for basis function matrices.

Replaces the old flat-file cache in Basismatrices with a manifest-based system:
- Each (basis_type, level, weight) combination gets a unique hash
- All matrices stored in a single .npz file per hash
- Central manifest.json tracks all entries with metadata
- Stale detection via basis definition fingerprint
"""

import hashlib
import json
import os
import time

import numpy as np
import sympy
from sympy.abc import z

from zoomy_core.misc.misc import get_main_directory


CACHE_ROOT = ".cache/basismatrices"
MANIFEST_NAME = "manifest.json"
MATRIX_NAMES = ["phib", "M", "A", "B", "D", "Dxi", "Dxi2", "DD", "D1", "DT"]


def _basis_fingerprint(basis):
    """
    Compute a fingerprint string that uniquely identifies a basis configuration.
    Changes if the basis definition, weight, or bounds change.
    """
    parts = [
        basis.name,
        str(basis.level),
        str(basis.bounds()),
        str(basis.weight(z)),
    ]
    for k in range(basis.level + 1):
        parts.append(str(basis.get(k)))
    return "|".join(parts)


def _compute_hash(fingerprint):
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


class BasisMatrixCache:
    """
    Hash-based cache for basis matrices.

    Usage:
        cache = BasisMatrixCache()
        matrices = cache.get_or_compute(basis, level)
        # matrices is a dict: {"M": np.array, "A": np.array, ...}
    """

    def __init__(self, cache_root=CACHE_ROOT):
        self.cache_root = cache_root
        self._abs_root = os.path.join(get_main_directory(), cache_root)

    def _manifest_path(self):
        return os.path.join(self._abs_root, MANIFEST_NAME)

    def _load_manifest(self):
        path = self._manifest_path()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self, manifest):
        os.makedirs(self._abs_root, exist_ok=True)
        path = self._manifest_path()
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _entry_dir(self, hash_key):
        return os.path.join(self._abs_root, hash_key)

    def _npz_path(self, hash_key):
        return os.path.join(self._entry_dir(hash_key), "matrices.npz")

    def get_or_compute(self, basis, level, compute_fn):
        """
        Load cached matrices or compute and cache them.

        Parameters
        ----------
        basis : Basisfunction
            The basis function object.
        level : int
            Polynomial level.
        compute_fn : callable
            Function that computes matrices. Called as compute_fn(level)
            and should set self.M, self.A, etc. on the Basismatrices object.
            Returns a dict of {name: np.ndarray}.

        Returns
        -------
        dict : {name: np.ndarray} for each matrix
        """
        fingerprint = _basis_fingerprint(basis)
        hash_key = _compute_hash(fingerprint)

        cached = self._try_load(hash_key, fingerprint)
        if cached is not None:
            return cached

        matrices = compute_fn(level)
        self._save(hash_key, fingerprint, basis, level, matrices)
        return matrices

    def _try_load(self, hash_key, fingerprint):
        manifest = self._load_manifest()
        entry = manifest.get(hash_key)
        if entry is None:
            return None
        if entry.get("fingerprint") != fingerprint:
            return None
        npz_path = self._npz_path(hash_key)
        if not os.path.exists(npz_path):
            return None
        try:
            data = np.load(npz_path, allow_pickle=True)
            return {name: data[name] for name in MATRIX_NAMES if name in data}
        except Exception:
            return None

    def _save(self, hash_key, fingerprint, basis, level, matrices):
        entry_dir = self._entry_dir(hash_key)
        os.makedirs(entry_dir, exist_ok=True)

        npz_path = self._npz_path(hash_key)
        np.savez(npz_path, **matrices)

        manifest = self._load_manifest()
        manifest[hash_key] = {
            "basis_name": basis.name,
            "level": level,
            "fingerprint": fingerprint,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "matrices": list(matrices.keys()),
        }
        self._save_manifest(manifest)

    def list_entries(self):
        return self._load_manifest()

    def clear(self):
        import shutil
        if os.path.exists(self._abs_root):
            shutil.rmtree(self._abs_root)
