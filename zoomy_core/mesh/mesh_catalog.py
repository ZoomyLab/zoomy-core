"""Unified interface for browsing, selecting, and downloading Zoomy meshes.

Replaces the scattered download helpers in browse_meshes.py, common.py,
and the demo scripts with a single ``MeshCatalog`` that mirrors the
CLI's *list / show / select* pattern on the Python level.

Quick-start::

    from zoomy_core.mesh.mesh_catalog import MeshCatalog

    catalog = MeshCatalog()          # fetches index once, caches it
    catalog.list()                   # grouped overview
    catalog.show("square__mesh")     # details for one mesh
    path = catalog.download("square__mesh", size="fine")  # -> local .h5
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Environment detection (CPython vs Pyodide/JupyterLite)
# ---------------------------------------------------------------------------
try:
    from pyodide.http import pyfetch  # type: ignore[import-untyped]

    _IN_PYODIDE = True
except ImportError:
    _IN_PYODIDE = False

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
BASE_URL = "https://zoomylab.github.io/meshes/"
INDEX_URL = BASE_URL + "index.json"
FILES_URL = BASE_URL + "meshes/"

ALLOWED_TYPES = frozenset({"msh", "geo", "h5"})
SIZE_LABELS = frozenset({"coarse", "medium", "fine"})
_SIZE_RE = re.compile(r"^(.+)__(" + "|".join(SIZE_LABELS) + r")$")

# Categories inferred from the directory-encoded mesh name (first segment).
CATEGORIES = {
    "basic_shapes": "Basic Shapes",
    "channels": "Channels",
    "curved": "Curved Geometries",
    "structural": "Structural",
    "volumes": "3-D Volumes",
    "complex": "Complex Assemblies",
}


def _http_get_json(url: str):
    """Fetch JSON synchronously (CPython only)."""
    import requests

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _http_get_bytes(url: str) -> bytes:
    """Fetch raw bytes synchronously (CPython only)."""
    import requests

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


async def _http_get_json_async(url: str):
    if _IN_PYODIDE:
        resp = await pyfetch(url)
        return await resp.json()
    import requests

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


async def _http_get_bytes_async(url: str) -> bytes:
    if _IN_PYODIDE:
        resp = await pyfetch(url)
        return await resp.bytes()
    import requests

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class MeshEntry:
    """One logical mesh (e.g. ``square__mesh``) with its available variants."""

    name: str
    sizes: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)

    # Derived from name
    @property
    def category(self) -> str:
        first = self.name.split("__")[0]
        return CATEGORIES.get(first, "")

    @property
    def short_name(self) -> str:
        """Strip the category prefix for display (e.g. 'square__mesh' -> 'square / mesh')."""
        parts = self.name.split("__")
        if parts[0] in CATEGORIES:
            parts = parts[1:]
        return " / ".join(parts)

    def filename(self, filetype: str = "h5", size: str = "medium") -> str:
        if size and self.sizes:
            return f"{self.name}__{size}.{filetype}"
        return f"{self.name}.{filetype}"

    def url(self, filetype: str = "h5", size: str = "medium") -> str:
        return FILES_URL + self.filename(filetype, size)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
class MeshCatalog:
    """Browse, inspect, and download meshes from the Zoomy mesh repository.

    Inspired by the CLI's ``overview / list / select / show`` pattern.
    """

    def __init__(self, cache_dir: Optional[str | Path] = None, auto_fetch: bool = True):
        self._entries: Dict[str, MeshEntry] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".zoomy" / "meshes"
        if auto_fetch:
            self.refresh()

    # -- index management ---------------------------------------------------

    def refresh(self) -> "MeshCatalog":
        """(Re)fetch the mesh index from GitHub Pages."""
        data = _http_get_json(INDEX_URL)
        self._entries = self._parse_index(data)
        return self

    async def refresh_async(self) -> "MeshCatalog":
        data = await _http_get_json_async(INDEX_URL)
        self._entries = self._parse_index(data)
        return self

    @staticmethod
    def _parse_index(data) -> Dict[str, MeshEntry]:
        entries: Dict[str, MeshEntry] = {}

        if isinstance(data, dict) and data.get("version", 1) >= 2:
            for name, info in data.get("meshes", {}).items():
                entries[name] = MeshEntry(
                    name=name,
                    sizes=list(info.get("sizes", [])),
                    types=list(info.get("types", [])),
                )
            return entries

        # v1 fallback: flat list
        files = data if isinstance(data, list) else data.get("files", [])
        for f in files:
            stem, dot, ext = f.rpartition(".")
            if not dot or ext not in ALLOWED_TYPES:
                continue
            m = _SIZE_RE.match(stem)
            if m:
                base, size = m.group(1), m.group(2)
            else:
                base, size = stem, None
            if base not in entries:
                entries[base] = MeshEntry(name=base)
            if size and size not in entries[base].sizes:
                entries[base].sizes.append(size)
            if ext not in entries[base].types:
                entries[base].types.append(ext)

        return entries

    # -- querying -----------------------------------------------------------

    @property
    def entries(self) -> Dict[str, MeshEntry]:
        return self._entries

    def names(self) -> List[str]:
        return sorted(self._entries)

    def get(self, name: str) -> MeshEntry:
        if name in self._entries:
            return self._entries[name]
        # Try fuzzy: user might pass "square" instead of "square__mesh"
        matches = [k for k in self._entries if name in k]
        if len(matches) == 1:
            return self._entries[matches[0]]
        if len(matches) > 1:
            raise KeyError(
                f"Ambiguous mesh name '{name}'. Matches: {matches}"
            )
        raise KeyError(
            f"Mesh '{name}' not found. Use catalog.list() to see available meshes."
        )

    def search(self, pattern: str) -> List[MeshEntry]:
        """Return entries whose name matches a substring or regex."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        return [e for e in self._entries.values() if rx.search(e.name)]

    def by_category(self) -> Dict[str, List[MeshEntry]]:
        """Group entries by their directory-based category."""
        groups: Dict[str, List[MeshEntry]] = {}
        for e in sorted(self._entries.values(), key=lambda x: x.name):
            cat = e.category or "Other"
            groups.setdefault(cat, []).append(e)
        return groups

    # -- display (mirrors CLI overview/list/show) ---------------------------

    def list(self, category: Optional[str] = None) -> Dict[str, List[MeshEntry]]:
        """Print a grouped overview and return the groups dict.

        Mirrors ``zoomy overview mesh`` from the CLI.
        """
        groups = self.by_category()
        if category:
            key = next(
                (k for k in groups if k.lower().startswith(category.lower())),
                None,
            )
            if key:
                groups = {key: groups[key]}
            else:
                print(f"No category matching '{category}'.")
                return {}

        for cat, entries in groups.items():
            print(f"\n  {cat.upper()}")
            print("  " + "-" * 50)
            for e in entries:
                sizes_str = f"  [{', '.join(e.sizes)}]" if e.sizes else ""
                types_str = f"  ({', '.join(sorted(e.types))})" if e.types else ""
                print(f"    {e.short_name:<35}{sizes_str}{types_str}")
        print()
        return groups

    def show(self, name: str) -> MeshEntry:
        """Print details for a single mesh. Mirrors ``zoomy show mesh <name>``."""
        entry = self.get(name)
        print()
        print(f"  Name:      {entry.name}")
        print(f"  Display:   {entry.short_name}")
        if entry.category:
            print(f"  Category:  {entry.category}")
        if entry.sizes:
            print(f"  Sizes:     {', '.join(entry.sizes)}")
        if entry.types:
            print(f"  Types:     {', '.join(sorted(entry.types))}")
        print(f"  URL:       {entry.url()}")
        print()
        return entry

    # -- download -----------------------------------------------------------

    def download(
        self,
        name: str,
        *,
        size: str = "medium",
        filetype: str = "h5",
        folder: Optional[str | Path] = None,
    ) -> Path:
        """Download a mesh file and return the local path.

        Mirrors ``zoomy select mesh <name>`` conceptually — you pick a mesh
        and it becomes available locally.

        Parameters
        ----------
        name : str
            Mesh name (full or partial).
        size : str
            One of ``"coarse"``, ``"medium"``, ``"fine"``.
        filetype : str
            ``"h5"``, ``"msh"``, or ``"geo"``.
        folder : Path, optional
            Download directory. Defaults to ``~/.zoomy/meshes/``.
        """
        if filetype not in ALLOWED_TYPES:
            raise ValueError(f"Invalid filetype '{filetype}'. Allowed: {ALLOWED_TYPES}")

        entry = self.get(name)
        dest = Path(folder) if folder else self._cache_dir
        dest.mkdir(parents=True, exist_ok=True)

        # Try size-suffixed first, then unsuffixed fallback
        candidates = self._download_candidates(entry, filetype, size)

        for filename, url in candidates:
            out_path = dest / filename
            if out_path.exists():
                return out_path
            try:
                data = _http_get_bytes(url)
                out_path.write_bytes(data)
                return out_path
            except Exception:
                continue

        tried = [url for _, url in candidates]
        raise FileNotFoundError(
            f"Could not download mesh '{entry.name}' (size={size}, type={filetype}).\n"
            f"Tried: {tried}"
        )

    async def download_async(
        self,
        name: str,
        *,
        size: str = "medium",
        filetype: str = "h5",
        folder: Optional[str | Path] = None,
    ) -> Path:
        """Async variant of :meth:`download` for Pyodide / JupyterLite."""
        if filetype not in ALLOWED_TYPES:
            raise ValueError(f"Invalid filetype '{filetype}'. Allowed: {ALLOWED_TYPES}")

        entry = self.get(name)
        dest = Path(folder) if folder else self._cache_dir
        dest.mkdir(parents=True, exist_ok=True)

        candidates = self._download_candidates(entry, filetype, size)

        for filename, url in candidates:
            out_path = dest / filename
            if out_path.exists():
                return out_path
            try:
                data = await _http_get_bytes_async(url)
                out_path.write_bytes(data)
                return out_path
            except Exception:
                continue

        tried = [url for _, url in candidates]
        raise FileNotFoundError(
            f"Could not download mesh '{entry.name}' (size={size}, type={filetype}).\n"
            f"Tried: {tried}"
        )

    @staticmethod
    def _download_candidates(
        entry: MeshEntry, filetype: str, size: str
    ) -> List[tuple[str, str]]:
        """Return (filename, url) pairs to try, in priority order."""
        candidates = []
        if entry.sizes:
            candidates.append(
                (entry.filename(filetype, size), entry.url(filetype, size))
            )
        # Unsuffixed fallback (backward compat / meshes without sizes)
        plain = f"{entry.name}.{filetype}"
        candidates.append((plain, FILES_URL + plain))
        return candidates

    # -- convenience: download + load in one step --------------------------

    def load(
        self,
        name: str,
        *,
        size: str = "medium",
        folder: Optional[str | Path] = None,
    ):
        """Download (if needed) and return a ``Mesh`` object.

        Requires ``zoomy_core.mesh.mesh.Mesh`` to be importable.
        """
        path = self.download(name, size=size, filetype="h5", folder=folder)
        from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh

        return Mesh.from_hdf5(str(path))

    async def load_async(
        self,
        name: str,
        *,
        size: str = "medium",
        folder: Optional[str | Path] = None,
    ):
        """Async variant of :meth:`load`."""
        path = await self.download_async(name, size=size, filetype="h5", folder=folder)
        from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh

        return Mesh.from_hdf5(str(path))

    # -- cache management ---------------------------------------------------

    def clear_cache(self) -> int:
        """Remove all cached mesh files. Returns the number of files removed."""
        count = 0
        if self._cache_dir.exists():
            for f in self._cache_dir.iterdir():
                if f.is_file():
                    f.unlink()
                    count += 1
        return count

    def cached(self) -> List[Path]:
        """List locally cached mesh files."""
        if not self._cache_dir.exists():
            return []
        return sorted(f for f in self._cache_dir.iterdir() if f.is_file())

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MeshCatalog({len(self._entries)} meshes)"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> MeshEntry:
        return self.get(name)
