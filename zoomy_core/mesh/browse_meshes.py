import os

try:
    import meshio

    _HAVE_MESHIO = True
except:
    _HAVE_MESHIO = False

# Detect Pyodide/JupyterLite
try:
    from pyodide.http import pyfetch

    _IN_PYODIDE = True
except ImportError:
    import requests

    _IN_PYODIDE = False

# Base URLs for GitHub Pages hosting
BASE_URL = "https://zoomylab.github.io/meshes/"
INDEX_URL = BASE_URL + "index.json"

ALLOWED_TYPES = {"msh", "geo", "h5"}


# ============================================================
# Helpers: fetch JSON / bytes in both environments
# ============================================================
async def _fetch_json(url):
    if _IN_PYODIDE:
        resp = await pyfetch(url)
        return await resp.json()
    else:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()


async def _fetch_bytes(url):
    if _IN_PYODIDE:
        resp = await pyfetch(url)
        return await resp.bytes()
    else:
        r = requests.get(url)
        r.raise_for_status()
        return r.content


# ============================================================
# Mesh listing (async + sync)
# ============================================================
def _base_name(filename):
    """Convert 'square_mesh.msh' → 'square_mesh'."""
    return filename.rsplit(".", 1)[0]


async def show_meshes_async(do_print=True):
    files = await _fetch_json(INDEX_URL)

    # Filter out unwanted file types
    filtered = [f for f in files if f.split(".")[-1] in ALLOWED_TYPES]

    # Unique base mesh names
    base_names = sorted({_base_name(f) for f in filtered})

    if do_print:
        print("Meshes available:")
        for name in base_names:
            print(" -", name)

    return base_names


def show_meshes(do_print=True):
    """
    CPython → synchronous
    JupyterLite → returns coroutine → use `await show_meshes()`
    """
    if _IN_PYODIDE:
        return show_meshes_async(do_print)

    # CPython sync path
    r = requests.get(INDEX_URL)
    r.raise_for_status()
    files = r.json()

    filtered = [f for f in files if f.split(".")[-1] in ALLOWED_TYPES]
    base_names = sorted({_base_name(f) for f in filtered})

    if do_print:
        print("Meshes available:")
        for name in base_names:
            print(" -", name)

    return base_names


# ============================================================
# Mesh download
# ============================================================
async def download_mesh_async(mesh_name, folder="./", filetype="msh"):
    if filetype not in ALLOWED_TYPES:
        raise ValueError(f"Invalid filetype '{
                         filetype}'. Allowed: {ALLOWED_TYPES}")

    filename = f"{mesh_name}.{filetype}"
    url = BASE_URL + "meshes/" + filename

    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, filename)

    print(f"Downloading {filename} → {out_path}")

    data = await _fetch_bytes(url)
    with open(out_path, "wb") as f:
        f.write(data)

    print("Download complete.")
    return out_path


def download_mesh(mesh_name, folder="./", filetype="msh"):
    """
    CPython → synchronous
    JupyterLite → async → use `await download_mesh(...)`
    """
    if _IN_PYODIDE:
        return download_mesh_async(mesh_name, folder, filetype)

    filename = f"{mesh_name}.{filetype}"
    url = BASE_URL + "meshes/" + filename

    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, filename)

    print(f"Downloading {filename} → {out_path}")

    r = requests.get(url)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

    print("Download complete.")
    return out_path


# ============================================================
# meshio physical boundaries
# ============================================================
def get_boundary_names(mesh_path, do_print=True):
    if not _HAVE_MESHIO:
        raise RuntimeError(
            "get_boundary_names requires meshio, which is not available."
        )
    if do_print:
        print(f"Reading mesh: {mesh_path}")
    mesh = meshio.read(mesh_path)

    tags = {}
    if "gmsh:physical" in mesh.cell_data:
        for block, phys in zip(mesh.cells, mesh.cell_data["gmsh:physical"]):
            tags[block.type] = sorted(set(phys))

    if do_print:
        print("Found physical tags:")
        for c, names in tags.items():
            print(f" - {c}: {names}")

    return tags
