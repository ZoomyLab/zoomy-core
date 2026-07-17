#!/usr/bin/env python
"""Fast-verify gate for ``zoomy_core`` — "is this change ok?" in 1-2 minutes.

ONE entry point.  Runs the DEFAULT (small) test tier — the pre-publish gate,
with ``large`` / ``rederive`` deselected (see ``tests/README.md``) — prints the
wall-clock time, and exits non-zero on any failure so CI / a git hook can gate
on it.

    micromamba run -n zoomy python scripts/verify.py             # full small tier
    micromamba run -n zoomy python scripts/verify.py --changed   # only the tests
                                                                 # for changed subdirs
    micromamba run -n zoomy python scripts/verify.py -k sme -x    # extra args pass
                                                                 # straight to pytest

The small tier assumes a WARM derivation cache (the shipped ``_prebuilt`` set +
the REQ-163 ``sm_cache``).  After a ``CACHE_VERSION`` bump or a derivation /
builder edit, regenerate it FIRST — otherwise the first run pays the cold
symbolic-build cost:

    python -m zoomy_core.systemmodel.build_prebuilt_cache
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent          # zoomy_core repo root
BUDGET_S = 120.0                                                # the 1-2 min acceptance


def changed_test_targets() -> list[str]:
    """Map ``git diff`` (unstaged+staged+untracked) to ``tests/<subdir>`` targets.

    Deliberately simple (the ``--changed`` fast loop is best-effort): a changed
    test file runs directly; a changed ``zoomy_core/<area>/…`` source file runs
    ``tests/<area>/`` when that subdir exists.  If nothing maps, returns ``[]``
    and the caller falls back to the full small tier (never silently skip)."""
    tests_dir = REPO / "tests"
    subdirs = {p.name for p in tests_dir.iterdir() if p.is_dir()}
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
        out += subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    targets: set[str] = set()
    for line in out.splitlines():
        f = line.strip()
        if not f:
            continue
        parts = pathlib.Path(f).parts
        if f.startswith("tests/") and f.endswith(".py") and (REPO / f).exists():
            targets.add(f)
        elif len(parts) >= 2 and parts[0] == "zoomy_core" and parts[1] in subdirs:
            targets.add(f"tests/{parts[1]}")
    return sorted(targets)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast-verify gate: run the default (small) test tier with a "
                    "wall-clock printout; non-zero exit on failure.")
    ap.add_argument("--changed", action="store_true",
                    help="only run tests for subdirs touched by git diff "
                         "(falls back to the full small tier if nothing maps).")
    ap.add_argument("--durations", type=int, default=10, metavar="N",
                    help="show the N slowest tests (0 = off; default 10).")
    ap.add_argument("pytest_args", nargs="*",
                    help="extra args forwarded verbatim to pytest.")
    args = ap.parse_args(argv)

    targets = ["tests/"]
    if args.changed:
        mapped = changed_test_targets()
        if mapped:
            targets = mapped
            print(f"verify: --changed → {' '.join(targets)}", flush=True)
        else:
            print("verify: --changed found no mapped subdirs → full small tier",
                  flush=True)

    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "-p", "no:cacheprovider"]
    if args.durations:
        cmd += [f"--durations={args.durations}"]
    cmd += args.pytest_args

    print(f"verify: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    wall = time.time() - t0

    status = "PASS" if rc == 0 else "FAIL"
    budget = "within" if wall <= BUDGET_S else "OVER"
    print(f"\nverify: {status} in {wall:.1f}s wall-clock "
          f"({budget} the {BUDGET_S:.0f}s budget)", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
