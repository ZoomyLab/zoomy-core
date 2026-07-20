"""``ExternalProcedure`` — the third declaration kind (validation appendix D12).

``register_symbolic_function`` cannot carry the solver's opaque blocks: it
evaluates the body EAGERLY at registration (``basefunction.py``) and a ``None``
body yields a ZERO definition, and the ``kernel_functions`` externals are
expression-position scalar atoms.  The design's opaque blocks
(``prepare_mesh`` / ``mood_resolve`` / ``write_fields`` / the coupling
surfaces / …) are statement-position, side-effecting and handle-typed.

So: declare the NAME + SIGNATURE in core, supply the BODY per backend, and put
every declaration on :data:`REQUIRED_PROCEDURES` — the same contract shape
``REQUIRED_KERNELS`` (``kernel_functions.py``) already uses for the numerical
UserFunctions, so a backend missing a body is a RED TEST rather than a link
error mid-build (C++) or a ``NameError`` at lambdify (python).
"""
from __future__ import annotations

import dataclasses

from zoomy_core.solver.ir import check_arg_keys, check_procedure_name


@dataclasses.dataclass(frozen=True)
class ExternalProcedure:
    """A solver block declared in core, bodied by each backend.

    ``args`` / ``results`` are solver argument-vocabulary slots; ``doc`` states
    the CONTRACT the backend body must satisfy (that text is the whole point —
    it is what a new backend implements against).
    """

    name: str
    args: tuple = ()
    results: tuple = ()
    doc: str = ""

    def __post_init__(self):
        object.__setattr__(self, "name", check_procedure_name(self.name))
        object.__setattr__(self, "args", check_arg_keys(self.name, self.args))
        object.__setattr__(self, "results", check_arg_keys(self.name, self.results))


def _decl(name, args=(), results=(), doc=""):
    return ExternalProcedure(name=name, args=args, results=results, doc=doc)


#: The opaque solver blocks — design "Core inventory (final)", section (b).
#: Every backend ships a contract test asserting its body table covers this
#: set (see ``required_procedure_gaps``).
REQUIRED_PROCEDURES = {
    p.name: p
    for p in (
        _decl("solver_prepare_mesh", doc=(
            "mesh + NSM -> MeshRT (cell_volumes, face_area, face_normal, "
            "face_cells, inradius_f, lsq_stencils, bc_face_groups, "
            "periodic_map).  Translates ONCE into the backend's containers.")),
        _decl("solver_build_operators", doc=(
            "kernels + MeshRT -> Ops (flux_face, fluct_face, bc_face, "
            "eigenvalues, reconstruct, limiter, source, update_variables, "
            "update_aux, tableau).")),
        _decl("solver_initialize_state", doc=(
            "MeshRT + IC + BC -> S(time=0, iteration=0, i_snapshot=0, Q, "
            "Qaux).")),
        _decl("solver_halo_bc",
              args=("variables", "aux_variables", "time"), doc=(
                  "Per-stage ghost/halo synchronisation (D3): amrex "
                  "UpdateState+FillPatch+FillBoundary, unstructured ghost "
                  "synthesis.  Load-bearing EVERY stage, not once per step.")),
        _decl("solver_mood_resolve",
              args=("troubled", "q0", "q_cand", "aux_variables", "parameters",
                    "time", "dt"), doc=(
                  "OUTCOME semantics only (D5): troubled cells come back "
                  "positive and finite AND mass is exactly conserved.  The "
                  "strategy is the backend's: whole-step O1 re-step (may "
                  "replay the stage loop), face-local re-solve, or "
                  "compute-both/select on GPU.")),
        _decl("solver_implicit_source",
              args=("q_cand", "aux_variables", "parameters", "dt"), doc=(
                  "a_ii*dt point-implicit source solve (local Newton or "
                  "JFNK).")),
        _decl("solver_implicit_diffusion",
              args=("q_cand", "aux_variables", "parameters", "dt"), doc=(
                  "a_ii*dt global diffusion solve; also reports rel_resid.")),
        _decl("solver_write_fields",
              args=("variables", "aux_variables", "time", "i_snapshot"), doc=(
                  "Ordered side effect (REQ-92): snapshot index is an INPUT, "
                  "the block must not re-derive it from time.")),
        _decl("solver_adapt_mesh", doc=(
            "Regrid owner (D4), between steps only — Q0 never spans a "
            "regrid.")),
        _decl("solver_reduce_min",
              args=("lam_f", "n_faces"), doc=(
                  "Global min reduction primitive: jnp.min / gMin / "
                  "ParReduce.")),
        _decl("solver_coupling_init", doc="preCICE participant setup."),
        _decl("solver_coupling_read", args=("time", "dt"), doc=(
            "Read peer data for this window (waveform-sampled at t).")),
        _decl("solver_coupling_write", args=("variables", "time"),
              doc="Write this side's interface data."),
        _decl("solver_coupling_advance", args=("dt",), doc=(
            "Advance the coupling; yields the remaining window size.")),
        _decl("solver_coupling_checkpoint", doc=(
            "Implicit-scheme checkpoint/rollback; a NO-OP in the explicit "
            "parallel scheme (v6 §2 ruling).")),
        _decl("solver_coupling_face_fixup",
              args=("flux_face", "d_plus", "d_minus", "n_faces"), doc=(
                  "7th coupling surface (D8): the per-stage face-array fixup "
                  "(applyFrozenMassRow) at the END of flux_pass, MOOD "
                  "recompute included.  STRICT no-op when coupling is "
                  "inactive.")),
        _decl("solver_coupling_finalize", doc="Tear the participant down."),
    )
}


class MissingProcedureBody(LookupError):
    """A backend's body table does not cover ``REQUIRED_PROCEDURES``."""


def required_procedure_gaps(bodies) -> tuple:
    """Slots of :data:`REQUIRED_PROCEDURES` a backend's body table misses.

    ``bodies`` is any container supporting ``in`` (a dict of callables for the
    python backends, a set of emitted symbol names for the C++ ones).  Returns
    a sorted tuple — empty means the contract is met.  The backend's own
    contract test turns a non-empty result into a RED TEST.
    """
    return tuple(sorted(n for n in REQUIRED_PROCEDURES if n not in bodies))


def assert_procedure_bodies(bodies, backend: str) -> None:
    """Contract assertion for a backend body table; RAISES on any gap."""
    gaps = required_procedure_gaps(bodies)
    if gaps:
        raise MissingProcedureBody(
            f"backend {backend!r} supplies no body for {list(gaps)} — every "
            "REQUIRED_PROCEDURES entry needs one (missing body = red test, "
            "never a link error)"
        )
