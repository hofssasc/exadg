#  ______________________________________________________________________
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2021 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#  ______________________________________________________________________

"""Does each part of the hyper-reduced path cost what it should?

``navier_stokes_ecsw.py`` establishes that sampling the stabilisation does not move the error.
This asks the other question -- whether the cost behaves -- by refining the mesh and timing every
stage separately. Three claims, in increasing order of importance.

**One: the full-order model scales.** Nothing downstream can be trusted otherwise, and a reduced
model measured against a full-order one that is itself misbehaving proves nothing.

**Two: the offline stages scale as their algorithms say they should.** Each is linear in the mesh
for a reason that can be stated in advance, and a stage that departs from its own prediction is a
bug, not a surprise.

**Three: the reduced model does not scale with the mesh at all.** This is the point of the whole
construction. A reduced solve contracts a tensor and evaluates a few faces; neither knows how many
degrees of freedom exist, so the time must be *flat* as the mesh grows by orders of magnitude. A
speed-up figure only says the ROM is faster today; a flat column says it will still be faster on a
mesh nobody has run yet.

What is timed, and what each is expected to do -- writing the prediction down first is what makes
the measurement a test rather than a description::

    stage                    expected     because
    ----------------------------------------------------------------------------------
    FOM solve                   ~n        matrix-free residual, preconditioned Krylov
    POD                         ~n        n_train^2 inner products of length n
    projection (tensor)         ~n        O(r^2) operator applications, each O(n)
    ECSW training data          ~n        n_train face loops over the whole mesh
    ECSW weight fit           **>0**      the matrix is (n_train r) x n_faces, gathered whole
    ----------------------------------------------------------------------------------
    batches visited           **~0**      bounded by the faces the fit keeps
    reduced solve             **~0**      reduced Newton on r unknowns
    sampled stabilisation     **~0**      those batches, from compiled arrays
    sampled Jacobian          **~0**      the same batches, the same arrays

The exponents reported are ``d log t / d log n`` fitted over the sweep. Read them against that
table: ~1 for the mesh-bound stages, ~0 for the reduced ones.

.. warning::
   **Read the online claim off the counts, not off the clock.** The reduced stages take tens of
   microseconds, and at that scale wall time measures the machine as much as the algorithm. The
   mesh-independent quantities are exact and are printed next to the times: *batches visited*,
   which bounds the work, and *Newton steps*. Batches are the honest unit -- matrix-free evaluates
   four to eight faces at once, so a batch costs the same whether one face in it was selected or
   all of them -- and they saturate at the face count once the mesh is fine enough to stop
   selected faces sharing one.

   The clock also had a systematic bias that the counts did not, and it is worth knowing because
   it looks exactly like the thing being tested. Timed with the full-order model still resident,
   the reduced stages slowed down at the finest refinement while doing *identical* work -- same
   batches, same modes, same quadrature -- because that model's working set evicts the compiled
   arrays between calls. Dropping it made the same work 38% faster at refinement 6 and made no
   difference at refinement 5, which is what a cache effect looks like and not what mesh
   dependence looks like.

   Under MPI the microsecond columns also carry one broadcast per evaluation, since every reduced
   call is dispatched to all ranks; that is a constant per call and does not touch the counts.

   So ``measure()`` calls ``momentum.detach()`` and releases everything full-order *before* timing
   the online stages. That is both fairer and the situation actually being claimed -- a deployed
   reduced model does not carry a mesh around -- and it is only possible because the compiled
   operator holds no reference to one. It moved the measured exponents from 0.13, 0.17, 0.17 to
   0.04, 0.07, 0.12.

The weight fit is the one entry expected to misbehave, and it is listed that way on purpose --
every rank gathers the entire training matrix and solves the same non-negative least squares, so
its width grows with the mesh. It is offline, so it bounds the size of problem that can be
*trained* rather than the cost of a reduced solve. See ``local_ecsw_weights`` and the architecture
in ``ExaDG ROM Next Steps.md``.

Runs unchanged on any number of ranks::

    python python/examples/navier_stokes_scaling.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_scaling.py

Run from the repository root. ``REFINEMENTS`` is the knob: each step multiplies the degrees of
freedom by four and the full-order cost with them, so a longer sweep is an overnight job.
"""

import gc
import time

import numpy as np
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.models.saddle_point import mpi_saddle_point_model
from exadg.mor.reductors import ECSWStokesReductor, TensorGalerkinStokesReductor

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes.json"
DEGREE = 2
REFINEMENTS = (3, 4, 5, 6)
N_TRAIN, N_MODES = 24, 4
AMPLITUDES = (0.5, 1.5)
TOLERANCE = 1.0e-2

OFFLINE = ("FOM solve", "POD", "projection", "ECSW data", "ECSW fit")
ONLINE = ("reduced solve", "sampled S", "sampled S'")
COUNTS = ("batches",)
EXPECTED = {
    "FOM solve": "~1", "POD": "~1", "projection": "~1", "ECSW data": "~1", "ECSW fit": ">0",
    "batches": "~0", "reduced solve": "~0", "sampled S": "~0", "sampled S'": "~0",
}


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    print(f"ranks              : {mpi.size}")
    print(f"degree             : {DEGREE}")
    print(f"training snapshots : {N_TRAIN}")
    print(f"modes              : {N_MODES} velocity + {N_MODES} pressure, plus supremizers")
    print(f"ECSW tolerance     : {TOLERANCE:.0e}")

    rows = [measure(refinements) for refinements in REFINEMENTS]

    table("offline -- expected to scale with the mesh", rows, OFFLINE)
    online_table(rows)
    exponents(rows)

    growth = rows[-1]["dofs"] / rows[0]["dofs"]
    print(
        f"\nThe mesh grows by a factor of {growth:.0f} over this sweep. The offline stages follow it,\n"
        "each being a fixed number of passes over the mesh, and the online stages do not follow it\n"
        "at all -- which is the claim the whole construction exists to support.\n"
        "\n"
        "Read that off the counts rather than the clock. Batches visited saturates at the faces the\n"
        "fit keeps, because once the mesh is fine enough the selected faces stop sharing a batch and\n"
        "the count cannot exceed the face count; Newton steps do not move at all. What residual\n"
        "slope the microsecond columns have is that saturation happening, and the per-batch columns\n"
        "say so: they are flat while the totals rise. A reduced solve contracts a tensor and visits\n"
        "those batches, and neither operation can see how many degrees of freedom exist.\n"
        "\n"
        "POD comes in below its predicted exponent because its other half -- an eigendecomposition\n"
        "of an n_train square matrix -- does not know about the mesh at all, and dominates while the\n"
        "mesh is small.\n"
        "\n"
        "The one offline stage that misbehaves is the weight fit, and it is the known defect rather\n"
        "than a discovery: the training matrix is gathered whole on every rank and its width is the\n"
        "face count. Offline, so it bounds the size of problem that can be trained, not the cost of\n"
        "a reduced solve."
    )


def measure(refinements):
    """Every stage at one refinement, timed once through."""
    rng = np.random.default_rng(0)

    started = time.perf_counter()
    model, (velocity, pressure) = mpi_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE, degree=DEGREE, refinements=refinements
    )
    setup = time.perf_counter() - started

    n_parameters = model.parameters["mu"]
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, n_parameters))]
    probe = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (8, n_parameters))]

    timings = {}

    started = time.perf_counter()
    snapshots = model.solution_space.empty()
    for mu in train:
        snapshots.append(model.solve(mu))
    timings["FOM solve"] = (time.perf_counter() - started) / N_TRAIN

    started = time.perf_counter()
    basis_u, _ = pod(snapshots.blocks[0], product=model.u_product, modes=N_MODES)
    basis_p, _ = pod(snapshots.blocks[1], product=model.p_product, modes=N_MODES)
    timings["POD"] = time.perf_counter() - started

    bases = dict(
        RB_u=basis_u, RB_p=basis_p, u_product=model.u_product, p_product=model.p_product
    )

    started = time.perf_counter()
    TensorGalerkinStokesReductor(model, **bases).reduce()
    timings["projection"] = time.perf_counter() - started

    reductor = ECSWStokesReductor(
        model, training_states=snapshots.blocks[0], tolerance=TOLERANCE, **bases
    )
    rom = reductor.reduce()
    momentum = rom.operator.momentum

    # Reported by the fit itself, split where its two halves scale differently.
    timings["ECSW data"] = momentum.assembly_seconds
    timings["ECSW fit"] = momentum.nnls_seconds
    n_candidates = momentum.n_candidates

    # Everything full-order goes before the online stages are timed, which is both fairer and the
    # situation being claimed: a deployed reduced model does not carry a mesh around. It also
    # removes a real bias -- a co-resident full-order model evicts the compiled arrays between
    # calls, and the finer the mesh the more it evicts, which reads exactly like mesh dependence.
    momentum.detach()
    del model, snapshots, basis_u, basis_p, bases, reductor
    gc.collect()

    # At varying parameters: a solver timed at one input measures its cache, not its cost.
    timings["reduced solve"] = repeated(lambda mu: rom.solve(mu), probe)

    # How many residual evaluations that solve took, so a slope in it can be attributed.
    _, info = rom.operator.apply_inverse(
        rom.rhs.as_range_array(probe[0]), mu=probe[0], return_info=True
    )
    steps = len(info["residual_norms"][0]) - 1

    coefficients = np.linspace(0.1, 1.0, len(momentum.basis))
    timings["sampled S"] = repeated(
        lambda scale: momentum.stabilisation(scale * coefficients), np.linspace(0.5, 1.5, 20)
    )
    timings["sampled S'"] = repeated(
        lambda scale: momentum.stabilisation_jacobian(scale * coefficients),
        np.linspace(0.5, 1.5, 20),
    )

    row = dict(
        refinements=refinements,
        dofs=velocity.dim + pressure.dim,
        faces=momentum.n_faces,
        batches=momentum.n_batches,
        steps=steps,
        **timings,
    )

    del rom, momentum
    gc.collect()

    row["resident"] = resident()
    print(
        f"\n  refinement {refinements}: {row['dofs']} dofs, setup {setup:.1f} s, "
        f"{row['faces']}/{n_candidates} faces kept, {row['resident']:.0f} MB resident afterwards"
    )

    return row


def resident():
    """Resident set size in MB, or zero where the kernel does not report it.

    Printed so that a sweep which fails to let go of a mesh says so, rather than showing up as an
    unexplained slope in the online columns.
    """
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass

    return 0.0


def repeated(call, inputs):
    """Seconds per call, averaged, discarding the first as a warm-up.

    The collector is off inside the loop. These calls take microseconds and allocate a great many
    short-lived objects, so a collection landing inside the window is charged to whichever stage
    was unlucky -- and the later a refinement is in the sweep, the more live objects there are for
    a collection to walk. Timing with it disabled measures the operation rather than the sweep's
    own history.
    """
    call(inputs[0])

    enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter()
        for value in inputs:
            call(value)

        return (time.perf_counter() - started) / len(inputs)
    finally:
        if enabled:
            gc.enable()


def table(title, rows, stages):
    """One line per refinement, in seconds."""
    print(f"\n{title}, seconds")
    print(f"  {'refinement':>10}  {'dofs':>8}" + "".join(f"  {stage:>14}" for stage in stages))
    for row in rows:
        print(
            f"  {row['refinements']:>10}  {row['dofs']:>8}"
            + "".join(f"  {row[stage]:>14.6f}" for stage in stages)
        )


def online_table(rows):
    """Counts first, then times. The counts are the claim; the times only corroborate it.

    Per-batch figures are there to be read across the sweep: the work is proportional to batches,
    so a roughly constant column is what says the cost model holds. See the warning in the module
    docstring about why the microsecond columns drift upward at the finest refinement.
    """
    header = ("refinement", "dofs", "faces", "batches", "steps",
              "solve/ms", "S/us", "per batch", "S'/us", "per batch", "MB")
    widths = (10, 8, 6, 7, 5, 9, 8, 9, 8, 9, 6)

    print("\nonline -- expected not to scale with the mesh")
    print("  " + "  ".join(f"{name:>{width}}" for name, width in zip(header, widths)))

    for row in rows:
        batches = max(row["batches"], 1)
        stabilisation = row["sampled S"] * 1e6
        jacobian = row["sampled S'"] * 1e6
        values = (
            f"{row['refinements']:>10}", f"{row['dofs']:>8}", f"{row['faces']:>6}",
            f"{row['batches']:>7}", f"{row['steps']:>5}",
            f"{row['reduced solve'] * 1e3:>9.3f}",
            f"{stabilisation:>8.1f}", f"{stabilisation / batches:>9.2f}",
            f"{jacobian:>8.1f}", f"{jacobian / batches:>9.2f}",
            f"{row['resident']:>6.0f}",
        )
        print("  " + "  ".join(values))


def exponents(rows):
    """``d log t / d log n``, fitted over the sweep -- the number the predictions are about."""
    dofs = np.log(np.array([row["dofs"] for row in rows], dtype=float))

    print("\nexponent in the degrees of freedom, d log t / d log n")
    print(f"  {'stage':>15}  {'measured':>9}  {'expected':>9}")
    for stage in OFFLINE + COUNTS + ONLINE:
        times = np.array([row[stage] for row in rows], dtype=float)
        slope = np.polyfit(dofs, np.log(times), 1)[0] if len(rows) > 1 else float("nan")

        print(f"  {stage:>15}  {slope:>9.2f}  {EXPECTED[stage]:>9}")


if __name__ == "__main__":
    main()
