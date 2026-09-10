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

"""Reducing a trajectory: POD in space *and* time, then ECSW over the states it visits.

The transient counterpart of ``navier_stokes_ecsw.py``, and almost all of it is inherited. The
reduced spatial operator is the same -- an exact convective tensor, a projected viscous block, and
a stabilisation sampled on a few faces -- because the *step* is the steady problem plus a mass
term, and only the mass term is new. What changes is what the snapshots are and how many.

**Snapshots are trajectories.** A parameter no longer contributes one state but ``nt + 1`` of
them, so the POD is over space and time together and the training set grows with the time
resolution. That is a cost, and it lands hardest on ECSW.

**The ECSW training matrix grows in its rows, not its columns.** ``G`` is
``(n_states * r) x n_faces``: refining the mesh widens it, and stepping in time lengthens it.
Measured below, and at production settings it is the binding constraint -- 24 parameters at 640
BDF-2 steps with ``r = 8`` is 122880 rows, or **7.6 GB per rank** at refinement 6 in 2D, gathered
whole before anything is solved.

``sketch_rows`` fits on a Gaussian sketch of those rows instead, applied to each state's block as
it is assembled so the matrix is never formed. Measured, degree 2, refinement 3, 6 trajectories
of 33 levels (1584 rows), tolerance 1e-2::

    sketch       faces      residual     ROM error       MB
     exact      19/144      9.93e-03    3.0417e-02     1.74
       128      20/144      9.10e-03    3.0418e-02     0.14
        32      15/144      3.01e-02    3.0420e-02     0.04

A sketch of 128 rows reproduces the exact fit -- 20 faces against 19, and the same residual to
within ten per cent -- for **a twelfth of the memory**. At the production row count that ratio is
closer to a thousand.

.. warning::
   The residual reported for a sketched fit is measured on a **second, independent** sketch that
   is never fitted against. Reporting it on its own sketch is meaningless: NNLS minimises over
   that sketch, so the number is in-sample and biased low -- a six-row sketch reports 5e-16 while
   being 28% wrong. See :func:`~exadg.mor.reductors.local_ecsw_weights`.

**Read the ROM error column with care.** It does not move, and that is not evidence that the
sketch is free: the reduced error here is dominated by basis truncation at 3e-02, well above what
the hyper-reduction contributes. The residual column is what measures the fit, and it does move.

The flow relaxes monotonically to a steady state -- at Re = 2-6 there is nothing else for it to do
-- so this reduces a startup transient rather than a dynamics. Good for the machinery, and a poor
demonstration of it; see ``navier_stokes_transient.py`` for what to change.

Runs unchanged on any number of ranks::

    python python/examples/navier_stokes_transient_rom.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_transient_rom.py

Run from the repository root.
"""

import numpy as np
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.models.instationary_saddle_point import mpi_instationary_saddle_point_model
from exadg.mor.reductors import (
    InstationaryECSWStokesReductor,
    InstationaryTensorStokesReductor,
)

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes_transient.json"
DEGREE, REFINEMENTS = 2, 3
T, NT, ORDER = 4.0, 32, 2
N_TRAIN, N_TEST, N_MODES = 6, 2, 4
AMPLITUDES = (0.5, 1.5)
SKETCHES = (None, 128, 32)
TOLERANCE = 1.0e-2


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    model, (velocity, pressure) = mpi_instationary_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE,
        degree=DEGREE, refinements=REFINEMENTS, T=T, nt=NT, order=ORDER,
    )
    n_parameters = model.parameters["mu"]

    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, n_parameters))]
    test = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TEST, n_parameters))]

    trajectories = model.solution_space.empty()
    for mu in train:
        trajectories.append(model.solve(mu))

    # Solved once and kept: every reductor below is measured against the same trajectories, and
    # a full-order trajectory costs nt + 1 nonlinear solves.
    reference = [model.solve(mu) for mu in test]

    basis_u, singular_u = pod(trajectories.blocks[0], product=model.u_product, modes=N_MODES)
    basis_p, _ = pod(trajectories.blocks[1], product=model.p_product, modes=N_MODES)
    bases = dict(
        RB_u=basis_u, RB_p=basis_p, u_product=model.u_product, p_product=model.p_product
    )

    exact_reductor = InstationaryTensorStokesReductor(model, **bases)
    exact_rom = exact_reductor.reduce()
    exact_error = worst_error(model, exact_reductor, exact_rom, test, reference)

    r = len(exact_rom.operator.momentum.basis)
    rows = len(trajectories) * r

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"time steps         : {NT} of BDF-{ORDER} over [0, {T}]")
    print(f"snapshots          : {N_TRAIN} trajectories x {NT + 1} levels = {len(trajectories)}")
    print(f"velocity spectrum  : {np.array2string(singular_u[:N_MODES], precision=4)}")
    print(f"reduced dimension  : {exact_rom.solution_space.dim}")
    print(f"error, exact S     : {exact_error:.4e}")

    print(f"\nECSW over the states the trajectories visit: G is {rows} x n_faces")
    print(f"  {'sketch':>8}  {'faces':>10}  {'residual':>10}  {'ROM error':>11}  {'MB':>7}")

    for sketch in SKETCHES:
        reductor = InstationaryECSWStokesReductor(
            model, training_states=trajectories.blocks[0], tolerance=TOLERANCE,
            sketch_rows=sketch, **bases,
        )
        rom = reductor.reduce()
        momentum = rom.operator.momentum

        error = worst_error(model, reductor, rom, test, reference)
        megabytes = (rows if sketch is None else sketch) * momentum.n_candidates * 8 / 2**20

        print(
            f"  {'exact' if sketch is None else sketch:>8}  "
            f"{momentum.n_faces:>4d}/{momentum.n_candidates:<5d}  "
            f"{momentum.training_residual:>10.2e}  {error:>11.4e}  {megabytes:>7.2f}"
        )

        assert error < 2.0 * exact_error, "sampling the stabilisation changed the answer"

    print(
        "\nThe transient reduction is the steady one plus a mass term, which is what carrying the\n"
        "step through the interface bought: the tensor, the viscous block and the sampled faces\n"
        "are unchanged, and the reduced model is stepped by the full-order model's own scheme with\n"
        "only the step solver swapped.\n"
        "\n"
        "What is genuinely new is the size of the training set. A parameter contributes a whole\n"
        "trajectory, so the ECSW matrix grows in its rows, and sketching them is what keeps that\n"
        "affordable -- a twelfth of the memory here, and far less than that at production step\n"
        "counts. The residual column is what says whether a sketch was large enough; the ROM error\n"
        "column cannot, because basis truncation dominates it."
    )


def worst_error(model, reductor, rom, test, reference):
    """Largest relative error over the test trajectories, in the mixed product.

    Taken over time levels as well as parameters -- a trajectory is only as good as its worst
    moment, and an average would hide a reduced model that drifts.
    """
    product = model.products["mixed"]

    worst = 0.0
    for mu, U in zip(test, reference):
        error = (U - reductor.reconstruct(rom.solve(mu))).norm(product).max()
        worst = max(worst, error / U.norm(product).max())

    return worst


if __name__ == "__main__":
    main()
