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

"""Reducing a trajectory: a hierarchical POD in space *and* time, then ECSW over what it visits.

The transient counterpart of ``navier_stokes_ecsw.py``, and almost all of it is inherited. The
reduced spatial operator is the same -- an exact convective tensor, a projected viscous block, and
a stabilisation sampled on a few faces -- because the *step* is the steady problem plus a mass
term, and only the mass term is new. What changes is what the snapshots are and how many.

**Snapshots are trajectories.** A parameter no longer contributes one state but ``nt + 1`` of
them, so the basis is built over space and time together and the training set grows with the time
resolution. That is a cost, and it lands twice: on the basis, and harder on ECSW.

**The basis comes from a hierarchical POD.** ``pymor.algorithms.pod`` uses the method of
snapshots, which forms an ``N x N`` Gramian and eigendecomposes it -- fine at 264 snapshots, and
1.76 GB *per rank* plus a six-minute eigensolve at 24 parameters by 641 levels. ``inc_hapod``
compresses each trajectory as it arrives and then compresses the modes, so no matrix bigger than
one chunk is ever formed, and it does so under a **certified** bound on the l2-mean projection
error rather than a mode count. Measured, 8 trajectories of 33 levels::

    method            modes   abs l2-mean       rel     time     matrix
    POD, modes=4          4     3.375e-03  4.263e-02    0.078    264x264
    POD, modes=8          8     5.206e-04  6.575e-03    0.091    264x264
    HAPOD, rel=1e-01      3     5.200e-03  6.567e-02    0.029      33x33
    HAPOD, rel=3e-02      6     1.523e-03  1.924e-02    0.033      33x33
    HAPOD, rel=1e-02      8     5.206e-04  6.576e-03    0.035      33x33

At matched accuracy HAPOD produces the same basis as the POD -- 8 modes at 5.206e-04 either way
-- from a Gramian sixty-four times smaller, and it is faster. Every row met its bound.

``eps`` is an *absolute* l2-mean error, so it is set here as a fraction of the rms snapshot norm;
velocity and pressure have different scales and a shared absolute tolerance would mean different
things to each.

.. note::
   **This script keeps every trajectory**, which is what makes it readable and what stops it
   scaling: 24 parameters at 641 levels and refinement 6 is nine gigabytes. It does not have to be
   that way -- ``navier_stokes_streaming.py`` builds the same reduced model without ever holding a
   snapshot, by carrying each level's coefficients through the basis updates instead of projecting
   at the end. The fit it produces differs by a face or two and by 0.12 % in the reduced error.

**The ECSW training matrix grows in its rows, not its columns.** ``G`` is
``(n_states * r) x n_faces``: refining the mesh widens it, and stepping in time lengthens it.
Measured below, and at production settings it is the binding constraint -- 24 parameters at 640
BDF-2 steps with ``r = 8`` is 122880 rows, or **7.6 GB per rank** at refinement 6 in 2D, gathered
whole before anything is solved.

``sketch_rows`` fits on a Gaussian sketch of those rows instead, applied to each state's block as
it is assembled so the matrix is never formed. Measured, degree 2, refinement 3, 6 trajectories
of 33 levels on the basis above (1782 rows), tolerance 1e-2::

    sketch       faces      residual     ROM error       MB
     exact      20/144      9.56e-03    3.9434e-02     1.96
       128      18/144      1.40e-02    3.9434e-02     0.14
        32      15/144      2.11e-02    3.9434e-02     0.04

A sketch of 128 rows keeps 18 faces against 20 for **a fourteenth of the memory**, at a residual
within a factor of 1.5. At the production row count that ratio is closer to a thousand.

.. warning::
   The residual reported for a sketched fit is measured on a **second, independent** sketch that
   is never fitted against. Reporting it on its own sketch is meaningless: NNLS minimises over
   that sketch, so the number is in-sample and biased low -- a six-row sketch reports 5e-16 while
   being 28% wrong. See :func:`~exadg.mor.reductors.local_ecsw_weights`.

A run leaves ``navier_stokes_transient_rom_{velocity,pressure}.pvd`` behind: the full-order
trajectory, the hyper-reduced one and their difference as three fields of one animated series.
Open the ``.pvd``, not the individual records.

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
from pymor.algorithms.hapod import inc_vectorarray_hapod
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
N_TRAIN, N_TEST = 40, 4
AMPLITUDES = (0.5, 1.5)
SKETCHES = (None, 128, 32)
TOLERANCE = 1.0e-2

#: Relative l2-mean projection error the basis is built to, and HAPOD's balance parameter.
BASIS_TOLERANCE, OMEGA = 3.0e-2, 0.9


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

    basis_u, error_u = trajectory_basis(trajectories.blocks[0], model.u_product, N_TRAIN)
    basis_p, error_p = trajectory_basis(trajectories.blocks[1], model.p_product, N_TRAIN)
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
    print(f"basis, HAPOD       : {len(basis_u)} velocity modes at {error_u:.2e} relative, "
          f"{len(basis_p)} pressure at {error_p:.2e}")
    print(f"  largest Gramian  : {int(np.ceil(len(trajectories) / N_TRAIN))} square, against "
          f"{len(trajectories)} square for the method of snapshots")
    print(f"reduced dimension  : {exact_rom.solution_space.dim}")
    print(f"error, exact S     : {exact_error:.4e}")

    print(f"\nECSW over the states the trajectories visit: G is {rows} x n_faces")
    print(f"  {'sketch':>8}  {'faces':>10}  {'residual':>10}  {'ROM error':>11}  {'MB':>7}")

    last = None
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
        last = (reductor, rom)

    # Something to look at: the full-order trajectory, the hyper-reduced one and their difference,
    # at one test parameter, as three fields of one time series per block. A .pvd collection ties
    # the per-level records together, so ParaView animates it and the record names stop mattering.
    reductor, rom = last
    mu = test[0]
    U_fom = reference[0]
    U_rom = reductor.reconstruct(rom.solve(mu))
    times = np.linspace(0.0, T, len(U_fom))

    written = model.visualize(
        (U_fom, U_rom, U_fom - U_rom),
        legend=("fom", "rom", "error"),
        filename="output/pymor/navier_stokes_transient_rom",
        times=times,
    )
    print(f"\nwrote {len(U_fom)} time levels:")
    for record in written:
        print(f"      {record}")

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


def trajectory_basis(snapshots, product, chunks):
    """A hierarchical POD of the snapshots, to a *relative* l2-mean projection error.

    ``inc_vectorarray_hapod`` bounds the **absolute** l2-mean error, so the tolerance is scaled by
    the rms snapshot norm here -- velocity and pressure differ by an order of magnitude in this
    problem, and one absolute number would mean something different to each.

    Returns the modes and the relative error actually achieved, which is what says the bound held.
    """
    scale = np.sqrt((snapshots.norm(product) ** 2).mean())
    modes, _, _ = inc_vectorarray_hapod(
        chunks, snapshots, BASIS_TOLERANCE * scale, omega=OMEGA, product=product
    )

    residual = snapshots - modes.lincomb(product.apply2(modes, snapshots))

    return modes, np.sqrt((residual.norm(product) ** 2).mean()) / scale


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
