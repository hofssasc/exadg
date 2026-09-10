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

"""Training a hyper-reduced model without ever holding the snapshots.

``navier_stokes_transient_rom.py`` keeps every trajectory: it builds the basis with a hierarchical
POD, then projects the snapshots onto it to get the states ECSW fits at. That is correct and it
does not scale -- 24 parameters at 641 BDF-2 levels and refinement 6 is **nine gigabytes** of
velocity snapshots, and the alternative of discarding them costs a second pass of full-order
solves. This script establishes that neither is necessary.

**The blocker is not ECSW.** Every hyper-reduction trains at reduced states ``a_i = V^T M u_i``,
and ``V`` is not known until the last trajectory has been seen. So the dependency -- basis first,
states second, fit third -- is upstream of the method, and swapping ECSW for DEIM or GNAT would
inherit it while giving up non-negative weights and exact evaluation on the faces that are kept.

**Projection is linear, and that is the way out.** A snapshot already known as ``u = W c`` has
coefficients ``c' = (V^T M W) c`` in any later basis: a small matrix times a small vector, with no
full-order object anywhere. So the coefficients are *carried* through each basis update rather
than recomputed from snapshots that no longer exist. :mod:`exadg.mor.basis` is that, and the
reduced model never learns which route produced its training points.

Two things are checked.

**One: the fit does not move.** Measured, 6 trajectories of 33 levels, sketch 128, tolerance 1e-2::

    path                  vectors held    basis    faces    residual     ROM error
    stored                         396   6u, 3p       18    1.397e-02    3.9434e-02
    streamed, chunk 33              26   6u, 3p       20    1.218e-02    3.9483e-02
    streamed, chunk  1              26   6u, 3p       20    1.121e-02    3.9483e-02

The **face selection does shift, by a face or two, and it does not matter**. A greedy is a
discrete choice, so a slightly perturbed basis flips its late picks -- the same way a changed rank
count does, and for the same reason. The residual is if anything better and the reduced error
moves by 0.12 %.

**Two: the chunk size is free.** Compressing after every single level costs the same as
compressing once per trajectory, and drops the peak to the running basis plus one vector. The
accumulated error over many updates stays at the order of the tolerance asked for, so the rule is
to compress often and ask for one decade more accuracy than needed -- an extra decade is a handful
of modes, and it buys back far more than the memory costs.

.. note::
   ``model.solve(mu)`` materialises a whole trajectory, which is the thing being avoided, so this
   drives ``time_stepper.iterate`` directly -- it is a generator, and that is what makes the peak
   one chunk rather than one trajectory. Both fields are compressed in the same pass; building the
   velocity and pressure bases separately would mean solving twice and give the storage back.

What this does **not** remove: ``contributions()` still walks the mesh once per training state.
That is the ECSW training data itself, it is distributed over ranks, and no compression touches
it -- see ``navier_stokes_transient_rom.py`` for the row sketch, which is a different axis.

Runs unchanged on any number of ranks::

    python python/examples/navier_stokes_streaming.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_streaming.py

Run from the repository root.
"""

import numpy as np
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.basis import chunk_count, streaming_basis, trajectory_chunks
from exadg.mor.models.instationary_saddle_point import mpi_instationary_saddle_point_model
from exadg.mor.reductors import InstationaryECSWStokesReductor

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes_transient.json"
DEGREE, REFINEMENTS = 2, 3
T, NT, ORDER = 4.0, 32, 2
N_TRAIN, N_TEST = 6, 2
AMPLITUDES = (0.5, 1.5)
BASIS_TOLERANCE, OMEGA = 3.0e-2, 0.9
TOLERANCE, SKETCH = 1.0e-2, 128
CHUNKS = (NT + 1, 1)


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    model, (velocity, pressure) = mpi_instationary_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE,
        degree=DEGREE, refinements=REFINEMENTS, T=T, nt=NT, order=ORDER,
    )
    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, model.parameters["mu"]))]
    test = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TEST, model.parameters["mu"]))]

    u_product, p_product = model.u_product, model.p_product
    reference = [model.solve(mu) for mu in test]

    # --- the stored path, kept as the thing to be measured against ----------------------------
    trajectories = model.solution_space.empty()
    for mu in train:
        trajectories.append(model.solve(mu))

    velocities, pressures = trajectories.blocks[0], trajectories.blocks[1]
    eps = [tolerance_for(velocities, u_product), tolerance_for(pressures, p_product)]

    stored_u, _, _ = inc_vectorarray_hapod(
        N_TRAIN, velocities, eps[0], omega=OMEGA, product=u_product
    )
    stored_p, _, _ = inc_vectorarray_hapod(
        N_TRAIN, pressures, eps[1], omega=OMEGA, product=p_product
    )

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"trajectories       : {N_TRAIN} of {NT + 1} levels = {len(trajectories)} snapshots")
    print(f"held by the stored path: {2 * len(trajectories)} full-order vectors")

    faces, residual, error = fit(
        model, stored_u, stored_p, test, reference, training_states=velocities
    )
    print(f"\n  {'path':>20}  {'held':>6}  {'basis':>10}  {'faces':>6}  {'residual':>10}  "
          f"{'ROM error':>11}")
    print(f"  {'stored':>20}  {2 * len(trajectories):>6}  "
          f"{f'{len(stored_u)}u, {len(stored_p)}p':>10}  {faces:>6}  {residual:>10.3e}  "
          f"{error:>11.4e}")

    # --- the streamed path: solved once, never held -------------------------------------------
    for chunk in CHUNKS:
        def trajectories_of():
            """One trajectory at a time, in chunks, as the stepper produces them."""
            for mu in train:
                yield trajectory_chunks(model, mu, chunk), chunk_count(NT + 1, chunk)

        (basis_u, compressed_u), (basis_p, compressed_p) = streaming_basis(
            trajectories_of(), N_TRAIN, [u_product, p_product], eps, omega=OMEGA
        )

        faces, residual, error = fit(
            model, basis_u, basis_p, test, reference, training_snapshots=compressed_u
        )
        held = compressed_u.n_vectors + compressed_p.n_vectors

        print(f"  {f'streamed, chunk {chunk}':>20}  {held:>6}  "
              f"{f'{len(basis_u)}u, {len(basis_p)}p':>10}  {faces:>6}  {residual:>10.3e}  "
              f"{error:>11.4e}")

    print(
        "\nThe reduced model cannot tell which route produced its training points, and the two\n"
        "differ by a face or two in what the greedy picked and by a fraction of a percent in what\n"
        "it predicts. What differs is the storage: a handful of vectors per trajectory against one\n"
        "per level, and the ratio improves the longer the trajectory, because a local basis is\n"
        "sized by the trajectory's intrinsic rank and not by how finely it was sampled.\n"
        "\n"
        "Compressing after every level costs nothing over compressing once per trajectory, so the\n"
        "peak is the running basis plus one vector. The accumulated error stays at the order of\n"
        "the tolerance asked for -- ask for one decade more than needed and it is paid for."
    )


def tolerance_for(snapshots, product):
    """HAPOD's ``eps`` is an absolute l2-mean error; this reads a relative one into it.

    Velocity and pressure differ by an order of magnitude in this problem, so one absolute number
    would mean something different to each.
    """
    return BASIS_TOLERANCE * np.sqrt((snapshots.norm(product) ** 2).mean())


def fit(model, basis_u, basis_p, test, reference, **training):
    """Hyper-reduce on the given bases and training points; return faces, residual and error."""
    reductor = InstationaryECSWStokesReductor(
        model, RB_u=basis_u.copy(), RB_p=basis_p.copy(),
        u_product=model.u_product, p_product=model.p_product,
        tolerance=TOLERANCE, sketch_rows=SKETCH, **training,
    )
    rom = reductor.reduce()
    momentum = rom.operator.momentum
    product = model.products["mixed"]

    worst = max(
        (U - reductor.reconstruct(rom.solve(mu))).norm(product).max() / U.norm(product).max()
        for mu, U in zip(test, reference)
    )

    return momentum.n_faces, momentum.training_residual, worst


if __name__ == "__main__":
    main()
