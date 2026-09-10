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

"""A shedding wake, reduced across Reynolds numbers and trained without holding the snapshots.

The forced problem verifies the transient machinery on a flow that has no dynamics: it relaxes to
its steady state and stops. This one has a von Karman street, and a parameter that changes the
operator rather than the forcing.

**The parameter is the viscosity, and the Reynolds number is what it means.** Reynolds could be
varied through the inflow instead, but that would rescale the convective term along with the
boundary condition and leave the momentum operator affine in nothing. The viscosity is the
coefficient the operator *is* affine in -- the interior penalty parameter is geometric and every
viscous flux carries the viscosity as a factor -- so a Reynolds sweep is an affine sweep, and the
reduced model assembles its momentum block online from two small dense matrices.

That is also the difference from every other example here. A parameter living in the right-hand
side never reaches ExaDG: the model holds the affine components and combines them itself. A
parameter of the operator has to go the other way, because the residual, the Jacobian and the
solve all belong to the application. It is installed before each of them, and recovered for the
reduced model by probing rather than by naming terms.

An inflow is imposed weakly, and that changes two things every other example here could ignore.

**The convective term is quadratic plus affine, not quadratic.** The prescribed value is carried
into the convective flux, so the polynomial half has a linear and a constant part that a plain
polarisation would fold into the tensor. Both are collected into the affine block instead, where
they belong.

**The continuity equation has a right-hand side.** The same boundary term appears in the
divergence operator, so the constraint is ``B u = g`` and not ``B u = 0``. ExaDG assembles it
inside its own solve, which is why a full-order model never has to be told -- and why a projected
one does: its pressure row is the projection of that equation, and nothing else supplies the
constant.

With homogeneous data both vanish and this reduces to the forced case.

Offline is one full-order pass. Each trajectory is compressed to a small basis and its levels'
coefficients as it is computed, then discarded; the global basis is a hierarchical POD of those
local ones; and ECSW is fitted at coefficients carried across to it, never at a snapshot.

The step count follows the mesh through ExaDG's own Courant criterion, so refining does not
silently change the time discretisation::

    python python/examples/cylinder_transient_rom.py
    python python/examples/cylinder_transient_rom.py --vtu     # modes, faces, trajectories

Runs unchanged on any number of ranks::

    mpirun -n 4 python -m pymor.tools.mpi python/examples/cylinder_transient_rom.py

Run from the repository root.
"""

import sys
import time

import numpy as np
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.basis import chunk_count, streaming_basis, trajectory_chunks
from exadg.mor.models.instationary_saddle_point import (
    BDFTimeStepper,
    mpi_instationary_saddle_point_model,
    steps_for_cfl,
    time_step_for_cfl,
)
from exadg.mor.models.saddle_point import exadg_model, exadg_models_id
from exadg.mor.reductors import InstationaryECSWStokesReductor

INPUT_FILE = "applications/incompressible_navier_stokes/flow_past_cylinder/input_rom.json"
DEGREE, REFINEMENTS = 2, 0
T, ORDER = 8.0, 2

#: Courant number the step count is derived from, on whatever mesh is used.
CFL = 8.0

#: Reynolds numbers to train and test at. All above the onset of shedding, so that every
#: trajectory is a limit cycle and the basis is not asked to span two regimes.
TRAIN_REYNOLDS = (80.0, 110.0, 140.0, 170.0)
TEST_REYNOLDS = (95.0, 155.0)

#: Relative l2-mean projection error the basis is built to, and HAPOD's balance parameter.
BASIS_TOLERANCE, OMEGA = 3.0e-3, 0.9

#: Relative residual at which the ECSW fit stops, and the rows kept of its training matrix.
TOLERANCE, SKETCH = 1.0e-1, 128

#: Levels compressed at a time. One keeps the peak at the running basis plus a single vector.
CHUNK = 1

OUTPUT = "output/pymor/cylinder_transient_rom"


def main(write_vtu=False):
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    # Built at one step so it can be asked what step it admits, then restepped. The criterion is a
    # property of the discretisation, so the model has to exist before it can be read off.
    model, (velocity, pressure) = mpi_instationary_saddle_point_model(
        "flow_past_cylinder", "CylinderFOM2D", INPUT_FILE,
        degree=DEGREE, refinements=REFINEMENTS, T=T, nt=1, order=ORDER,
    )
    nt = steps_for_cfl(model, CFL, T)
    model = model.with_(
        time_stepper=BDFTimeStepper(nt, order=ORDER, solver=model.time_stepper.solver)
    )
    scale = reynolds_scale(model)
    train = [Mu(viscosity=scale / re) for re in TRAIN_REYNOLDS]
    test = [Mu(viscosity=scale / re) for re in TEST_REYNOLDS]

    print(f"ranks              : {mpi.size}")
    print(f"dofs               : {velocity.dim} velocity, {pressure.dim} pressure")
    print(f"trajectory         : {nt} steps of BDF-{ORDER} over [0, {T}], "
          f"dt = {T / nt:.4e} at CFL {CFL} (limit {time_step_for_cfl(model, 1.0):.4e})")
    print(f"parameter          : {model.parameters}, i.e. Reynolds "
          f"{min(TRAIN_REYNOLDS):.0f} to {max(TRAIN_REYNOLDS):.0f}")
    print(f"training           : {len(train)} trajectories = {len(train) * (nt + 1)} levels")
    print(f"tolerances         : basis {BASIS_TOLERANCE:.0e}, ECSW {TOLERANCE:.0e}, "
          f"sketch {SKETCH}, chunk {CHUNK}")

    timings = {}

    started = time.perf_counter()
    (basis_u, snapshots_u), (basis_p, snapshots_p) = stream(model, train, CHUNK)
    timings["basis (one FOM pass)"] = time.perf_counter() - started

    started = time.perf_counter()
    reductor = InstationaryECSWStokesReductor(
        model, RB_u=basis_u.copy(), RB_p=basis_p.copy(),
        u_product=model.u_product, p_product=model.p_product,
        tolerance=TOLERANCE, sketch_rows=SKETCH, training_snapshots=snapshots_u,
    )
    rom = reductor.reduce()
    timings["reduce (total)"] = time.perf_counter() - started

    momentum = rom.operator.momentum
    timings["  ECSW data (face loops)"] = momentum.assembly_seconds
    timings["  ECSW fit (NNLS)"] = momentum.nnls_seconds

    print(f"\nbasis              : {len(basis_u)} velocity + {len(basis_p)} pressure modes, "
          f"reduced dimension {rom.solution_space.dim}")
    print(f"held               : {snapshots_u.n_vectors + snapshots_p.n_vectors} full-order "
          f"vectors, against {2 * len(train) * (nt + 1)} for keeping every level")
    print(f"faces              : {momentum.n_faces} of {momentum.n_candidates}, "
          f"fit residual {momentum.training_residual:.3e}")

    started = time.perf_counter()
    reference = [model.solve(mu) for mu in test]
    timings["FOM trajectory"] = (time.perf_counter() - started) / len(test)

    started = time.perf_counter()
    reduced = [rom.solve(mu) for mu in test]
    timings["ROM trajectory"] = (time.perf_counter() - started) / len(test)

    report(timings)

    # The projection error is the floor: it is what the basis can do with the trajectory in hand,
    # so a reduced error near it is the basis's fault and one far above it is the dynamics'.
    product = model.products["mixed"]
    print(f"\n  {'Reynolds':>9}  {'worst level':>12}  {'at t':>6}  {'projection':>11}  "
          f"{'final level':>12}")
    for reynolds, U, coefficients in zip(TEST_REYNOLDS, reference, reduced):
        # Relative at each level, not to the trajectory's largest. An impulsive start puts a
        # pressure spike two orders above the developed flow in the first step, and dividing the
        # whole trajectory by that reports a reduced model as accurate on the strength of a
        # transient it never has to reproduce.
        level_norm = U.norm(product)
        level_norm[0] = 1.0                  # at rest, and the error there is exactly zero
        error = (U - reductor.reconstruct(coefficients)).norm(product) / level_norm
        floor = (U - project_onto(reductor, U, product)).norm(product) / level_norm
        worst = int(np.argmax(error))
        print(f"  {reynolds:>9.0f}  {error.max():>12.4e}  {worst * T / (len(U) - 1):>6.2f}  "
              f"{floor[worst]:>11.4e}  {error[-1]:>12.4e}")

    if write_vtu:
        visualise(model, reductor, momentum, basis_u, basis_p, reference[0], reduced[0])

    print(
        "\nThe Reynolds numbers tested were not trained at, and the reduced model reached them\n"
        "without touching ExaDG: its momentum block is assembled from a base and a slope, which is\n"
        "what insisting on affinity in the coefficient buys. The offline phase held a basis per\n"
        "trajectory rather than every level, and the fit never saw a snapshot."
    )


def _local_reynolds_scale(model):
    fom = exadg_model(model)

    return fom.viscosity * fom.reynolds_number


def project_onto(reductor, U, product):
    """The best the basis can do: the orthogonal projection of a trajectory onto it.

    The block basis is orthonormal in the mixed product -- the velocity half in the mass product
    it was enriched and orthonormalised in, the pressure half in its own -- so the projection is
    the inner products, with no mass matrix to invert.
    """
    basis = reductor._block_basis

    return basis.lincomb(basis.inner(U, product=product))


def reynolds_scale(model):
    """``nu * Re``, which is fixed: the mean inflow times the cylinder diameter.

    Read off the application at the value it was built with, so the conversion is the benchmark's
    own definition rather than a second copy of it. Dispatched because under MPI the model is a
    wrapper and the application lives on the ranks.
    """
    if not mpi.parallel:
        return _local_reynolds_scale(model)

    return mpi.call(mpi.function_call, _local_reynolds_scale, exadg_models_id(model))


def stream(model, parameters, chunk):
    """The offline phase: solve once, compress as it goes, keep no snapshot.

    Returns one ``(basis, compressed)`` pair per field.
    """
    levels = model.time_stepper.nt + 1
    scales = trajectory_scales(model, parameters[0], chunk)

    def trajectories():
        for mu in parameters:
            yield trajectory_chunks(model, mu, chunk), chunk_count(levels, chunk)

    return streaming_basis(
        trajectories(), len(parameters),
        [model.u_product, model.p_product],
        [BASIS_TOLERANCE * scale for scale in scales],
        omega=OMEGA,
    )


def trajectory_scales(model, mu, chunk):
    """The rms level norm of each field, so a relative tolerance can be read into HAPOD's eps.

    ``eps`` is an absolute l2-mean error and the two fields differ in magnitude, so one absolute
    number would mean something different to each. Costs one extra trajectory and holds nothing.
    """
    products = (model.u_product, model.p_product)
    totals, levels = np.zeros(len(products)), 0

    for blocks in trajectory_chunks(model, mu, chunk):
        levels += len(blocks[0])
        for field, (block, product) in enumerate(zip(blocks, products)):
            totals[field] += (block.norm(product) ** 2).sum()

    return np.sqrt(totals / levels)


def visualise(model, reductor, momentum, basis_u, basis_p, reference, reduced):
    """Everything worth opening in ParaView: the modes, the sampled faces, and the trajectories.

    Colour a velocity mode by a *component*. Modes are orthogonal as vector fields, but their
    magnitudes are correlated and their norms nearly equal, so ParaView's default for a vector
    array makes them all look like the same picture.
    """
    written = []

    modes = max(len(basis_u), len(basis_p))
    velocity, pressure = model.solution_space.subspaces
    fields = [
        model.solution_space.make_array([
            basis_u[k] if k < len(basis_u) else velocity.zeros(1),
            basis_p[k] if k < len(basis_p) else pressure.zeros(1),
        ])
        for k in range(modes)
    ]
    written += list(model.visualize(
        fields, legend=[f"mode_{k}" for k in range(modes)], filename=f"{OUTPUT}_modes"
    ))
    written.append(momentum.write_selection(f"{OUTPUT}_faces"))

    approximation = reductor.reconstruct(reduced)
    written += list(model.visualize(
        (reference, approximation, reference - approximation),
        legend=("fom", "rom", "error"), filename=OUTPUT,
        times=np.linspace(0.0, T, len(reference)),
    ))

    print("\nwrote " + "\n      ".join(written))


def report(timings):
    """Seconds per stage, in the order they run."""
    print(f"\n  {'stage':<24}  {'seconds':>9}")
    for name, seconds in timings.items():
        print(f"  {name:<24}  {seconds:>9.3f}")


if __name__ == "__main__":
    main(write_vtu="--vtu" in sys.argv)
