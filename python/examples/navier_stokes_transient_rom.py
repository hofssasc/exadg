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

"""A hyper-reduced transient Navier-Stokes model, trained without holding the snapshots.

One full-order pass. Each trajectory is compressed to a small basis and its levels' coefficients
as it is computed, then discarded; the global basis is a hierarchical POD of those local ones; and
ECSW is fitted at coefficients carried across to it, never at a snapshot. Peak storage is one
chunk plus the running bases.

The reduced model is the steady one plus a mass term -- an exact convective tensor, a projected
viscous block, and the Lax-Friedrichs stabilisation on a few faces -- stepped by the full-order
model's own BDF scheme with only the step solver swapped, so the two are one discretisation and
the error below is the reduction alone.

Three tolerances, controlling different things:

    BASIS_TOLERANCE   how well the basis represents the training trajectories
    TOLERANCE         how well the sampled faces reproduce the projected stabilisation
    SKETCH            rows kept of the ECSW training matrix, which grows with the level count

**The step count follows the mesh.** ``CFL`` is fixed and the count is derived from it through
ExaDG's own criterion, read off the element sizes. Fixing the count instead would mean that
refining silently changes the Courant number, and then a refinement study varies two things at
once.

A default run streams and reports what each stage cost and how far the reduced trajectories are
from the full-order ones. Three additions are optional and off by default; the two comparisons
each need the stored path as their reference, which is the expense streaming exists to avoid, so
neither belongs in a run on a case that is hard to solve::

    python python/examples/navier_stokes_transient_rom.py
    python python/examples/navier_stokes_transient_rom.py --vtu       # modes, faces, trajectories
    python python/examples/navier_stokes_transient_rom.py --chunks    # compression granularity
    python python/examples/navier_stokes_transient_rom.py --sketches  # sketch size

Runs unchanged on any number of ranks::

    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_transient_rom.py

Run from the repository root.
"""

import sys
import time

import numpy as np
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.basis import chunk_count, streaming_basis, trajectory_chunks
from exadg.mor.binding import output_levels
from exadg.mor.models.instationary_saddle_point import (
    BDFTimeStepper,
    mpi_instationary_saddle_point_model,
    steps_for_cfl,
    time_step_for_cfl,
)
from exadg.mor.reductors import InstationaryECSWStokesReductor

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes_transient.json"
DEGREE, REFINEMENTS = 2, 6
T, ORDER = 4.0, 2

#: Courant number the step count is derived from, on whatever mesh is used.
CFL = 0.5

N_TRAIN, N_TEST = 6, 2
AMPLITUDES = (0.5, 1.5)

#: Relative l2-mean projection error the basis is built to, and HAPOD's balance parameter.
BASIS_TOLERANCE, OMEGA = 1.0e-3, 0.9

#: Relative residual at which the ECSW fit stops, and the rows kept of its training matrix.
TOLERANCE, SKETCH = 1.0e-1, 128

#: Levels compressed at a time. One keeps the peak at the running basis plus a single vector.
CHUNK = 1

#: Time between the records --vtu writes, independent of the step the CFL number sets: a fine mesh
#: takes thousands of steps and cannot afford a record for each. None writes every level.
VTU_INTERVAL = 0.05

OUTPUT = "output/pymor/navier_stokes_transient_rom"


def main(compare_chunks=False, compare_sketches=False, write_vtu=False):
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    # Built at one step so it can be asked what step it admits, then restepped. The criterion is a
    # property of the discretisation, so the model has to exist before it can be read off.
    model, (velocity, pressure) = mpi_instationary_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE,
        degree=DEGREE, refinements=REFINEMENTS, T=T, nt=1, order=ORDER,
    )
    nt = steps_for_cfl(model, CFL, T)
    model = model.with_(
        time_stepper=BDFTimeStepper(nt, order=ORDER, solver=model.time_stepper.solver)
    )
    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, model.parameters["mu"]))]
    test = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TEST, model.parameters["mu"]))]

    print(f"ranks              : {mpi.size}")
    print(f"dofs               : {velocity.dim} velocity, {pressure.dim} pressure")
    print(f"trajectory         : {nt} steps of BDF-{ORDER} over [0, {T}], "
          f"dt = {T / nt:.4e} at CFL {CFL} (limit {time_step_for_cfl(model, 1.0):.4e})")
    print(f"training           : {N_TRAIN} trajectories = {N_TRAIN * (nt + 1)} levels")
    print(f"tolerances         : basis {BASIS_TOLERANCE:.0e}, ECSW {TOLERANCE:.0e}, "
          f"sketch {SKETCH}, chunk {CHUNK}")

    timings = {}

    started = time.perf_counter()
    fields = stream(model, train, CHUNK)
    timings["basis (one FOM pass)"] = time.perf_counter() - started

    (basis_u, snapshots_u), (basis_p, snapshots_p) = fields
    held = snapshots_u.n_vectors + snapshots_p.n_vectors

    started = time.perf_counter()
    reductor, rom = hyper_reduce(model, basis_u, basis_p, training_snapshots=snapshots_u)
    timings["reduce (total)"] = time.perf_counter() - started

    momentum = rom.operator.momentum
    timings["  ECSW data (face loops)"] = momentum.assembly_seconds
    timings["  ECSW fit (NNLS)"] = momentum.nnls_seconds

    print(f"\nbasis              : {len(basis_u)} velocity + {len(basis_p)} pressure modes, "
          f"reduced dimension {rom.solution_space.dim}")
    print(f"held               : {held} full-order vectors, against "
          f"{2 * N_TRAIN * (nt + 1)} for keeping every level")
    print(f"faces              : {momentum.n_faces} of {momentum.n_candidates}, "
          f"fit residual {momentum.training_residual:.3e}")

    started = time.perf_counter()
    reference = [model.solve(mu) for mu in test]
    timings["FOM trajectory"] = (time.perf_counter() - started) / N_TEST

    started = time.perf_counter()
    reduced = [rom.solve(mu) for mu in test]
    timings["ROM trajectory"] = (time.perf_counter() - started) / N_TEST

    report(timings)

    product = model.products["mixed"]
    print(f"\n  {'test':>6}  {'worst level':>12}  {'final level':>12}")
    worst = 0.0
    for index, (U, coefficients) in enumerate(zip(reference, reduced)):
        error = (U - reductor.reconstruct(coefficients)).norm(product) / U.norm(product).max()
        worst = max(worst, error.max())
        print(f"  {index:>6}  {error.max():>12.4e}  {error[-1]:>12.4e}")
    print(f"  {'worst':>6}  {worst:>12.4e}")

    if write_vtu:
        visualise(model, reductor, momentum, basis_u, basis_p, reference[0], reduced[0])

    if compare_chunks:
        chunk_comparison(model, train, test, reference)
    if compare_sketches:
        sketch_comparison(model, train, test, reference, fields)

    print(
        "\nThe reduced model is stepped by the full-order model's own scheme, so the error above is\n"
        "the reduction and not a difference of time discretisations. The offline phase held a basis\n"
        "per trajectory rather than every level, and the fit never saw a snapshot: ECSW trains at\n"
        "coefficients, and coefficients survive a change of basis."
    )


def visualise(model, reductor, momentum, basis_u, basis_p, reference, reduced):
    """Everything worth opening in ParaView, for the streamed model only.

    Three records, and they answer different questions. The **modes** are what the basis spans;
    the **faces** are where ECSW put its quadrature, drawn as the faces themselves rather than the
    cells beside them; and the **trajectories** are the full-order one, the reduced one and their
    difference as one animated series per block.

    Off by default because a run on a case that is hard to solve wants numbers, not files.
    """
    written = []

    # One field per mode. The two bases need not be the same length -- the pressure usually needs
    # fewer -- so the shorter is padded rather than the extra modes dropped.
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

    # One record per VTU_INTERVAL, not per step. The levels are selected before reconstructing: a
    # view of the reference costs nothing, a reconstructed trajectory is as large as the reference.
    times = np.linspace(0.0, T, len(reference))
    levels = output_levels(times, VTU_INTERVAL)
    reference = reference[levels]
    approximation = reductor.reconstruct(reduced[levels])
    written += list(model.visualize(
        (reference, approximation, reference - approximation),
        legend=("fom", "rom", "error"), filename=OUTPUT, times=times[levels],
    ))

    print("\nwrote " + "\n      ".join(written))


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

    ``eps`` is an absolute l2-mean error, and velocity and pressure differ by an order of magnitude
    here, so one absolute number would mean something different to each.

    Costs one extra trajectory -- a relative tolerance needs a scale and a scale needs data -- but
    holds nothing: the chunks are streamed and only the running sums of squares are kept. One
    trajectory in ``N_TRAIN + 1`` is the price of not having to guess an absolute number.
    """
    products = (model.u_product, model.p_product)
    totals, levels = np.zeros(len(products)), 0

    for blocks in trajectory_chunks(model, mu, chunk):
        levels += len(blocks[0])
        for field, (block, product) in enumerate(zip(blocks, products)):
            totals[field] += (block.norm(product) ** 2).sum()

    return np.sqrt(totals / levels)


def hyper_reduce(model, basis_u, basis_p, sketch=SKETCH, **training):
    """Hyper-reduce onto the given bases; returns the reductor and the reduced model."""
    reductor = InstationaryECSWStokesReductor(
        model, RB_u=basis_u.copy(), RB_p=basis_p.copy(),
        u_product=model.u_product, p_product=model.p_product,
        tolerance=TOLERANCE, sketch_rows=sketch, **training,
    )

    return reductor, reductor.reduce()


def worst_error(model, reductor, rom, test, reference):
    """Largest relative error over the test trajectories and over time."""
    product = model.products["mixed"]

    return max(
        (U - reductor.reconstruct(rom.solve(mu))).norm(product).max() / U.norm(product).max()
        for mu, U in zip(test, reference)
    )


def stored(model, parameters):
    """Every level of every training trajectory, kept, and the basis built from all of them.

    The reference the comparisons measure against, and the thing streaming exists to avoid.
    """
    trajectories = model.solution_space.empty()
    for mu in parameters:
        trajectories.append(model.solve(mu))

    bases = []
    for block, product in ((0, model.u_product), (1, model.p_product)):
        snapshots = trajectories.blocks[block]
        eps = BASIS_TOLERANCE * np.sqrt((snapshots.norm(product) ** 2).mean())
        modes, _, _ = inc_vectorarray_hapod(
            len(parameters), snapshots, eps, omega=OMEGA, product=product
        )
        bases.append(modes)

    return trajectories, bases


def chunk_comparison(model, train, test, reference):
    """How the compression granularity moves the fit, against keeping every level."""
    print("\ncompression granularity, against keeping every level")
    print(f"  {'path':>20}  {'held':>6}  {'basis':>10}  {'faces':>6}  {'residual':>10}  "
          f"{'ROM error':>11}")

    trajectories, (basis_u, basis_p) = stored(model, train)
    reductor, rom = hyper_reduce(model, basis_u, basis_p, training_states=trajectories.blocks[0])
    momentum = rom.operator.momentum

    print(f"  {'stored':>20}  {2 * len(trajectories):>6}  "
          f"{f'{len(basis_u)}u, {len(basis_p)}p':>10}  {momentum.n_faces:>6}  "
          f"{momentum.training_residual:>10.3e}  "
          f"{worst_error(model, reductor, rom, test, reference):>11.4e}")

    for chunk in (model.time_stepper.nt + 1, 8, 1):
        fields = stream(model, train, chunk)
        (basis_u, snapshots_u), (basis_p, snapshots_p) = fields

        reductor, rom = hyper_reduce(model, basis_u, basis_p, training_snapshots=snapshots_u)
        momentum = rom.operator.momentum

        print(f"  {f'streamed, chunk {chunk}':>20}  "
              f"{snapshots_u.n_vectors + snapshots_p.n_vectors:>6}  "
              f"{f'{len(basis_u)}u, {len(basis_p)}p':>10}  {momentum.n_faces:>6}  "
              f"{momentum.training_residual:>10.3e}  "
              f"{worst_error(model, reductor, rom, test, reference):>11.4e}")


def sketch_comparison(model, train, test, reference, fields):
    """What the row sketch of the ECSW training matrix costs, against fitting on it whole.

    The residual is measured on a second, independent sketch: NNLS minimises over the one it is
    given, so a fit's residual on its own sketch is in-sample and biased low.
    """
    print("\nrow sketch of the ECSW training matrix, against fitting on it whole")
    print(f"  {'sketch':>10}  {'rows':>6}  {'faces':>6}  {'residual':>10}  {'ROM error':>11}  "
          f"{'MB':>7}")

    (basis_u, snapshots_u), (basis_p, _) = fields

    for sketch in (None, 128, 32):
        reductor, rom = hyper_reduce(
            model, basis_u, basis_p, sketch=sketch, training_snapshots=snapshots_u
        )
        momentum = rom.operator.momentum
        rows = len(snapshots_u) * len(momentum.basis)
        megabytes = (rows if sketch is None else sketch) * momentum.n_candidates * 8 / 2**20

        print(f"  {'exact' if sketch is None else sketch:>10}  {rows:>6}  {momentum.n_faces:>6}  "
              f"{momentum.training_residual:>10.3e}  "
              f"{worst_error(model, reductor, rom, test, reference):>11.4e}  {megabytes:>7.2f}")


def report(timings):
    """Seconds per stage. An indented entry is part of the one above it."""
    print(f"\n  {'stage':<26}  {'seconds':>9}")
    for stage, seconds in timings.items():
        print(f"  {stage:<26}  {seconds:>9.3f}")

    print(f"  {'FOM / ROM per trajectory':<26}  "
          f"{timings['FOM trajectory'] / timings['ROM trajectory']:>8.1f}x")


if __name__ == "__main__":
    main(
        compare_chunks="--chunks" in sys.argv,
        compare_sketches="--sketches" in sys.argv,
        write_vtu="--vtu" in sys.argv,
    )
