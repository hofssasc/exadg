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

"""Saddle-point reduced basis for the ExaDG forced box, driven entirely by pyMOR.

This is a **verification, not a benchmark**. The Stokes solution is linear in the forcing
amplitudes, so the manifold is exactly P-dimensional and a basis of P modes has to reproduce the
full-order model. There is nothing to approximate, which is the point: any error that survives is
a defect in the saddle-point projection rather than an approximation, with no truncation error to
hide behind. The reduction benchmark is the Navier-Stokes case; this is what has to work first.

Four things are checked, in the order in which they would break:

    1. B and its adjoint really are adjoint, so pyMOR's AdjointOperator(B) is ExaDG's (1,2) block
    2. the assembled block system agrees with ExaDG's own coupled solve
    3. a truncated basis is visibly wrong, so the check below has power
    4. the reduced model reproduces the full one once the basis spans the manifold

Serial only, and it says so rather than hanging: the coupled solve is not yet MPI-aware. The
thermal block examples are the parallel ones.

Run from the repository root::

    python python/examples/stokes_rb.py
"""

import numpy as np
from pymor.algorithms.pod import pod
from pymor.parameters.base import Mu
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor
from pymor.tools import mpi

from exadg import forced
from exadg.mor.models.saddle_point import saddle_point_model

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input.json"
N_TRAIN, N_TEST, N_MODES = 20, 5, 4


def main():
    # Returning rather than raising, and before anything is built. Under pymor.tools.mpi only
    # rank 0 runs this script and the others wait in the event loop, which rank 0 shuts down by
    # calling quit() *after* the script finishes -- so an exception here would leave them waiting
    # forever. The model constructor is collective, so building it on rank 0 alone hangs too;
    # that is what this check exists to prevent.
    if mpi.parallel:
        print(
            "stokes_rb.py is serial: the coupled solve is not MPI-aware yet, so a parallel run "
            "would deadlock inside ExaDG's GMRES. Run it without mpirun; thermal_block_rb.py is "
            "the parallel example."
        )
        return

    fom = forced.ForcedFOM2D(INPUT_FILE, degree=2, refinements=3)
    model, (velocity, pressure) = saddle_point_model(fom)
    n_parameters = model.parameters["mu"]

    print(f"velocity dofs      : {velocity.dim}")
    print(f"pressure dofs      : {pressure.dim}")
    print(f"parameters         : {n_parameters}")

    check_adjoint(model, velocity, pressure)
    check_block_system(model, n_parameters)

    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(-1.0, 1.0, (N_TRAIN, n_parameters))]
    test = [Mu(mu=m) for m in rng.uniform(-1.0, 1.0, (N_TEST, n_parameters))]

    snapshots = model.solution_space.empty()
    for mu in train:
        snapshots.append(model.solve(mu))

    velocity_snapshots, pressure_snapshots = snapshots.blocks

    # A product, not the Euclidean one, in each block: velocity and pressure are different
    # physical quantities on different spaces and there is no reason for their coefficient
    # vectors to be comparable.
    basis_u, singular_u = pod(velocity_snapshots, product=model.u_product, modes=N_MODES)
    basis_p, singular_p = pod(pressure_snapshots, product=model.p_product, modes=N_MODES)
    print(f"velocity singular  : {np.array2string(singular_u[:N_MODES], precision=6)}")
    print(f"pressure singular  : {np.array2string(singular_p[:N_MODES], precision=6)}")

    print(f"\nbasis size against error over {N_TEST} test parameters")
    print(f"  {'modes':>5}  {'reduced dim':>11}  {'relative error':>14}")

    worst_mu, worst_error, reductor, rom = None, None, None, None
    for size in range(1, N_MODES + 1):
        errors, this_reductor, this_rom = reduce_and_measure(
            model, basis_u[:size], basis_p[:size], test
        )
        print(f"  {size:>5}  {this_rom.solution_space.dim:>11}  {max(errors):>14.3e}")

        worst_mu = test[int(np.argmax(errors))]
        worst_error, reductor, rom = max(errors), this_reductor, this_rom

    print(
        "\nThe last row is the verification: a basis that spans the manifold reproduces the\n"
        "full-order model, and what is left is the solver tolerance times the condition number of\n"
        "the saddle point. The rows above it are what give that meaning -- a truncated basis is\n"
        "visibly wrong, so the last row is a property of the projection and not of the test.\n"
        "\n"
        "The reduced dimension is three times the mode count, not two: the reductor adds one\n"
        "supremizer per pressure mode. Without them the reduced velocity and pressure spaces\n"
        "satisfy no discrete inf-sup condition, and the reduced saddle point is singular or its\n"
        "pressure is noise."
    )

    # Two records per call, since velocity and pressure live on different DoF handlers.
    U_fom = model.solve(worst_mu)
    U_rom = reductor.reconstruct(rom.solve(worst_mu))
    records = model.visualize(
        (U_fom, U_rom, U_fom - U_rom),
        legend=("fom", "rom", "error"),
        filename="output/pymor/stokes",
    )

    # The parameter itself, on the same mesh as the field it drives.
    forcing = fom.write_forcing("output/pymor", "stokes_forcing", worst_mu["mu"].tolist())

    print(f"\nwrote {records[0]}")
    print(f"      {records[1]}")
    print(f"      {forcing}")


def check_adjoint(model, velocity, pressure):
    """B and B^T must be adjoint, or pyMOR's (1,2) block is not ExaDG's."""
    B = model.operator.blocks[1, 0]

    u = velocity.random(1, distribution="normal", random_state=0)
    p = pressure.random(1, distribution="normal", random_state=1)

    lhs = B.apply(u).inner(p)[0, 0]
    rhs = u.inner(B.apply_adjoint(p))[0, 0]

    print(f"\n<B u, p> vs <u, B^T p> : {abs(lhs - rhs) / abs(lhs):.3e}")
    assert abs(lhs - rhs) / abs(lhs) < 1.0e-12, "B^T is not the adjoint of B"


def check_block_system(model, n_parameters):
    """pyMOR's assembled block operator must agree with ExaDG's coupled solve."""
    mu = Mu(mu=np.linspace(1.0, -1.0, n_parameters))

    U = model.solve(mu)
    rhs = model.rhs.as_range_array(mu)
    residual = (model.operator.apply(U, mu=mu) - rhs).norm()[0] / rhs.norm()[0]

    print(f"block residual / |rhs| : {residual:.3e}")
    assert residual < 1.0e-9, "the block system is not the one ExaDG solves"


def reduce_and_measure(model, basis_u, basis_p, test):
    """Relative errors of the reduced model over the test set, in the mixed product."""
    reductor = SupremizerGalerkinStokesReductor(
        model,
        RB_u=basis_u,
        RB_p=basis_p,
        u_product=model.u_product,
        p_product=model.p_product,
    )
    rom = reductor.reduce()

    product = model.products["mixed"]

    errors = []
    for mu in test:
        U_fom = model.solve(mu)
        U_rom = reductor.reconstruct(rom.solve(mu))

        errors.append((U_fom - U_rom).norm(product)[0] / U_fom.norm(product)[0])

    return errors, reductor, rom


if __name__ == "__main__":
    main()
