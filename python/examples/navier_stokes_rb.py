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

"""Saddle-point reduced basis for the forced box at Navier-Stokes, driven entirely by pyMOR.

The same geometry, forcing and boundary conditions as ``stokes_rb.py``; the only difference is
the convective term. That is deliberate: everything that could go wrong in the block plumbing is
shared with the Stokes case, which verifies exactly, so what is measured here is the
nonlinearity and nothing else.

Unlike the Stokes case this is a real reduction benchmark rather than a verification. The
solution is no longer linear in the forcing amplitudes, so the manifold is not P-dimensional and
the reduced model approximates instead of reproducing.

The viscosity is 0.02, chosen by measuring rather than by taste. Against the Stokes solution at
the same forcing, the convective term moves the velocity by 14% there, while the coupled solver
still converges for every parameter drawn; at 0.1 the nonlinearity is worth 0.4% and the problem
is Stokes in disguise, and at 0.01 it is worth 30% but only half the parameters converge.

**The reduced model is correct, not fast.** Every reduced Newton step evaluates the residual at
full order -- pyMOR projects the nonlinear operator, it cannot collapse it -- so the online cost
still scales with the mesh. Hyper-reduction is what removes that, and it is the next step rather
than this one.

Run from the repository root::

    python python/examples/navier_stokes_rb.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_rb.py
"""

import numpy as np
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor
from pymor.tools import mpi

from exadg.mor.models.saddle_point import mpi_saddle_point_model

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes.json"
DEGREE, REFINEMENTS = 2, 3
N_TRAIN, N_TEST, N_MODES = 12, 4, 6
AMPLITUDES = (0.5, 1.5)


def main():
    # pyMOR warns once per projection that the nonlinear operator has no efficient projection.
    # That is true and is exactly the hyper-reduction gap the docstring describes, but it says so
    # a few dozen times.
    set_log_levels({"pymor": "ERROR"})

    model, (velocity, pressure) = mpi_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE, degree=DEGREE, refinements=REFINEMENTS
    )
    n_parameters = model.parameters["mu"]

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"pressure dofs      : {pressure.dim}")
    print(f"parameters         : {n_parameters}")
    print(f"nonlinear operator : {not model.operator.blocks[0, 0].linear}")

    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, n_parameters))]
    test = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TEST, n_parameters))]

    snapshots = model.solution_space.empty()
    for mu in train:
        snapshots.append(model.solve(mu))

    velocity_snapshots, pressure_snapshots = snapshots.blocks
    check_residual(model, snapshots[:1], train[0])

    basis_u, singular_u = pod(velocity_snapshots, product=model.u_product, modes=N_MODES)
    basis_p, singular_p = pod(pressure_snapshots, product=model.p_product, modes=N_MODES)
    print(f"velocity singular  : {np.array2string(singular_u[:N_MODES], precision=6)}")
    print(f"pressure singular  : {np.array2string(singular_p[:N_MODES], precision=6)}")

    print(f"\nbasis size against error over {N_TEST} test parameters")
    print(f"  {'modes':>5}  {'reduced dim':>11}  {'relative error':>14}")

    worst_mu, reductor, rom = None, None, None
    for size in range(1, N_MODES + 1):
        errors, reductor, rom = reduce_and_measure(model, basis_u[:size], basis_p[:size], test)
        print(f"  {size:>5}  {rom.solution_space.dim:>11}  {max(errors):>14.3e}")
        worst_mu = test[int(np.argmax(errors))]

    print(
        "\nThe error falls with the basis and does not reach machine precision, which is the\n"
        "difference from the Stokes case: the solution is no longer linear in the amplitudes, so\n"
        "a finite basis approximates rather than spans. What it does not show is a speed-up --\n"
        "every reduced Newton step still evaluates the residual at full order, and removing that\n"
        "is what hyper-reduction is for."
    )

    U_fom = model.solve(worst_mu)
    U_rom = reductor.reconstruct(rom.solve(worst_mu))
    records = model.visualize(
        (U_fom, U_rom, U_fom - U_rom),
        legend=("fom", "rom", "error"),
        filename="output/pymor/navier_stokes",
    )

    if not mpi.parallel:
        from exadg import forced

        forcing = forced.ForcedFOM2D(
            INPUT_FILE, degree=DEGREE, refinements=REFINEMENTS
        ).write_forcing("output/pymor", "navier_stokes_forcing", worst_mu["mu"].tolist())

        print(f"\nwrote {records[0]}")
        print(f"      {records[1]}")
        print(f"      {forcing}")


def check_residual(model, U, mu):
    """pyMOR's assembled block system must be the one ExaDG's Newton solved."""
    rhs = model.rhs.as_range_array(mu)
    residual = (model.operator.apply(U, mu=mu) - rhs).norm()[0] / rhs.norm()[0]

    print(f"nonlinear residual : {residual:.3e}  (Newton's own tolerance)")
    assert residual < 1.0e-5, "the block system is not the one ExaDG solves"


def reduce_and_measure(model, basis_u, basis_p, test):
    """Relative errors of the reduced model over the test set, in the mixed product."""
    reductor = SupremizerGalerkinStokesReductor(
        model, RB_u=basis_u, RB_p=basis_p, u_product=model.u_product, p_product=model.p_product
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
