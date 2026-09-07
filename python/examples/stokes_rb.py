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
amplitudes, so the P solutions at the unit amplitudes span the whole solution manifold and a
reduced model built on them has to reproduce the full-order model exactly. There is nothing to
approximate, which is the point: any error that survives is a defect in the saddle-point
projection rather than an approximation, and there is no truncation error to hide behind. The
reduction benchmark is the Navier-Stokes case; this is what has to work before it.

Four things are checked, in the order in which they would break:

    1. B and its adjoint really are adjoint, so pyMOR's AdjointOperator(B) is ExaDG's (1,2) block
    2. the assembled block system agrees with ExaDG's own coupled solve
    3. a truncated basis is visibly wrong, so the check below has power
    4. the reduced model reproduces the full one once the basis spans the manifold

Run from the repository root::

    python python/examples/stokes_rb.py
"""

import numpy as np
from pymor.algorithms.pod import pod
from pymor.parameters.base import Mu
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor

from exadg import forced
from exadg.mor.models.saddle_point import saddle_point_model

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input.json"


def main():
    fom = forced.ForcedFOM2D(INPUT_FILE, degree=2, refinements=3)
    model, (velocity, pressure) = saddle_point_model(fom)
    n_modes = fom.n_modes

    print(f"velocity dofs     : {velocity.dim}")
    print(f"pressure dofs     : {pressure.dim}")
    print(f"forcing modes     : {n_modes}")

    check_adjoint(model, velocity, pressure)
    check_block_system(model, n_modes)

    # The P solutions at the unit amplitudes. By linearity these span every solution, so this
    # is the whole manifold rather than a sample of it.
    snapshots = model.solution_space.empty()
    for i in range(n_modes):
        snapshots.append(model.solve(Mu(mu=np.eye(n_modes)[i])))

    velocity_snapshots, pressure_snapshots = snapshots.blocks

    rng = np.random.default_rng(0)
    test = [Mu(mu=m) for m in rng.uniform(-1.0, 1.0, (5, n_modes))]

    print(f"\nreduced model, basis size against error over {len(test)} test parameters")
    print(f"  {'modes':>5}  {'reduced dim':>11}  {'relative error':>14}")

    for size in range(1, n_modes + 1):
        error, dim = reduce_and_measure(
            model, velocity_snapshots[:size], pressure_snapshots[:size], test
        )
        print(f"  {size:>5}  {dim:>11}  {error:>14.3e}")

    print(
        "\nThe last row is the verification: a basis that spans the manifold reproduces the\n"
        "full-order model, and what is left is the solver tolerance times the condition number of\n"
        "the saddle point. The rows above it are what gives that meaning -- a truncated basis is\n"
        "visibly wrong, so the last row is a property of the projection and not of the test.\n"
        "\n"
        "The reduced dimension is larger than twice the number of modes because the reductor\n"
        "enriches the velocity space with one supremizer per pressure mode. Without them the\n"
        "reduced velocity and pressure spaces satisfy no discrete inf-sup condition, and the\n"
        "reduced saddle point is singular or its pressure is noise."
    )


def check_adjoint(model, velocity, pressure):
    """B and B^T must be adjoint, or pyMOR's (1,2) block is not ExaDG's."""
    B = model.operator.blocks[1, 0]

    u = velocity.random(1, distribution="normal", random_state=0)
    p = pressure.random(1, distribution="normal", random_state=1)

    lhs = B.apply(u).inner(p)[0, 0]
    rhs = u.inner(B.apply_adjoint(p))[0, 0]

    print(f"\n<B u, p> vs <u, B^T p> : {abs(lhs - rhs) / abs(lhs):.3e}")
    assert abs(lhs - rhs) / abs(lhs) < 1.0e-12, "B^T is not the adjoint of B"


def check_block_system(model, n_modes):
    """pyMOR's assembled block operator must agree with ExaDG's coupled solve."""
    mu = Mu(mu=np.linspace(1.0, -1.0, n_modes))

    U = model.solve(mu)
    rhs = model.rhs.as_range_array(mu)
    residual = (model.operator.apply(U, mu=mu) - rhs).norm()[0] / rhs.norm()[0]

    print(f"block residual / |rhs| : {residual:.3e}")
    assert residual < 1.0e-9, "the block system is not the one ExaDG solves"


def reduce_and_measure(model, velocity_snapshots, pressure_snapshots, test):
    """Relative error of the reduced model over the test set, in the mixed product."""
    RB_u = pod(velocity_snapshots, product=model.u_product)[0]
    RB_p = pod(pressure_snapshots, product=model.p_product)[0]

    reductor = SupremizerGalerkinStokesReductor(
        model,
        RB_u=RB_u,
        RB_p=RB_p,
        u_product=model.u_product,
        p_product=model.p_product,
    )
    rom = reductor.reduce()

    product = model.products["mixed"]

    worst = 0.0
    for mu in test:
        u_fom = model.solve(mu)
        u_rom = reductor.reconstruct(rom.solve(mu))

        worst = max(worst, (u_fom - u_rom).norm(product)[0] / u_fom.norm(product)[0])

    return worst, rom.solution_space.dim


if __name__ == "__main__":
    main()
