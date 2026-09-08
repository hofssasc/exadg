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

"""Where the convective term is a polynomial, and where it is not.

A reduced model can represent a polynomial operator **exactly** as a small tensor, with an online
cost independent of the mesh and no interpolation anywhere. This script establishes which part of
ExaDG's convective operator qualifies, by measurement rather than by reading the flux formulas.

In divergence form the volume integral and the central part of the numerical flux are trilinear.
What is not is the Lax-Friedrichs stabilisation, whose

    lambda = upwind_factor * 2 * max(|uM.n|, |uP.n|)

is a maximum of absolute values. Writing S(u) = N(u) - B(u, u) for whatever that leaves over,

    N(u) = B(u, u) + S(u)          B trilinear, S the stabilisation

is the split a reduced model has to respect: B becomes a third-order tensor, S is what
hyper-reduction has to handle. Reducing the two together, or dropping S, means projecting an
operator the full-order model never solved.

Four things are checked, and the third is the one that changes how the tensor must be built.

Run from the repository root::

    python python/examples/convective_split.py
"""

import numpy as np
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu

from exadg import forced
from exadg.mor.models.saddle_point import saddle_point_model

UPWIND = "applications/incompressible_navier_stokes/forced/input_navier_stokes.json"
CENTRAL = "applications/incompressible_navier_stokes/forced/input_navier_stokes_central.json"
DEGREE, REFINEMENTS, N_SNAPSHOTS = 2, 3, 4


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    upwind = forced.ForcedFOM2D(UPWIND, degree=DEGREE, refinements=REFINEMENTS)
    central = forced.ForcedFOM2D(CENTRAL, degree=DEGREE, refinements=REFINEMENTS)

    print(f"upwind factors     : {upwind.upwind_factor} and {central.upwind_factor}")
    print(f"quadrature indices : {upwind.quadrature_indices}  (linearised, over-integrated)")

    snapshots = solve_at(upwind, N_SNAPSHOTS)
    u, v = snapshots[0], snapshots[1]

    quadratic(central, snapshots)
    polarisation(central, u, v)
    refinement()
    stabilisation(upwind, snapshots)
    jacobian(upwind, central)

    print(
        "\nThe convective operator is exactly quadratic once the Lax-Friedrichs term is out, so a\n"
        "third-order tensor represents it with no approximation at all. Build that tensor by\n"
        "polarising N rather than from the linearly-implicit operator: the two are different\n"
        "bilinear maps at any given mesh, and a reduced model should reproduce the operator its\n"
        "own snapshots came from. They are consistent with the same continuous form, so the gap\n"
        "converges away -- which bounds how much this costs, but does not make either choice the\n"
        "other.\n"
        "\n"
        "Everything that resists the tensor is the one stabilisation term, and it shows up twice:\n"
        "as the part of the operator no tensor can hold, and as the reason the Jacobian is only\n"
        "first-order accurate."
    )


def solve_at(fom, count):
    """Velocity snapshots at random forcing amplitudes."""
    model, _ = saddle_point_model(fom)
    rng = np.random.default_rng(0)

    return [
        model.solve(Mu(mu=m)).blocks[0].vectors[0].impl
        for m in rng.uniform(0.5, 1.5, (count, fom.n_modes))
    ]


def relative(a, b):
    """|a - b| / |b|, on ExaDG vectors."""
    difference = a.copy()
    difference.axpy(-1.0, b)

    return difference.norm() / b.norm()


def quadratic(central, snapshots):
    """N(alpha u) = alpha^2 N(u): the property that makes an exact tensor possible."""
    print("\nis the central-flux operator exactly quadratic?")
    for alpha in (2.0, 3.0, -1.0):
        scaled = snapshots[0].copy()
        scaled.scal(alpha)

        expected = central.apply_convective(snapshots[0])
        expected.scal(alpha * alpha)

        print(f"  N({alpha:>4}u) vs {alpha * alpha:>4} N(u)     : "
              f"{relative(central.apply_convective(scaled), expected):.3e}")


def polarisation(central, u, v):
    """Is ExaDG's linearly-implicit operator the bilinear form of the nonlinear one?

    It is trilinear, and it is not that form. The defect is exactly quadratic in u -- so the two
    are different bilinear maps, not one map integrated two ways. A tensor built from the wrong
    one reduces an operator nobody solves.
    """
    print("\nis ExaDG's linearly-implicit operator the polarisation of N?")

    linear_in_w = central.apply_trilinear(add(u, v), u)
    parts = central.apply_trilinear(u, u)
    parts.axpy(1.0, central.apply_trilinear(v, u))
    print(f"  C(w1+w2, v) vs sum         : {relative(linear_in_w, parts):.3e}")

    lhs = central.apply_convective(add(u, v))
    lhs.axpy(-1.0, central.apply_convective(u))
    lhs.axpy(-1.0, central.apply_convective(v))
    rhs = central.apply_trilinear(u, v)
    rhs.axpy(1.0, central.apply_trilinear(v, u))
    print(f"  N(u+v)-N(u)-N(v) vs C+C    : {relative(lhs, rhs):.3e}   <- not zero")
    print(f"  C(u,u) vs N(u)             : "
          f"{relative(central.apply_trilinear(u, u), central.apply_convective(u)):.3e}   <- not zero")


def refinement():
    """Does the gap between the two bilinear maps close as the discretisation improves?

    It does, and quickly -- about a factor of 40 per refinement at degree 2. So the two are
    consistent with the same continuous operator and differ only at the discrete level. That
    bounds the cost of picking the wrong one; it does not make them interchangeable, because a
    reduced model is judged against the discrete operator its snapshots solve.
    """
    print("\ndoes that gap converge away?")
    for degree, refinements in ((2, 2), (2, 3), (2, 4), (3, 3)):
        fom = forced.ForcedFOM2D(CENTRAL, degree=degree, refinements=refinements)
        u = solve_at(fom, 1)[0]

        print(f"  degree {degree}, refinement {refinements}       : "
              f"{relative(fom.apply_trilinear(u, u), fom.apply_convective(u)):.3e}")


def stabilisation(upwind, snapshots):
    """How much of the operator is the term no tensor can hold."""
    print("\nshare of the operator carried by the Lax-Friedrichs term")
    for i, u in enumerate(snapshots):
        full = upwind.apply_convective(u)
        S = full.copy()
        S.axpy(-1.0, upwind.apply_convective_central(u))
        print(f"  snapshot {i}                 : |S|/|N| = {S.norm() / full.norm():.4f}")


def jacobian(upwind, central):
    """The Jacobian is exact without the stabilisation and only first-order accurate with it.

    lambda is not differentiable, so ExaDG freezes it at the linearisation point instead of
    differentiating it. That is the same term the tensor cannot hold, showing up a second way.
    """
    print("\nJacobian against a finite difference, best over eps in [1e-7, 1e-3]")
    for fom, label in ((upwind, "upwind"), (central, "central")):
        model, _ = saddle_point_model(fom)
        rng = np.random.default_rng(0)
        U = model.solve(Mu(mu=rng.uniform(0.5, 1.5, fom.n_modes)))

        u = U.blocks[0]
        direction = u.copy()
        direction.scal(0.1)

        A = model.operator.blocks[0, 0]
        exact = A.jacobian(u).apply(direction)

        best = min(
            ((A.apply(u + direction * eps) - A.apply(u)) * (1.0 / eps) - exact).norm()[0]
            for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        )
        print(f"  upwind_factor {fom.upwind_factor}  ({label:>7}) : {best / exact.norm()[0]:.3e}")


def add(a, b):
    result = a.copy()
    result.axpy(1.0, b)

    return result


if __name__ == "__main__":
    main()
