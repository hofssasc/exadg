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

Every check runs on all ranks, because that is where the bugs are. The identities themselves are
local to a face or a cell, so a partition cannot change them -- which is exactly what makes them
worth checking in parallel: a face on a partition boundary is the one place a flux can be
evaluated against the wrong exterior state, and nothing else in the suite would say so. Running
this at four ranks is what found the transport velocity's missing ghost exchange, which made
``C(w, w)`` fifty per cent wrong while every nonlinear path agreed to eight digits.

The raw operators speak ExaDG vectors, not pyMOR VectorArrays, so each check is dispatched whole
rather than assembled from remote vector operations: ``reductors.dispatch`` runs it on every rank
and only floats come back. Norms are ``l2_norm``, which is collective, so each rank computes the
same global number and pyMOR keeps rank 0's.

Run from the repository root::

    python python/examples/convective_split.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/convective_split.py
"""

import numpy as np
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.models.saddle_point import mpi_saddle_point_model
from exadg.mor.reductors import _bound_model, dispatch

UPWIND = "applications/incompressible_navier_stokes/forced/input_navier_stokes.json"
CENTRAL = "applications/incompressible_navier_stokes/forced/input_navier_stokes_central.json"
DEGREE, REFINEMENTS, N_SNAPSHOTS = 2, 3, 4


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    upwind = build(UPWIND, DEGREE, REFINEMENTS)
    central = build(CENTRAL, DEGREE, REFINEMENTS)

    snapshots = solve_at(upwind, N_SNAPSHOTS)
    central_snapshots = solve_at(central, 2)

    print(f"ranks              : {mpi.size}")
    print(f"upwind factors     : {info(upwind)[0]} and {info(central)[0]}")
    print(f"quadrature indices : {[int(i) for i in info(upwind)[2:]]}  "
          f"(linearised, over-integrated)")

    quadratic(central, central_snapshots)
    polarisation(central, central_snapshots)
    refinement()
    stabilisation(upwind, snapshots)
    face_sum(upwind, snapshots)
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


def info(model):
    """The model's scalars, gathered on every rank."""
    return dispatch(model, local_info, model.solution_space.empty().blocks[0])


def build(input_file, degree, refinements):
    """The pyMOR model, constructed collectively. The raw FOM is reached through dispatch."""
    model, _ = mpi_saddle_point_model(
        "forced", "ForcedFOM2D", input_file, degree=degree, refinements=refinements
    )

    return model


def solve_at(model, count):
    """Velocity snapshots at random forcing amplitudes, as one VectorArray."""
    rng = np.random.default_rng(0)
    snapshots = model.solution_space.empty()
    for amplitudes in rng.uniform(0.5, 1.5, (count, model.parameters["mu"])):
        snapshots.append(model.solve(Mu(mu=amplitudes)))

    return snapshots.blocks[0]


# --- the rank-local halves -----------------------------------------------------------------
#
# Each takes the bound ExaDG model and a VectorArray of snapshots, already resolved to this
# rank's piece, and returns the global answer as a small array of floats. Nothing but floats
# crosses back, so no ExaDG vector ever has to be wrapped for pyMOR.


def relative(a, b):
    """|a - b| / |b|, on ExaDG vectors. Both norms are collective, so this is global."""
    difference = a.copy()
    difference.axpy(-1.0, b)

    return difference.norm() / b.norm()


def add(a, b):
    result = a.copy()
    result.axpy(1.0, b)

    return result


def local_info(model, snapshots):
    """Scalars the printing side needs: the upwind factor, the faces, the quadrature indices.

    The face count is rank-local, so it is reduced *here*. Reducing it on the printing side would
    be a collective call made by rank 0 alone, with the other ranks parked in pyMOR's event loop
    and no matching call coming -- which does not fail, it hangs.

    ``mpi`` is imported here rather than at module scope for the same reason ``reductors`` does it:
    ranks 1..n-1 import this module from inside ``pymor.tools.mpi``, which is still initialising,
    and a module-level ``from pymor.tools import mpi`` then binds the half-built package instead.
    """
    from pymor.tools import mpi

    fom = _bound_model(model)
    faces = mpi.comm.allreduce(fom.n_faces) if mpi.parallel else fom.n_faces

    return np.array([fom.upwind_factor, faces, *fom.quadrature_indices], dtype=float)


def local_quadratic(model, snapshots, alphas):
    fom = _bound_model(model)
    u = snapshots.vectors[0].impl

    errors = []
    for alpha in alphas:
        scaled = u.copy()
        scaled.scal(alpha)

        expected = fom.apply_convective(u)
        expected.scal(alpha * alpha)

        errors.append(relative(fom.apply_convective(scaled), expected))

    return np.array(errors)


def local_polarisation(model, snapshots):
    fom = _bound_model(model)
    u, v = snapshots.vectors[0].impl, snapshots.vectors[1].impl

    linear_in_w = fom.apply_trilinear(add(u, v), u)
    parts = fom.apply_trilinear(u, u)
    parts.axpy(1.0, fom.apply_trilinear(v, u))

    lhs = fom.apply_convective(add(u, v))
    lhs.axpy(-1.0, fom.apply_convective(u))
    lhs.axpy(-1.0, fom.apply_convective(v))
    rhs = fom.apply_trilinear(u, v)
    rhs.axpy(1.0, fom.apply_trilinear(v, u))

    return np.array([
        relative(linear_in_w, parts),
        relative(lhs, rhs),
        relative(fom.apply_trilinear(u, u), fom.apply_convective(u)),
    ])


def local_gap(model, snapshots):
    fom = _bound_model(model)
    u = snapshots.vectors[0].impl

    return np.array([relative(fom.apply_trilinear(u, u), fom.apply_convective(u))])


def local_stabilisation(model, snapshots):
    fom = _bound_model(model)

    shares = []
    for vector in snapshots.vectors:
        full = fom.apply_convective(vector.impl)
        S = full.copy()
        S.axpy(-1.0, fom.apply_convective_central(vector.impl))
        shares.append(S.norm() / full.norm())

    return np.array(shares)


def local_face_sum(model, snapshots):
    fom = _bound_model(model)

    errors = []
    for vector in snapshots.vectors:
        exact = fom.apply_convective(vector.impl)
        exact.axpy(-1.0, fom.apply_convective_central(vector.impl))
        errors.append(relative(fom.apply_stabilisation(vector.impl), exact))

    return np.array(errors)


# --- the checks ------------------------------------------------------------------------------


def quadratic(central, snapshots):
    """N(alpha u) = alpha^2 N(u): the property that makes an exact tensor possible."""
    alphas = (2.0, 3.0, -1.0)

    print("\nis the central-flux operator exactly quadratic?")
    for alpha, error in zip(alphas, dispatch(central, local_quadratic, snapshots, alphas)):
        print(f"  N({alpha:>4}u) vs {alpha * alpha:>4} N(u)     : {error:.3e}")


def polarisation(central, snapshots):
    """Is ExaDG's linearly-implicit operator the bilinear form of the nonlinear one?

    It is trilinear, and it is not that form. The defect is exactly quadratic in u -- so the two
    are different bilinear maps, not one map integrated two ways. A tensor built from the wrong
    one reduces an operator nobody solves.

    The first line is also the one that fails loudly when the transport velocity reaches the
    linearised operator without its ghost values: C is then wrong on every partition boundary
    while N, which updates them itself, is not.
    """
    trilinear, polarised, against_n = dispatch(central, local_polarisation, snapshots)

    print("\nis ExaDG's linearly-implicit operator the polarisation of N?")
    print(f"  C(w1+w2, v) vs sum         : {trilinear:.3e}")
    print(f"  N(u+v)-N(u)-N(v) vs C+C    : {polarised:.3e}   <- not zero")
    print(f"  C(u,u) vs N(u)             : {against_n:.3e}   <- not zero")


def refinement():
    """Does the gap between the two bilinear maps close as the discretisation improves?

    It does, and quickly -- about a factor of 40 per refinement at degree 2. So the two are
    consistent with the same continuous operator and differ only at the discrete level. That
    bounds the cost of picking the wrong one; it does not make them interchangeable, because a
    reduced model is judged against the discrete operator its snapshots solve.
    """
    print("\ndoes that gap converge away?")
    for degree, refinements in ((2, 2), (2, 3), (2, 4), (3, 3)):
        model = build(CENTRAL, degree, refinements)
        gap = dispatch(model, local_gap, solve_at(model, 1))[0]

        print(f"  degree {degree}, refinement {refinements}       : {gap:.3e}")


def stabilisation(upwind, snapshots):
    """How much of the operator is the term no tensor can hold."""
    print("\nshare of the operator carried by the Lax-Friedrichs term")
    for i, share in enumerate(dispatch(upwind, local_stabilisation, snapshots)):
        print(f"  snapshot {i}                 : |S|/|N| = {share:.4f}")


def face_sum(upwind, snapshots):
    """The face-by-face stabilisation must add up to the operator it was split out of.

    This is the check the hyper-reduced path rests on: if a single face's contribution is wrong --
    a boundary flux, a missing lane, an exterior state read across a partition -- every weight
    fitted against it is wrong too, and nothing downstream would say so.

    The face count is the one number here that legitimately grows with the rank count: matrix-free
    pads its face batches per rank, and the padding slots contribute nothing.
    """
    print(f"\nsum over {int(info(upwind)[1])} faces against N(u) - B(u, u)")
    for i, error in enumerate(dispatch(upwind, local_face_sum, snapshots)):
        print(f"  snapshot {i}                 : {error:.3e}")


def jacobian(upwind, central):
    """The Jacobian is exact without the stabilisation and only first-order accurate with it.

    lambda is not differentiable, so ExaDG freezes it at the linearisation point instead of
    differentiating it. That is the same term the tensor cannot hold, showing up a second way.

    This one needs no dispatch: it is pyMOR all the way down, and every operator in it is already
    wrapped for MPI.
    """
    print("\nJacobian against a finite difference, best over eps in [1e-7, 1e-3]")
    for model, label in ((upwind, "upwind"), (central, "central")):
        factor = info(model)[0]

        rng = np.random.default_rng(0)
        U = model.solve(Mu(mu=rng.uniform(0.5, 1.5, model.parameters["mu"])))

        u = U.blocks[0]
        direction = u.copy()
        direction.scal(0.1)

        A = model.operator.blocks[0, 0]
        exact = A.jacobian(u).apply(direction)

        best = min(
            ((A.apply(u + direction * eps) - A.apply(u)) * (1.0 / eps) - exact).norm()[0]
            for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
        )
        print(f"  upwind_factor {factor}  ({label:>7}) : {best / exact.norm()[0]:.3e}")


if __name__ == "__main__":
    main()
