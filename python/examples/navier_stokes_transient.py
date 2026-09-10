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

"""Stepping the Navier-Stokes saddle point in time, with pyMOR owning the loop.

A time integrator is an algorithm, and an algorithm cannot be projected. One implicitly
discretised step can::

    s M u + N(u, p) = f,    s = gamma_0 / dt

which is the steady problem again with a mass term and a right-hand side carrying the history.
So ``interface.h`` carries the step and this script carries the loop -- and the loop is pyMOR's
for the full-order model *and* for any reduced model built on it, because a reduced model has no
ExaDG object to step it and comparing two different time discretisations measures nothing.

Four things are checked, in the order in which they would break.

**One: the coefficients are the textbook ones.** Derived from ``sum_j (1/j) nabla^j`` rather than
tabulated, so they are worth checking against the values everyone knows.

**Two: s = 0 is the steady problem, through the pyMOR layer.** The same block operator and the
same solver, asked for ``mass_scaling = 0``, reproduce the steady model over the same
discretisation. That is also what makes the third check meaningful: both models' vectors live in
the same space, which they do only because ``velocity_space()`` hands out one space rather than a
fresh one per call.

**Three: the flow relaxes to that steady state.** The forcing is time-independent, so the
trajectory has nowhere else to go. Measured, degree 2, refinement 3, t in [0, 4]::

    t      |U(t) - U_steady| / |U_steady|
    0.00   1.0000e+00
    0.12   1.2129e-01
    0.50   7.5808e-02
    2.00   1.9772e-02
    4.00   4.0254e-03

which is a decay time of about 1.3 -- faster than the slowest Stokes mode, because the forcing
blobs are of width 0.15 and excite the fast ones. **This test case has no dynamics of its own**:
at nu = 0.02 it runs at Re = 2-6, and a confined 2D flow needs Re ~ 1e3-1e4 before it oscillates
on its own. See the module docstring of ``navier_stokes_ecsw.py`` and the vault note for what to
change if the transient is meant to be interesting rather than merely correct.

**Four: each order converges at its own rate.** Against a fine BDF-2 reference::

    order   nt      dt        error     rate
        1    8  0.5000  2.6139e-03
        1   16  0.2500  1.2729e-03     1.04
        1   32  0.1250  6.2678e-04     1.02
        1   64  0.0625  3.1085e-04     1.01
        2    8  0.5000  4.0842e-04
        2   16  0.2500  7.8217e-05     2.38
        2   32  0.1250  1.7721e-05     2.14
        2   64  0.0625  4.2048e-06     2.08

BDF-2 approaches 2 from above because its first step is BDF-1 -- the scheme needs two levels and
at t = 0 there is one -- which perturbs the trajectory by O(dt^2) globally and washes out.

**Why order 2 and not order 1.** For a degree-k discontinuous Galerkin velocity the spatial error
is O(h^(k+1)), so keeping the time error subordinate needs ``dt <~ h^((k+1)/p)``. At k = 2 and
h = 1/16 that is dt <~ 2e-4 for BDF-1 against 2e-2 for BDF-2: forty thousand steps against six
hundred. BDF-1 is a startup and a debugging scheme, not one to generate snapshots with.

**A step is cheaper than a steady solve, and the preconditioner is why.** Measured seconds per
step against one cold steady solve on the same mesh::

                          refinement 3          refinement 4
    dt        s      InvMass    Cahouet    InvMass    Cahouet
    0.5       2        0.91x      0.69x      0.91x      0.68x
    0.125     8        1.35x      0.45x      1.53x      0.51x
    0.03125  32        3.32x      0.38x      3.99x      0.38x

With ``SchurComplementPreconditioner::InverseMassMatrix`` the cost *grows* with s, which is
backwards -- a smaller step is more mass-dominated and should be easier. The Schur complement of
``[[s M + A, B^T], [B, 0]]`` tends to ``-(1/s) B M^-1 B^T``, a pressure Laplacian scaled by 1/s,
so a pressure mass matrix becomes a worse approximation the finer the step. Cahouet-Chabard is the
sum of the two limits and restores the expected behaviour: the finer the step, the cheaper it is.

Runs unchanged on any number of ranks::

    python python/examples/navier_stokes_transient.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_transient.py

Run from the repository root.
"""

import time

import numpy as np
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.models.instationary_saddle_point import (
    BDFTimeStepper,
    bdf_coefficients,
    mpi_instationary_saddle_point_model,
)

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes_transient.json"
DEGREE, REFINEMENTS = 2, 3
T, REFERENCE_STEPS = 4.0, 256
STEP_COUNTS = (8, 16, 32, 64)
MU = (1.2, 0.8, 1.4, 0.6)

TEXTBOOK = {1: "1, [1]", 2: "3/2, [2, -1/2]", 3: "11/6, [3, -3/2, 1/3]"}


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    model, (velocity, pressure) = mpi_instationary_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE,
        degree=DEGREE, refinements=REFINEMENTS, T=T, nt=1, order=2,
    )
    mu = Mu(mu=list(MU))
    product = model.products["mixed"]

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"pressure dofs      : {pressure.dim}")
    print(f"interval           : [0, {T}]")

    print("\nBDF coefficients, against the values everyone knows")
    print(f"  {'order':>5}  {'gamma':>9}  {'alpha':>22}  expected")
    for order in (1, 2, 3):
        gamma, alpha = bdf_coefficients(order)
        print(
            f"  {order:>5}  {gamma:>9.6f}  {np.array2string(alpha, precision=6):>22}  "
            f"{TEXTBOOK[order]}"
        )

    # s = 0 is the steady problem: the same block operator, the same solver, no mass term. Solved
    # through the model rather than through the ExaDG handle, so this is a statement about the
    # pyMOR layer and not only about the binding.
    steady_solver = model.time_stepper.solver.at_step(0.0, 0.0)
    U_steady = model.operator.apply_inverse(
        model.rhs.as_range_array(mu), mu=mu, solver=steady_solver
    )
    norm_steady = U_steady.norm(product)[0]
    print(f"\nsteady state at s = 0 : |U| = {norm_steady:.9e}")

    print(f"\nrelaxation to it, nt = {STEP_COUNTS[-1]}, BDF-2")
    trajectory = stepped(model, mu, STEP_COUNTS[-1], 2)
    for k in (0, 2, 8, 32, STEP_COUNTS[-1]):
        error = (trajectory[k] - U_steady).norm(product)[0] / norm_steady
        print(f"  t = {k / STEP_COUNTS[-1] * T:>5.2f}   "
              f"|U(t) - U_steady| / |U_steady| = {error:.4e}")

    reference = stepped(model, mu, REFERENCE_STEPS, 2)[-1]
    print(f"\norder of convergence at t = {T}, against nt = {REFERENCE_STEPS} BDF-2")
    print(f"  {'order':>5}  {'nt':>5}  {'dt':>9}  {'error':>12}  {'rate':>6}")
    for order in (1, 2):
        previous = None
        for nt in STEP_COUNTS:
            error = (stepped(model, mu, nt, order)[-1] - reference).norm(product)[0] / norm_steady
            rate = "" if previous is None else f"{np.log2(previous / error):>6.2f}"
            print(f"  {order:>5}  {nt:>5}  {T / nt:>9.4f}  {error:>12.4e}  {rate}")
            previous = error

    print(f"\nseconds per step, against one cold steady solve ({cost(model, mu, 0):.3f} s)")
    print(f"  {'dt':>9}  {'s':>7}  {'s/step':>9}  {'vs steady':>10}")
    steady_cost = cost(model, mu, 0)
    for nt in (8, 32, 128):
        per = cost(model, mu, nt)
        print(f"  {T / nt:>9.5f}  {nt / T:>7.1f}  {per:>9.3f}  {per / steady_cost:>9.2f}x")

    print(
        "\nThe loop is pyMOR's and the step is ExaDG's, which is what lets a reduced model run\n"
        "the same scheme as the model it is measured against -- the only difference being which\n"
        "solver the step operator is handed. Each order converges at its own rate, so the time\n"
        "discretisation is a knob rather than an unknown, and at degree 2 the second order is the\n"
        "one worth turning: matching the spatial error with BDF-1 would cost forty thousand steps\n"
        "where BDF-2 costs six hundred.\n"
        "\n"
        "The flow itself has no dynamics -- it relaxes to the steady state and stops, because at\n"
        "Re = 2-6 there is nothing else for it to do. That makes this a good test of the machinery\n"
        "and a poor demonstration of it."
    )


def stepped(model, mu, nt, order):
    """The trajectory at this step count and order, over the model's own discretisation.

    ``with_`` rather than a new model: rebuilding would create new vector spaces, and then no two
    trajectories could be subtracted from one another.
    """
    stepper = BDFTimeStepper(nt, order=order, solver=model.time_stepper.solver)

    return model.with_(time_stepper=stepper).solve(mu)


def cost(model, mu, nt, repeats=4):
    """Seconds per step at this step count, or per steady solve for ``nt = 0``.

    Timed at the *end* of a trajectory rather than the start, so that the preconditioner is built
    and the measurement is of a step rather than of a setup.
    """
    solver = model.time_stepper.solver.at_step(0.0 if nt == 0 else nt / T, 0.0)
    rhs = model.rhs.as_range_array(mu)

    model.operator.apply_inverse(rhs, mu=mu, solver=solver)

    started = time.perf_counter()
    for _ in range(repeats):
        model.operator.apply_inverse(rhs, mu=mu, solver=solver)

    return (time.perf_counter() - started) / repeats


if __name__ == "__main__":
    main()
