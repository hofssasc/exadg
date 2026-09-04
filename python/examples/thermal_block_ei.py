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

"""Empirical interpolation of the thermal block operator, measured against the exact reference.

Two things are checked here, and the second is the interesting one.

**The restriction contract.** ``Operator.restricted`` is the hook the whole of hyper-reduction
rests on, and its contract is an identity rather than an approximation::

    op.apply(U, mu).dofs(dofs) == restricted.apply(U.dofs(source_dofs), mu)

A violation does not raise; empirical interpolation just converges to a slightly different
operator than the one being reduced. The C++ counterpart of this check is
``tests/pymor/restricted_operator.cc``; here it is exercised through pyMOR's own call path.

**Whether interpolation is worth it.** It is not, for this problem, and the point of the sweep
is to show that against a reference rather than assert it. The reduced model is compared with
the exact affine projection on the same basis, so the only difference between the two numbers
is the interpolation. Serial only: ``restricted`` raises under MPI.

Run from the repository root::

    python python/examples/thermal_block_ei.py
"""

import numpy as np
from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.parameters.base import Mu
from pymor.reductors.basic import StationaryRBReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace

from exadg import thermal_block
from exadg.mor.pymor_binding import stationary_model

INPUT_FILE = "applications/poisson/thermal_block/input.json"
N_TRAIN, N_TEST, N_MODES = 60, 10, 10
EI_SIZES = (5, 10, 20, 40, 60)


def main():
    # The reduced model's linear systems are solved by converting to NumPy, which is legitimate
    # at size r but produces a warning per solve.
    set_log_levels(
        {
            "pymor": "ERROR",
            "pymor.solvers.default": "ERROR",
            "pymor.bindings.scipy": "ERROR",
        }
    )

    fom = thermal_block.ThermalBlockFOM2D(INPUT_FILE, degree=2, refinements=4)

    # The same equation twice: as a sum of affine components, and as one operator whose
    # parameter dependence pyMOR cannot see. The second is what interpolation is for.
    field, space = stationary_model(fom, form="field")
    affine, _ = stationary_model(fom, form="affine")
    n_parameters = fom.n_parameters

    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(-1.0, 1.0, (N_TRAIN, n_parameters))]
    test = [Mu(mu=m) for m in rng.uniform(-1.0, 1.0, (N_TEST, n_parameters))]

    snapshots = space.empty()
    for mu in train:
        snapshots.append(field.solve(mu))
    basis, _ = pod(snapshots, product=field.energy_product, modes=N_MODES)

    check_restriction_contract(field, space, basis, train)

    print(f"\nreference: exact affine projection on the same {N_MODES}-mode basis")
    reductor = StationaryRBReductor(affine, RB=basis, product=affine.energy_product)
    print(f"  relative state error            : {state_error(affine, reductor, test):.3e}")

    sweep(field, affine, space, basis, train, test, n_parameters)


def state_error(model, reductor, test):
    """Relative energy-norm error of the reduced model against the full one."""
    rom = reductor.reduce()

    worst = 0.0
    for mu in test:
        u = model.solve(mu)
        difference = u - reductor.reconstruct(rom.solve(mu))
        worst = max(worst, difference.norm(model.energy_product)[0] / u.norm(model.energy_product)[0])

    return worst


def check_restriction_contract(field, space, basis, train):
    """Assert pyMOR's restriction identity through pyMOR's own call path."""
    dofs = np.array([0, 37, 111, 260, 601, 1088])
    restricted, source_dofs = field.operator.restricted(dofs)

    worst = 0.0
    for mu in train[:5]:
        exact = field.operator.apply(basis, mu=mu).dofs(dofs)
        via_stencil = restricted.apply(
            NumpyVectorSpace.make_array(basis.dofs(source_dofs)), mu=mu
        ).to_numpy()
        worst = max(worst, np.abs(exact - via_stencil).max())

    print("restriction contract")
    print(f"  output dofs / stencil dofs      : {len(dofs)} / {len(source_dofs)}")
    print(f"  max |apply.dofs - restricted|   : {worst:.3e}")
    assert worst < 1.0e-10, "the restriction does not reproduce the operator"


def sweep(field, affine, space, basis, train, test, n_parameters):
    """Interpolate the operator at increasing numbers of points and measure what it buys."""
    # The evaluation set is A(mu) V, not A(mu) u(mu). The latter is what pyMOR's
    # interpolate_operators builds by default, and for a linear stationary problem it is rank
    # one -- A(mu) u(mu) is the right-hand side for every mu. The greedy then converges to
    # machine precision after a few points having learnt nothing about the operator.
    evaluations = space.empty()
    for mu in train[:20]:
        evaluations.append(field.operator.apply(basis, mu=mu))

    print(f"\ninterpolation, trained on {len(evaluations)} evaluations of A(mu) V")
    print(f"  {'points':>6}  {'greedy err':>11}  {'operator err':>13}  {'ROM err':>11}  blocks")

    for size in EI_SIZES:
        dofs, collateral, data = ei_greedy(evaluations, max_interpolation_dofs=size, copy=True)
        interpolated = EmpiricalInterpolatedOperator(
            field.operator, dofs, collateral, triangular=True
        )

        operator_error = 0.0
        for mu in test:
            u = field.solve(mu)
            exact = field.operator.apply(u, mu=mu)
            operator_error = max(
                operator_error,
                (exact - interpolated.apply(u, mu=mu)).norm()[0] / exact.norm()[0],
            )

        reductor = StationaryRBReductor(
            field.with_(operator=interpolated), RB=basis, product=field.energy_product
        )
        try:
            rom_error = state_error(affine, reductor, test)
        except Exception:
            # a rank-deficient collateral basis makes the reduced operator singular
            rom_error = float("nan")

        n_blocks = len(field.operator.restricted(dofs)[0].handle.blocks)
        print(
            f"  {len(dofs):>6}  {data['errors'][-1]:>11.3e}  {operator_error:>13.3e}  "
            f"{rom_error:>11.3e}  {n_blocks:>3}/{n_parameters}"
        )

    print(
        "\nThe interpolated model only reaches the affine reference once the stencil reads every\n"
        "block, i.e. once it has stopped being a reduction of the parameter dependence. The\n"
        "coefficient field here *is* the parameter, so there is no low-dimensional structure for\n"
        "the greedy to find -- see ExaDGFieldOperator's docstring."
    )


if __name__ == "__main__":
    main()
