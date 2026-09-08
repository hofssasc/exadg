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

"""Reduced basis approximation of the ExaDG thermal block, driven entirely by pyMOR.

Runs unchanged on any number of ranks::

    python python/examples/thermal_block_rb.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/thermal_block_rb.py

Every quantity it prints is a global one, so the output is identical for every rank count. That
is the point of the script: it is the parallel regression check as much as it is a demo. Nine
significant digits, because that is what is genuinely rank-independent -- a different partitioning
sums floating-point reductions in a different order, and the iterative solve carries that through
to about the twelfth digit. Any real defect moves things far more than that.
Paths are relative to the repository root, so run it from there.

Sized to finish in the time you will actually give it. The certified estimator costs
O((P*r)^2) full-order applies to assemble -- with P = 64 parameters and r = 8 modes that is half
an hour -- so the mesh and the basis are kept small here. Raise ``refinements`` and ``N_MODES``
for a real study; the certificate is what makes this example worth running, not the size.
"""

import numpy as np
from pymor.algorithms.error import reduction_error_analysis
from pymor.core.logger import set_log_levels
from pymor.algorithms.pod import pod
from pymor.parameters.base import Mu
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.tools import mpi

from exadg.mor.models.stationary import mpi_stationary_model

INPUT_FILE = "applications/poisson/thermal_block/input.json"
N_TRAIN, N_TEST, N_MODES = 30, 5, 4


def main():
    # pyMOR logs one line per solve at INFO, and this script solves a great many times.
    set_log_levels({"pymor": "WARNING"})

    model, space = mpi_stationary_model(
        "thermal_block", "ThermalBlockFOM3D", INPUT_FILE, degree=2, refinements=3
    )
    n_parameters = model.operator.parameters["mu"]

    print(f"ranks              : {mpi.size}")
    print(f"degrees of freedom : {space.dim}")
    print(f"parameters         : {n_parameters}")

    rng = np.random.default_rng(0)
    train = rng.uniform(-1.0, 1.0, (N_TRAIN, n_parameters))
    test = rng.uniform(-1.0, 1.0, (N_TEST, n_parameters))

    snapshots = space.empty()
    for mu in train:
        snapshots.append(model.solve(Mu(mu=mu)))

    # A product, not the Euclidean one: a POD in the coefficient inner product weights degrees
    # of freedom by the local mesh size and is not an optimal basis in any norm of interest.
    # The energy product is used here because it is also the norm the error estimator below
    # bounds, so basis and certificate speak about the same quantity.
    basis, singular_values = pod(snapshots, product=model.energy_product, modes=N_MODES)
    print(f"POD modes          : {len(basis)}")
    print(f"singular values    : {np.array2string(singular_values[:4], precision=10)}")

    # A(mu) >= min_p exp(mu_p) * A(0) in the energy inner product, so the smallest coefficient
    # is an exact lower bound on the coercivity constant and no successive-constraint method is
    # needed. This bound holds *relative to the energy product* and not relative to the mass
    # product, which is why the reductor is given the former.
    reductor = CoerciveRBReductor(
        model,
        RB=basis,
        product=model.energy_product,
        coercivity_estimator=MinThetaParameterFunctional(
            model.operator.coefficients, np.ones(n_parameters)
        ),
    )
    rom = reductor.reduce()

    errors, estimates = [], []
    for mu in test:
        parameter = Mu(mu=mu)
        u_fom = model.solve(parameter)
        u_rom = reductor.reconstruct(rom.solve(parameter))

        # Both relative, and to the same norm. The estimator bounds the *absolute* energy-norm
        # error, so comparing it with a relative error is comparing two different quantities and
        # says nothing about whether the bound holds.
        norm = u_fom.norm(model.energy_product)[0]
        errors.append((u_fom - u_rom).norm(model.energy_product)[0] / norm)
        estimates.append(rom.estimate_error(parameter)[0] / norm)

    print(f"max relative error : {max(errors):.9e}")
    print(f"max relative bound : {max(estimates):.9e}")
    print(f"estimator is upper : {all(e >= r for e, r in zip(estimates, errors))}")

    # The two operations that had to become collective to work on more than one rank.
    u = model.solve(Mu(mu=test[0]))
    index, value = u.amax()
    print(f"amax               : index {index[0]} value {value[0]:.9e}")
    print(f"dofs([0, 17, 113]) : {np.array2string(u.dofs([0, 17, 113]).ravel(), precision=9)}")

    # pyMOR's standard convergence table: the error against basis size, with the estimator's
    # effectivity. plot=False because this normally runs without a display.
    analysis = reduction_error_analysis(
        rom,
        fom=model,
        reductor=reductor,
        test_mus=[Mu(mu=mu) for mu in test],
        basis_sizes=3,
        error_norms=[model.energy_norm],
        condition=True,
        plot=False,
    )
    print()
    print(analysis["summary"])

    # Three fields in one record, which is the comparison worth looking at.
    worst = Mu(mu=test[int(np.argmax(errors))])
    u_fom = model.solve(worst)
    u_rom = reductor.reconstruct(rom.solve(worst))
    record = model.visualize(
        (u_fom, u_rom, u_fom - u_rom),
        legend=("fom", "rom", "error"),
        filename="output/pymor/thermal_block",
    )

    if not mpi.parallel:
        # The parameter itself, next to the field it produced. One value per cell, since the
        # diffusivity is piecewise constant per block, and exp(mu) because that is what the
        # coefficient functionals evaluate to -- the parameter is the logarithm.
        coefficient = space.impl.write_coefficient(
            "output/pymor", "thermal_block_diffusivity", np.exp(worst["mu"]).tolist()
        )

        print(f"\nwrote {record}")
        print(f"      {coefficient}")


if __name__ == "__main__":
    main()
