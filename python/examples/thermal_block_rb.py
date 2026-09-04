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

Every quantity it prints is a global one, so the output must be identical for every rank count.
That is the point of the script: it is the parallel regression check as much as it is a demo.
Paths are relative to the repository root, so run it from there.
"""

import numpy as np
from pymor.algorithms.error import reduction_error_analysis
from pymor.algorithms.pod import pod
from pymor.parameters.base import Mu
from pymor.parameters.functionals import MinThetaParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.tools import mpi

from exadg.mor.pymor_binding import mpi_stationary_model

INPUT_FILE = "applications/poisson/thermal_block/input.json"
N_TRAIN, N_TEST, N_MODES = 100, 5, 8


def main():
    model, space = mpi_stationary_model(INPUT_FILE, degree=2, refinements=4, dim=3)
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

        norm = u_fom.norm(model.energy_product)[0]
        errors.append((u_fom - u_rom).norm(model.energy_product)[0] / norm)
        estimates.append(rom.estimate_error(parameter)[0])

    print(f"max relative error : {max(errors):.12e}")
    print(f"max error estimate : {max(estimates):.12e}")
    print(f"estimator is upper : {all(e >= r for e, r in zip(estimates, errors))}")

    # The two operations that had to become collective to work on more than one rank.
    u = model.solve(Mu(mu=test[0]))
    index, value = u.amax()
    print(f"amax               : index {index[0]} value {value[0]:.12e}")
    print(f"dofs([0, 17, 113]) : {np.array2string(u.dofs([0, 17, 113]).ravel(), precision=12)}")

    # pyMOR's standard convergence table: the error against basis size, with the estimator's
    # effectivity. plot=False because this normally runs without a display.
    analysis = reduction_error_analysis(
        rom,
        fom=model,
        reductor=reductor,
        test_mus=[Mu(mu=mu) for mu in test],
        basis_sizes=4,
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
        print(f"\nwrote {record}")


if __name__ == "__main__":
    main()
