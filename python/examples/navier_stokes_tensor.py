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

"""The Navier-Stokes convective term reduced exactly, as a third-order tensor.

The convective operator splits into a trilinear part and the Lax-Friedrichs stabilisation. The
trilinear part has an exact Galerkin projection -- a fixed third-order tensor, contracted online
at a cost independent of the mesh -- so a reduced model need not evaluate it at full order at all.

**A verification, not a speed-up.** Neither this reduced model nor a plain Galerkin one
approximates the convective term, so the two have to agree to solver tolerance. That agreement is
what the script checks, and it is what makes this the reference a hyper-reduced stabilisation is
later measured against. The stabilisation and the momentum Jacobian are still evaluated at full
order here, which is exactly what ECSW removes.

Runs unchanged on any number of ranks::

    python python/examples/navier_stokes_tensor.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_tensor.py

Run from the repository root.
"""

import numpy as np
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor
from pymor.tools import mpi

from exadg.mor.models.saddle_point import mpi_saddle_point_model
from exadg.mor.reductors import TensorGalerkinStokesReductor

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes.json"
DEGREE, REFINEMENTS = 2, 3
N_TRAIN, N_TEST, N_MODES = 12, 4, 4
AMPLITUDES = (0.5, 1.5)


def main():
    # pyMOR warns once per projection that the reference model's nonlinear operator has no
    # efficient projection. That is the whole point of the comparison, but it says so repeatedly.
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    model, (velocity, pressure) = mpi_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE, degree=DEGREE, refinements=REFINEMENTS
    )
    n_parameters = model.parameters["mu"]

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"pressure dofs      : {pressure.dim}")

    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, n_parameters))]
    test = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TEST, n_parameters))]

    snapshots = model.solution_space.empty()
    for mu in train:
        snapshots.append(model.solve(mu))

    basis_u, _ = pod(snapshots.blocks[0], product=model.u_product, modes=N_MODES)
    basis_p, _ = pod(snapshots.blocks[1], product=model.p_product, modes=N_MODES)

    reference, reference_rom = reduce_with(SupremizerGalerkinStokesReductor, model, basis_u, basis_p)
    tensor, tensor_rom = reduce_with(TensorGalerkinStokesReductor, model, basis_u, basis_p)

    print(f"reduced dimension  : {tensor_rom.solution_space.dim}")
    print(f"convective tensor  : {tensor_rom.operator.tensor.shape}")

    product = model.products["mixed"]

    against_reference, against_full = 0.0, 0.0
    for mu in test:
        U = model.solve(mu)
        U_reference = reference.reconstruct(reference_rom.solve(mu))
        U_tensor = tensor.reconstruct(tensor_rom.solve(mu))

        against_reference = max(
            against_reference,
            (U_reference - U_tensor).norm(product)[0] / U_reference.norm(product)[0],
        )
        against_full = max(against_full, (U - U_tensor).norm(product)[0] / U.norm(product)[0])

    print(f"tensor vs Galerkin : {against_reference:.3e}")
    print(f"tensor vs full     : {against_full:.3e}")

    assert against_reference < 1.0e-10, "the tensor is not the projection of the convective term"

    print(
        "\nThe first number is the verification: two reduced models that both represent the\n"
        "convective term exactly, one by projecting it at every residual and one by contracting a\n"
        "tensor built once. The second is the ordinary reduction error, and matches\n"
        "navier_stokes_rb.py at the same basis size.\n"
        "\n"
        "What is still full order is the Lax-Friedrichs stabilisation and the momentum Jacobian.\n"
        "Removing those is hyper-reduction, and this model is what it will be measured against."
    )


def reduce_with(reductor_type, model, basis_u, basis_p):
    """Reduce with the given reductor, returning it alongside the reduced model."""
    reductor = reductor_type(
        model,
        RB_u=basis_u,
        RB_p=basis_p,
        u_product=model.u_product,
        p_product=model.p_product,
    )

    return reductor, reductor.reduce()


if __name__ == "__main__":
    main()
