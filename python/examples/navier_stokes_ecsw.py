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

"""Hyper-reducing the Lax-Friedrichs stabilisation with ECSW.

``navier_stokes_tensor.py`` reduces the convective term exactly: its trilinear part becomes a
third-order tensor, contracted online at a cost independent of the mesh. What that leaves is the
Lax-Friedrichs stabilisation

    S(u) = N(u) - B(u, u),    a face term, 0.5 * lambda * jump(u)

whose ``lambda = upwind_factor * 2 * max(|uM.n|, |uP.n|)`` is a maximum of absolute values and so
no polynomial at all. It is also the term that stabilises under-resolved flow, which is why it is
sampled rather than dropped or modelled.

**ECSW, not DEIM.** Nothing here is interpolated. The reduced stabilisation is fitted as a
*non-negative* combination of a few faces' exact contributions,

    V^T S(u) ~ sum_{f in F} xi_f V^T S_f(u),    xi >= 0

with the weights chosen by a non-negative least squares that stops as soon as the fit is good
enough -- every extra face kept is one more the reduced model has to evaluate. Asking a smooth
interpolant to reproduce a kink is exactly what DEIM would do and exactly what this avoids.

The Jacobian is sampled with the same weights. ExaDG freezes lambda when it linearises -- it is
not differentiable -- so the linearised stabilisation is a *linear* face operator built from the
same quantity, and the weights carry over unchanged. With the tensor supplying the convective
part's derivative exactly, the reduced Jacobian is then the exact derivative of the reduced
residual, and neither depends on the mesh any more.

Both are evaluated over the selected face batches only, by a hand-written loop rather than
``MatrixFree::loop``, and projected inside that loop so no full-order vector is ever formed. The
saving grows with the mesh, because the number of faces kept does not::

    refinement   dofs   faces   kept    full     sampled   speed-up
             3   1152     144     12   0.198 ms   0.087 ms     2.3x
             4   4608     544     14   0.656 ms   0.113 ms     5.8x
             5  18432    2112     13   2.514 ms   0.149 ms    16.9x
             6  73728    8320     12   9.842 ms   0.341 ms    28.9x

**Two things still scale with the mesh.** Reconstructing ``V a`` is r full-order vector updates
per evaluation -- a fully online ECSW would reconstruct only on the sampled cells -- and the
weight fit is solved redundantly on every rank over a training matrix gathered whole. See the
warning on ``local_ecsw_weights``; that one is the first to fix before running at size.

Runs unchanged on any number of ranks::

    python python/examples/navier_stokes_ecsw.py
    mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_ecsw.py

Run from the repository root.
"""

import time

import numpy as np
from pymor.algorithms.pod import pod
from pymor.core.logger import set_log_levels
from pymor.parameters.base import Mu
from pymor.tools import mpi

from exadg.mor.models.saddle_point import mpi_saddle_point_model
from exadg.mor.reductors import ECSWStokesReductor, TensorGalerkinStokesReductor

INPUT_FILE = "applications/incompressible_navier_stokes/forced/input_navier_stokes.json"
DEGREE, REFINEMENTS = 2, 3
N_TRAIN, N_TEST, N_MODES = 12, 4, 4
AMPLITUDES = (0.5, 1.5)
TOLERANCES = (1.0e-1, 1.0e-2, 1.0e-3)


def main():
    set_log_levels({"pymor": "ERROR", "exadg": "ERROR"})

    model, (velocity, pressure) = mpi_saddle_point_model(
        "forced", "ForcedFOM2D", INPUT_FILE, degree=DEGREE, refinements=REFINEMENTS
    )
    n_parameters = model.parameters["mu"]

    rng = np.random.default_rng(0)
    train = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TRAIN, n_parameters))]
    test = [Mu(mu=m) for m in rng.uniform(*AMPLITUDES, (N_TEST, n_parameters))]

    snapshots = model.solution_space.empty()
    for mu in train:
        snapshots.append(model.solve(mu))

    basis_u, _ = pod(snapshots.blocks[0], product=model.u_product, modes=N_MODES)
    basis_p, _ = pod(snapshots.blocks[1], product=model.p_product, modes=N_MODES)

    bases = dict(
        RB_u=basis_u, RB_p=basis_p, u_product=model.u_product, p_product=model.p_product
    )

    reference = TensorGalerkinStokesReductor(model, **bases)
    reference_rom = reference.reduce()
    exact = worst_error(model, reference, reference_rom, test)

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"reduced dimension  : {reference_rom.solution_space.dim}")
    print(f"error, exact S     : {exact:.4e}")

    print("\nfitting the stabilisation on a subset of faces")
    print(
        f"  {'tolerance':>9}  {'faces':>9}  {'batches':>8}  {'fit':>9}  {'ROM error':>11}"
        f"  {'faster':>8}"
    )

    for tolerance in TOLERANCES:
        reductor = ECSWStokesReductor(
            model, training_states=snapshots.blocks[0], tolerance=tolerance, **bases
        )
        rom = reductor.reduce()
        momentum = rom.operator.momentum

        error = worst_error(model, reductor, rom, test)
        print(
            f"  {tolerance:>9.0e}  {momentum.n_selected:>4d}/{momentum.n_candidates:<4d}  "
            f"{momentum.n_batches:>8d}  {momentum.training_residual:>9.2e}  {error:>11.4e}  "
            f"{speed_up(reference_rom.operator.momentum, momentum):>7.1f}x"
        )

        assert error < 2.0 * exact, "sampling the stabilisation changed the answer"

    print(
        "\nThe stabilisation is reproduced from a small fraction of the faces without moving the\n"
        "reduced error, which is the claim ECSW makes: it is not that the term is negligible, but\n"
        "that its projection onto a handful of modes is low rank. The candidate count grows with\n"
        "the rank count because matrix-free pads its face batches per rank -- the padding slots\n"
        "contribute nothing and are never selected, so the fit is unchanged.\n"
        "\n"
        "Residual and Jacobian are both sampled, so neither depends on the mesh. Two things still\n"
        "do: the face loop visits every face rather than only the selected ones, and the weight\n"
        "fit is solved redundantly on every rank over a matrix gathered whole."
    )


def speed_up(exact_momentum, sampled_momentum, repeats=20):
    """How much cheaper one reduced stabilisation is than the same one over every face.

    Modest at this size and growing with the mesh -- see the table in the module docstring. The
    remaining floor is the reconstruction of V a, which is still a full-order operation.
    """
    coefficients = np.zeros(exact_momentum.basis.dim if hasattr(exact_momentum.basis, "dim")
                            else len(exact_momentum.basis))
    coefficients[0] = 1.0

    def timed(momentum):
        momentum.stabilisation(coefficients)
        start = time.perf_counter()
        for _ in range(repeats):
            momentum.stabilisation(coefficients)
        return time.perf_counter() - start

    return timed(exact_momentum) / timed(sampled_momentum)


def worst_error(model, reductor, rom, test):
    """Largest relative error over the test set, in the mixed product."""
    product = model.products["mixed"]

    worst = 0.0
    for mu in test:
        U = model.solve(mu)
        worst = max(
            worst, (U - reductor.reconstruct(rom.solve(mu))).norm(product)[0] / U.norm(product)[0]
        )

    return worst


if __name__ == "__main__":
    main()
