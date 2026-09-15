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

``navier_stokes_tensor.py`` reduces the convective term exactly, as a third-order tensor. What
that leaves is the stabilisation ``S(u) = N(u) - B(u, u)``, a face term whose
``lambda = upwind_factor * 2 * max(|uM.n|, |uP.n|)`` is a maximum of absolute values and so no
polynomial at all. It is also what keeps under-resolved flow stable, which is why it is sampled
rather than dropped or modelled.

**ECSW, not DEIM.** Nothing is interpolated. The reduced stabilisation is fitted as a
*non-negative* combination of a few faces' exact contributions,

    V^T S(u) ~ sum_{f in F} xi_f V^T S_f(u),    xi >= 0

with the weights chosen by a non-negative least squares that stops as soon as the fit is good
enough -- every extra face kept is one more the reduced model evaluates. Asking a smooth
interpolant to reproduce a kink is what DEIM would do and what this avoids. Non-negativity is not
decoration either: it is what makes the sampled operator inherit the sign structure of the one it
replaces.

The Jacobian is sampled with the same weights. ExaDG freezes ``lambda`` when it linearises -- it is
not differentiable -- so the linearised stabilisation is a *linear* face operator over the same
faces. With the tensor supplying the convective part's derivative exactly, the reduced Jacobian is
then the exact derivative of the reduced residual and neither depends on the mesh.

Both are read off a ``SampledOperator`` that speaks *reduced coefficients* rather than velocity
vectors. That is what makes the online cost independent of the mesh: given a vector it would have
to reconstruct ``V a`` everywhere before looking at a dozen faces. Those coefficients, the weights
and the face geometry are all the evaluation needs, so they are split into a ``CompiledOperator``
holding no reference to the discretisation.

Two records come out of a run. ``navier_stokes_ecsw_{velocity,pressure}`` holds the POD modes, the
full-order field at one parameter, and each tolerance's reduced field and error -- all at the same
parameter, so the errors are comparable. ``navier_stokes_ecsw_faces_<tol>`` is the selection
itself: a surface mesh of the faces the fit kept, one cell per face, carrying ``ecsw_weight``.

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
DEGREE, REFINEMENTS = 2, 4
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
    exact, plot_mu = worst_error(model, reference, reference_rom, test)

    print(f"ranks              : {mpi.size}")
    print(f"velocity dofs      : {velocity.dim}")
    print(f"reduced dimension  : {reference_rom.solution_space.dim}")
    print(f"error, exact S     : {exact:.4e}")

    print("\nfitting the stabilisation on a subset of faces")
    print(
        f"  {'tolerance':>9}  {'faces':>9}  {'batches':>8}  {'fit':>9}  {'ROM error':>11}"
        f"  {'faster':>8}"
    )

    # Everything is plotted at one parameter -- the reference ROM's worst -- so that the errors
    # below are comparable to each other and the full-order field is written once rather than
    # three times. The modes are views into the bases; nothing here is copied.
    U_fom = model.solve(plot_mu)
    fields = [model.solution_space.make_array([basis_u[k], basis_p[k]]) for k in range(N_MODES)]
    names = [f"mode_{k + 1}" for k in range(N_MODES)]
    fields.append(U_fom)
    names.append("fom")

    for tolerance in TOLERANCES:
        reductor = ECSWStokesReductor(
            model, training_states=snapshots.blocks[0], tolerance=tolerance, **bases
        )
        rom = reductor.reduce()
        momentum = rom.operator.momentum

        error, _ = worst_error(model, reductor, rom, test)
        print(
            f"  {tolerance:>9.0e}  {momentum.n_faces:>4d}/{momentum.n_candidates:<4d}  "
            f"{momentum.n_batches:>8d}  {momentum.training_residual:>9.2e}  {error:>11.4e}  "
            f"{speed_up(reference_rom.operator.momentum, momentum):>7.1f}x"
        )

        assert error < 2.0 * exact, "sampling the stabilisation changed the answer"

        tag = f"{tolerance:.0e}".replace("-", "_")
        U_rom = reductor.reconstruct(rom.solve(plot_mu))
        fields.extend([U_rom, U_fom - U_rom])
        names.extend([f"rom_{tag}", f"error_{tag}"])

        # Where in the domain the fit put its quadrature: the selected faces themselves, as a
        # surface mesh with one cell per face.
        momentum.write_selection(f"output/pymor/navier_stokes_ecsw_faces_{tag}")

    model.visualize(fields, legend=names, filename="output/pymor/navier_stokes_ecsw")

    print(
        "\nThe stabilisation is reproduced from a small fraction of the faces without moving the\n"
        "reduced error, which is the claim ECSW makes: it is not that the term is negligible, but\n"
        "that its projection onto a handful of modes is low rank. The candidate count grows with\n"
        "the rank count because matrix-free pads its face batches per rank -- the padding slots\n"
        "contribute nothing and are never selected, so the fit is unchanged.\n"
        "\n"
        "Residual and Jacobian are both sampled and neither touches the mesh -- the cost is set\n"
        "by the faces kept, and the faces kept are set by the rank of the term rather than by\n"
        "the mesh. What still touches it is the weight fit, which is offline."
    )


def speed_up(exact_momentum, sampled_momentum, repeats=20):
    """How much cheaper one reduced stabilisation is than the same one over every face.

    Both go through a compiled operator, so this compares two array traversals of different
    lengths: the sampled one is flat in the mesh, and the ratio grows with it.
    """
    coefficients = np.zeros(len(exact_momentum.basis))
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
    worst_mu = None
    for mu in test:
        U = model.solve(mu)
        error = (U - reductor.reconstruct(rom.solve(mu))).norm(product)[0] / U.norm(product)[0]
        if error > worst:
            worst = error
            worst_mu = mu

    return worst, worst_mu


if __name__ == "__main__":
    main()
