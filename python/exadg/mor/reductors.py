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

"""Saddle-point reduction with the convective term projected exactly, as a third-order tensor.

The convective operator of a discontinuous Galerkin discretisation splits into a part that is a
polynomial in the velocity and a part that is not::

    N(u) = B(u, u) + S(u)

``B`` is the volume integral together with the central part of the numerical flux: trilinear, so
its Galerkin projection is a fixed third-order tensor ``C[i,j,k] = <phi_i, B(phi_j, phi_k)>``,
built once offline and contracted online at a cost independent of the mesh. ``S`` is the
Lax-Friedrichs stabilisation, whose ``lambda`` is a maximum of absolute normal velocities and
therefore no polynomial at all -- see ``python/examples/convective_split.py``, which measures both
halves of that statement.

This reductor projects ``B`` exactly and leaves ``S`` at full order. That is deliberately **not**
fast: every reduced residual still evaluates ``S`` on the whole mesh. It is the reference against
which a hyper-reduced ``S`` is measured, in the same way ``stokes_rb.py`` is the reference for
``navier_stokes_rb.py`` -- because nothing here approximates anything, the reduced model has to
reproduce a plain Galerkin projection to solver tolerance.

The tensor is built from ``B`` by polarisation,

    B(a, b) = 0.5 * ( N_c(a + b) - N_c(a) - N_c(b) )

rather than from ExaDG's linearly-implicit convective operator. That operator is also trilinear,
but it is a *different* bilinear map -- see the docstring of ``ForcedFOM::apply_trilinear``.

.. warning::
   Serial only for now. Building the tensor and evaluating ``S`` reach the bound model through
   ``A.fom``, which under MPI is rank 0's handle; both need dispatching to every rank in the shape
   ``MPIExaDGCoupledSolver`` already uses. The inner products are global either way, so rank 0's
   answer is the right one -- it is the dispatch that is missing, not the arithmetic.
"""

import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.models.basic import StationaryModel
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.numpy import NumpyVectorSpace


def convective_tensor(fom, basis):
    """``C[i,j,k] = <phi_i, B(phi_j, phi_k)>`` for the trilinear part of the convective operator.

    Args:
        fom: A bound model offering ``apply_convective_central``, i.e. the convective operator
            with a central flux. That operator is exactly quadratic, which is what makes the
            tensor exact rather than a fit.
        basis: The velocity basis, **after** supremizer enrichment -- it has to be the basis the
            rest of the model is projected onto.

    Returns:
        ``numpy.ndarray`` of shape ``(r, r, r)``, symmetric in its last two indices.

    Costs ``r + r(r-1)/2`` applications of the convective operator, plus the projections.
    """
    phi = [v.impl for v in basis.vectors]
    r = len(phi)

    diagonal = [fom.apply_convective_central(p) for p in phi]

    tensor = np.zeros((r, r, r))
    for j in range(r):
        for k in range(j, r):
            if j == k:
                # B(a, a) = N_c(a) exactly, so the diagonal costs no extra evaluation
                image = diagonal[j]
            else:
                sum_jk = phi[j].copy()
                sum_jk.axpy(1.0, phi[k])

                image = fom.apply_convective_central(sum_jk)
                image.axpy(-1.0, diagonal[j])
                image.axpy(-1.0, diagonal[k])
                image.scal(0.5)

            for i in range(r):
                tensor[i, j, k] = tensor[i, k, j] = phi[i].inner(image)

    return tensor


class FullOrderMomentum:
    """The two pieces of the momentum block that are not the tensor, evaluated at full order.

    Deliberately a small object with two methods rather than inlined code: replacing it is what
    hyper-reduction *is*. An ECSW version evaluates the same two quantities on a weighted subset
    of elements and is a drop-in for this one.

    Args:
        fom: A bound ``SaddlePointModel`` offering ``apply_convective``,
            ``apply_convective_central`` and ``jacobian_momentum``.
        basis: The enriched velocity basis.
    """

    def __init__(self, fom, basis):
        self.fom = fom
        self.basis = basis
        self.phi = [v.impl for v in basis.vectors]

    def reconstruct(self, coefficients):
        """``V a`` as an ExaDG vector."""
        u = self.basis.space.impl.zero_vector()
        for weight, mode in zip(coefficients, self.phi):
            u.axpy(float(weight), mode)

        return u

    def stabilisation(self, coefficients):
        """``V^T S(V a)`` with ``S(u) = N(u) - B(u, u)``, the Lax-Friedrichs term."""
        u = self.reconstruct(coefficients)

        residual = self.fom.apply_convective(u)
        residual.axpy(-1.0, self.fom.apply_convective_central(u))

        return np.array([mode.inner(residual) for mode in self.phi])

    def jacobian(self, coefficients):
        """``V^T A'(V a) V``, the projected momentum Jacobian.

        Assembled at full order, one operator application per basis vector. The tensor covers the
        *residual's* convective part; the Jacobian is what hyper-reduction has to remove as well,
        and until it does this is the term that keeps the reduced model tied to the mesh.
        """
        jacobian = self.fom.jacobian_momentum(self.reconstruct(coefficients))

        columns = []
        for mode in self.phi:
            image = self.basis.space.impl.zero_vector()
            jacobian.apply(image, mode)
            columns.append([other.inner(image) for other in self.phi])

        return np.array(columns).T


class ReducedSaddlePointOperator(Operator):
    """The projected block system, with the convective term contracted from its tensor.

    Acts on the flat reduced space the supremizer reductor uses: the first ``n_u`` coefficients
    are velocity, the rest pressure. The rows are

        u:  C a_u a_u  +  K a_u  +  c  +  S(a_u)  +  B^T a_p
        p:  B a_u

    with ``C`` the convective tensor, ``K`` and ``c`` the viscous block and its constant part, and
    ``S`` supplied by a collaborator so that it can be hyper-reduced later.
    """

    linear = False

    def __init__(self, tensor, viscous, constant, divergence, momentum, name="A_r"):
        self.tensor = tensor
        self.viscous = viscous
        self.constant = constant
        self.divergence = divergence
        self.momentum = momentum
        self.name = name

        self.n_velocity = tensor.shape[0]
        self.n_pressure = divergence.shape[0]

        self.source = self.range = NumpyVectorSpace(self.n_velocity + self.n_pressure)
        self.parameters_own = {}

    def _split(self, coefficients):
        return coefficients[: self.n_velocity], coefficients[self.n_velocity :]

    def apply(self, U, mu=None):
        assert U in self.source

        images = []
        for column in U.to_numpy().T:
            a_u, a_p = self._split(column)

            velocity = (
                np.einsum("ijk,j,k->i", self.tensor, a_u, a_u)
                + self.viscous @ a_u
                + self.constant
                + self.momentum.stabilisation(a_u)
                + self.divergence.T @ a_p
            )
            images.append(np.concatenate([velocity, self.divergence @ a_u]))

        return self.range.make_array(np.array(images).T)

    def jacobian(self, U, mu=None):
        assert len(U) == 1

        a_u, _ = self._split(U.to_numpy()[:, 0])
        n, n_u = self.source.dim, self.n_velocity

        matrix = np.zeros((n, n))
        matrix[:n_u, :n_u] = self.momentum.jacobian(a_u)
        matrix[:n_u, n_u:] = self.divergence.T
        matrix[n_u:, :n_u] = self.divergence

        return NumpyMatrixOperator(matrix, name=f"{self.name}_jacobian")


class TensorGalerkinStokesReductor(SupremizerGalerkinStokesReductor):
    """:class:`~pymor.reductors.stokes.SupremizerGalerkinStokesReductor` with an exact convective
    tensor in place of pyMOR's generic projection of the nonlinear momentum block.

    Same supremizer enrichment, same reduced space, same ``reconstruct``. What differs is that the
    convective term is projected once into a third-order tensor rather than evaluated at full
    order and projected on every reduced residual.

    Takes the same arguments as its base class.
    """

    def project_operators(self):
        from pymor.algorithms.projection import project

        fom = self.fom
        RB_u, RB_p = self.bases["RB_u"], self.bases["RB_p"]

        # Enrichment, as the base class does it: one supremizer per pressure mode is what gives
        # the reduced spaces a discrete inf-sup condition.
        if len(self.supremizers) < len(RB_p):
            self.supremizers.append(
                self.compute_supremizers(RB_p, offset=len(self.supremizers)),
                remove_from_other=True,
            )

        velocity = cat_arrays([RB_u, self.supremizers])
        velocity = gram_schmidt(velocity, offset=len(RB_u), product=self.u_product, copy=False)
        self._block_basis = fom.solution_space.make_block_diagonal_array((velocity, RB_p))

        A = fom.operator.blocks[0, 0]
        B = fom.operator.blocks[1, 0]

        # the bound model, reached through the operator that holds it; it is the same handle the
        # coupled solver uses
        impl = A.fom

        # B, and with it the (1,2) block: <phi_i, B^T psi_j> = <B phi_i, psi_j>, so one
        # projection serves both and the transpose is exact by construction rather than by
        # a second implementation agreeing with the first.
        divergence = RB_p.inner(B.apply(velocity))

        # The viscous block and its constant part, isolated from A by removing the convective
        # term. Both are affine in the velocity, so r applications suffice.
        convective = velocity.space.make_array(
            [velocity.space.make_vector(impl.apply_convective(v.impl)) for v in velocity.vectors]
        )
        constant = velocity.inner(A.apply(velocity.space.zeros(1))).ravel()
        viscous = velocity.inner(A.apply(velocity) - convective) - constant[:, None]

        operator = ReducedSaddlePointOperator(
            tensor=convective_tensor(impl, velocity),
            viscous=viscous,
            constant=constant,
            divergence=divergence,
            momentum=FullOrderMomentum(impl, velocity),
        )

        return {
            "operator": operator,
            "rhs": project(fom.rhs, self._block_basis, None),
            "products": {k: project(v, self._block_basis, self._block_basis)
                         for k, v in fom.products.items()},
            "output_functional": project(fom.output_functional, None, self._block_basis),
        }

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)
