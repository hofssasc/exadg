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

Runs on any number of ranks. Everything reduced here is an inner product, which C++ reduces over
the communicator, so every rank computes the same small array and pyMOR keeps rank 0's. What the
dispatch below buys is only that the *evaluation* happens everywhere: the bound ExaDG model is
reachable through the coupled solver, and under MPI that solver holds an
:class:`~pymor.tools.mpi.ObjectId` for the per-rank models rather than one handle.
"""

import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.models.basic import StationaryModel
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.numpy import NumpyVectorSpace


def dispatch(model, function, basis, *args):
    """Run ``function(model, basis, *args)`` against the bound ExaDG model on every rank.

    Serially that is a direct call. In parallel the model and the basis are addressed by
    ``ObjectId`` and the call is broadcast, which is the shape
    :class:`~exadg.mor.models.saddle_point.MPIExaDGCoupledSolver` already uses. The function must
    return the *global* answer -- here always a small NumPy array assembled from inner products,
    which are reduced in C++ -- because pyMOR keeps rank 0's return value.
    """
    from pymor.tools import mpi

    if not mpi.parallel:
        return function(model, basis, *args)

    return mpi.call(
        mpi.function_call, function, model.operator.solver.models_id, basis.impl.obj_id, *args
    )


def _bound_model(model):
    """The ExaDG handle, reached through the solver that already holds it."""
    return model.operator.solver.fom


def _reconstruct(basis, coefficients):
    """``V a`` as an ExaDG vector, from a rank-local basis."""
    u = basis.space.impl.zero_vector()
    for weight, mode in zip(coefficients, basis.vectors):
        u.axpy(float(weight), mode.impl)

    return u


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


def local_momentum_blocks(model, basis):
    """The three parameter-independent pieces of the momentum block, on one rank.

    Returns ``(tensor, viscous, constant)``: the convective tensor, the projected viscous block,
    and the right-hand side's constant part. The viscous block is isolated from the momentum
    operator by removing the convective term, which is legitimate because what remains is affine
    in the velocity -- so ``r`` applications determine it.
    """
    fom = _bound_model(model)
    momentum = model.operator.blocks[0, 0]

    convective = basis.space.make_array(
        [basis.space.make_vector(fom.apply_convective(v.impl)) for v in basis.vectors]
    )
    constant = basis.inner(momentum.apply(basis.space.zeros(1))).ravel()
    viscous = basis.inner(momentum.apply(basis) - convective) - constant[:, None]

    return convective_tensor(fom, basis), viscous, constant


def local_stabilisation(model, basis, coefficients):
    """``V^T S(V a)`` with ``S(u) = N(u) - B(u, u)``, the Lax-Friedrichs term, on one rank."""
    fom = _bound_model(model)
    u = _reconstruct(basis, coefficients)

    residual = fom.apply_convective(u)
    residual.axpy(-1.0, fom.apply_convective_central(u))

    return np.array([mode.impl.inner(residual) for mode in basis.vectors])


def local_momentum_jacobian(model, basis, coefficients):
    """``V^T A'(V a) V``, the projected momentum Jacobian, on one rank."""
    fom = _bound_model(model)
    jacobian = fom.jacobian_momentum(_reconstruct(basis, coefficients))

    columns = []
    for mode in basis.vectors:
        image = basis.space.impl.zero_vector()
        jacobian.apply(image, mode.impl)
        columns.append([other.impl.inner(image) for other in basis.vectors])

    return np.array(columns).T


class FullOrderMomentum:
    """The two pieces of the momentum block that are not the tensor, evaluated at full order.

    Deliberately a small object with two methods rather than inlined code: replacing it is what
    hyper-reduction *is*. An ECSW version evaluates the same two quantities on a weighted subset
    of elements, returns the same shapes, and is a drop-in for this one.
    """

    def __init__(self, model, basis):
        self.model = model
        self.basis = basis

    def stabilisation(self, coefficients):
        return dispatch(self.model, local_stabilisation, self.basis, coefficients)

    def jacobian(self, coefficients):
        """Assembled at full order, one operator application per basis vector.

        The tensor covers the *residual's* convective part; the Jacobian is what hyper-reduction
        has to remove as well, and until it does this is what keeps the reduced model tied to the
        mesh.
        """
        return dispatch(self.model, local_momentum_jacobian, self.basis, coefficients)


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

        # B, and with it the (1,2) block: <phi_i, B^T psi_j> = <B phi_i, psi_j>, so one projection
        # serves both and the transpose is exact by construction rather than by a second
        # implementation agreeing with the first. No dispatch: these are pyMOR operators, which
        # mpi_wrap_model has already made collective.
        divergence = RB_p.inner(fom.operator.blocks[1, 0].apply(velocity))

        tensor, viscous, constant = dispatch(fom, local_momentum_blocks, velocity)

        operator = ReducedSaddlePointOperator(
            tensor=tensor,
            viscous=viscous,
            constant=constant,
            divergence=divergence,
            momentum=self.build_momentum(velocity),
        )

        return {
            "operator": operator,
            "rhs": project(fom.rhs, self._block_basis, None),
            "products": {k: project(v, self._block_basis, self._block_basis)
                         for k, v in fom.products.items()},
            "output_functional": project(fom.output_functional, None, self._block_basis),
        }

    def build_momentum(self, velocity):
        """The collaborator supplying the stabilisation and the Jacobian. Overridden to sample."""
        return FullOrderMomentum(self.fom, velocity)

    def build_rom(self, projected_operators, error_estimator):
        return StationaryModel(error_estimator=error_estimator, **projected_operators)


class ECSWStokesReductor(TensorGalerkinStokesReductor):
    """:class:`TensorGalerkinStokesReductor` with the stabilisation hyper-reduced by ECSW.

    Energy-conserving sampling and weighting: the reduced stabilisation is fitted as a
    non-negative combination of a few faces' contributions, chosen so that the fit reproduces the
    exact projected stabilisation on the training states. Nothing is interpolated -- the residual
    is evaluated exactly on the faces that are kept, which is what makes it a reasonable treatment
    of a term that is not smooth in the state.

    Args:
        training_states: Velocity snapshots to fit on. Their coefficients on the enriched basis
            are what the weights have to reproduce.
        tolerance: Relative residual at which the fit stops; larger means fewer faces.
        max_entries: Hard cap on the number of faces kept.
        Everything else as for the base class.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, u_product=None, p_product=None,
                 training_states=None, tolerance=1.0e-2, max_entries=None, **kwargs):
        super().__init__(fom, RB_u=RB_u, RB_p=RB_p, u_product=u_product, p_product=p_product,
                         **kwargs)

        self.training_states = training_states
        self.tolerance = tolerance
        self.max_entries = max_entries

    def build_momentum(self, velocity):
        # The basis is orthonormal in u_product, so this is the projection of each snapshot onto
        # the enriched space -- the states the reduced model will actually be evaluated near.
        coefficients = self.u_product.apply2(velocity, self.training_states).T

        return ECSWMomentum(
            self.fom, velocity, list(coefficients), self.tolerance, self.max_entries
        )


def sparse_nnls(matrix, target, tolerance=1.0e-2, max_entries=None):
    """Non-negative least squares, stopped as soon as the residual is small enough.

    Lawson-Hanson, terminated on a relative residual rather than on optimality. That is the whole
    point: an exact solution would use every column, and each column kept is one more face the
    reduced model has to evaluate online. Non-negativity is not decoration either -- weights that
    could go negative would let the fit cancel one face's contribution against another's, and the
    stability argument for ECSW rests on them staying positive.

    Args:
        matrix: ``G``, one column per candidate face.
        target: ``b``; for ECSW the exact totals, i.e. ``G @ ones``.
        tolerance: stop once ``|G xi - b| <= tolerance * |b|``.
        max_entries: stop after this many faces whatever the residual.

    Returns:
        ``numpy.ndarray`` of non-negative weights, mostly zero.
    """
    _, n = matrix.shape
    limit = n if max_entries is None else min(max_entries, n)

    weights = np.zeros(n)
    active = np.zeros(n, dtype=bool)
    residual = target.copy()
    threshold = tolerance * np.linalg.norm(target)

    while active.sum() < limit and np.linalg.norm(residual) > threshold:
        gradient = matrix.T @ residual
        gradient[active] = -np.inf

        candidate = int(np.argmax(gradient))
        if gradient[candidate] <= 0.0:
            break

        active[candidate] = True

        # inner loop: least squares on the active set, backing off any weight that turns negative
        while True:
            index = np.flatnonzero(active)
            solution = np.linalg.lstsq(matrix[:, index], target, rcond=None)[0]

            if (solution > 0.0).all():
                weights = np.zeros(n)
                weights[index] = solution
                break

            negative = solution <= 0.0
            step = np.min(
                weights[index][negative] / (weights[index][negative] - solution[negative])
            )
            weights[index] = weights[index] + step * (solution - weights[index])
            active[index[weights[index] <= 1.0e-14]] = False

            if not active.any():
                return np.zeros(n)

        residual = target - matrix @ weights

    return weights


def local_ecsw_weights(model, basis, states, tolerance, max_entries):
    """Train ECSW weights on one rank, keeping that rank's slice.

    Every rank assembles the columns for the faces it owns, all ranks agree on the global fit --
    the same deterministic least-squares problem, solved redundantly rather than solved once and
    scattered -- and each keeps the weights belonging to its own faces. The weights stay on the
    rank because that is where the faces are; only the diagnostics are global and returnable.

    Stashed on the local model rather than returned, because ``mpi.call`` keeps rank 0's value and
    every rank needs a different answer.
    """
    from pymor.tools import mpi

    fom = _bound_model(model)
    phi = [v.impl for v in basis.vectors]
    n_faces = fom.n_faces

    rows = []
    for coefficients in states:
        contributions = np.array(
            fom.stabilisation_contributions(phi, _reconstruct(basis, coefficients))
        ).reshape(n_faces, len(phi))
        rows.append(contributions.T)

    local = np.vstack(rows)

    pieces = mpi.comm.allgather(local) if mpi.parallel else [local]
    matrix = np.hstack(pieces)

    weights = sparse_nnls(matrix, matrix.sum(axis=1), tolerance, max_entries)

    offset = sum(piece.shape[1] for piece in pieces[: mpi.rank]) if mpi.parallel else 0
    model._ecsw_weights = weights[offset : offset + n_faces].tolist()

    selected = int((weights > 0.0).sum())
    residual = np.linalg.norm(matrix @ weights - matrix.sum(axis=1))

    return np.array([selected, matrix.shape[1], residual / np.linalg.norm(matrix.sum(axis=1))])


def local_ecsw_stabilisation(model, basis, coefficients):
    """``V^T sum_f xi_f S_f(V a)`` on one rank, using the weights trained above."""
    fom = _bound_model(model)
    residual = fom.apply_stabilisation(_reconstruct(basis, coefficients), model._ecsw_weights)

    return np.array([mode.impl.inner(residual) for mode in basis.vectors])


class ECSWMomentum(FullOrderMomentum):
    """:class:`FullOrderMomentum` with the stabilisation restricted to a weighted set of faces.

    A drop-in: same two methods, same shapes. Only ``stabilisation`` changes -- the Jacobian is
    still assembled at full order, so this is a statement about the residual and not yet a
    speed-up. Sampling the Jacobian is the next step and reuses the same weights.
    """

    def __init__(self, model, basis, states, tolerance=1.0e-2, max_entries=None):
        super().__init__(model, basis)

        selected, candidates, residual = dispatch(
            model, local_ecsw_weights, basis, states, tolerance, max_entries
        )
        self.n_selected = int(selected)
        self.n_candidates = int(candidates)
        self.training_residual = float(residual)

    def stabilisation(self, coefficients):
        return dispatch(self.model, local_ecsw_stabilisation, self.basis, coefficients)
