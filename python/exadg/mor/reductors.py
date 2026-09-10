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

"""Saddle-point reduction for the incompressible Navier-Stokes momentum block.

The convective operator of a discontinuous Galerkin discretisation splits into a polynomial part
and one that is not::

    N(u) = B(u, u) + S(u)

``B`` -- the volume integral together with the central part of the numerical flux -- is trilinear,
so its Galerkin projection is a fixed third-order tensor ``C[i,j,k] = <phi_i, B(phi_j, phi_k)>``:
exact, and free of the mesh online. ``S`` is the Lax-Friedrichs stabilisation, whose ``lambda`` is
a maximum of absolute normal velocities and no polynomial at all. It is also what keeps
under-resolved flow stable, so it is sampled rather than dropped or modelled.
``python/examples/convective_split.py`` measures both halves of that statement.

Two reductors follow:

===============================  ====================================================
:class:`TensorGalerkinStokesReductor`  tensor for ``B``, every face for ``S`` -- exact
:class:`ECSWStokesReductor`            tensor for ``B``, a weighted subset of faces for ``S``
===============================  ====================================================

The first is the reference the second is measured against: neither approximates the convective
term, so they have to agree to solver tolerance.

Both halves reach this module through the vocabulary rather than by name: the model hands out a
``SplitOperator`` whose ``apply_polynomial`` is ``B`` and whose ``apply`` is ``N``, and a
``SampledOperator`` for ``S``. Nothing here calls a method that only one application binds, so a
second flow application costs a ``python_bindings.cpp`` and nothing in Python.

The tensor is built by polarisation, ``B(a, b) = 0.5 (N_c(a+b) - N_c(a) - N_c(b))``, and not from
ExaDG's linearly-implicit convective operator -- that one is also trilinear but is a *different*
bilinear map, see ``ForcedFOM::apply_trilinear``.

Runs on any number of ranks. Everything reduced here is an inner product, which C++ reduces over
the communicator, so every rank computes the same small array and pyMOR keeps rank 0's; the
dispatch below only ensures the *evaluation* happens everywhere.

.. warning::
   :func:`local_ecsw_weights` fits the weights redundantly on every rank, over a training matrix
   gathered whole. Not a speed problem -- the fit is a fraction of a second and a fraction of a
   percent of offline time -- but a memory one, and it grows along two axes: the mesh widens the
   matrix and a transient training set lengthens it. See its own warning.
"""

import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.models.basic import StationaryModel
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.stokes import SupremizerGalerkinStokesReductor
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.numpy import NumpyVectorSpace

from exadg.mor.models.saddle_point import exadg_model, exadg_models_id


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
        mpi.function_call, function, exadg_models_id(model), basis.impl.obj_id, *args
    )


def convective_tensor(split, basis):
    """``C[i,j,k] = <phi_i, Q(phi_j, phi_k)>`` for the polynomial half of a nonlinear term.

    Args:
        split: The model's ``SplitOperator``. ``apply_polynomial`` is exactly quadratic, which is
            what makes the tensor exact rather than a fit -- and it is the operator's own
            polynomial half rather than a linearisation that happens to be multilinear.
        basis: The velocity basis, **after** supremizer enrichment -- it has to be the basis the
            rest of the model is projected onto.

    Returns:
        ``numpy.ndarray`` of shape ``(r, r, r)``, symmetric in its last two indices.

    Costs ``r + r(r-1)/2`` applications of the polynomial half, plus the projections.
    """
    phi = [v.impl for v in basis.vectors]
    r = len(phi)

    diagonal = [split.apply_polynomial(p) for p in phi]

    tensor = np.zeros((r, r, r))
    for j in range(r):
        for k in range(j, r):
            if j == k:
                # Q(a, a) = Q(a) exactly, so the diagonal costs no extra evaluation
                image = diagonal[j]
            else:
                sum_jk = phi[j].copy()
                sum_jk.axpy(1.0, phi[k])

                image = split.apply_polynomial(sum_jk)
                image.axpy(-1.0, diagonal[j])
                image.axpy(-1.0, diagonal[k])
                image.scal(0.5)

            for i in range(r):
                tensor[i, j, k] = tensor[i, k, j] = phi[i].inner(image)

    return tensor


def local_momentum_blocks(model, basis):
    """The three parameter-independent pieces of the momentum block, on one rank.

    Returns ``(tensor, viscous, constant)``: the convective tensor, the projected viscous block,
    and the operator's value at zero. The viscous block is isolated by removing the whole
    nonlinear term from the momentum operator, which is legitimate because what remains is affine
    in the velocity -- so ``r`` applications determine it. Both halves of that subtraction come
    from the model's declared ``SplitOperator``; nothing here names a convective operator.
    """
    split = exadg_model(model).split_momentum()
    momentum = model.operator.blocks[0, 0]

    if split is None:
        raise ValueError(
            "this model declares no split_momentum(), so its nonlinear term has no polynomial "
            "half to build a tensor from"
        )

    nonlinear = basis.space.make_array(
        [basis.space.make_vector(split.apply(v.impl)) for v in basis.vectors]
    )
    constant = basis.inner(momentum.apply(basis.space.zeros(1))).ravel()
    viscous = basis.inner(momentum.apply(basis) - nonlinear) - constant[:, None]

    return convective_tensor(split, basis), viscous, constant


def local_sampled(model, basis, weights):
    """Create the sampled stabilisation on one rank, over the given weights.

    Returned as an :class:`~pymor.tools.mpi.ObjectId` under MPI, so that later calls address every
    rank's evaluator rather than rank 0's alone.
    """
    evaluator = exadg_model(model).sampled_momentum([mode.impl for mode in basis.vectors])

    if weights is not None:
        evaluator.set_weights(weights)

    return evaluator


def local_write_selection(builder, directory, basename):
    """Draw the selected entities on every rank. The write is collective; rank 0's path is kept."""
    return builder.write_selection(directory, basename)


def local_compiled(builder):
    """Compile one rank's builder into the evaluation half, which refers to no model."""
    return builder.compiled()


def local_evaluate(evaluator, method, coefficients):
    """Call one of the evaluator's methods on every rank. Its result is already summed over them."""
    return np.array(getattr(evaluator, method)(list(coefficients)))


def local_selected(compiled):
    """Faces the compiled operator visits, summed over ranks."""
    from pymor.tools import mpi

    count = compiled.n_selected

    return np.array([mpi.comm.allreduce(count) if mpi.parallel else count])


def _make_builder(model, basis, weights):
    """One :class:`SampledOperator` per rank, addressed by ObjectId when there is more than one."""
    from pymor.tools import mpi

    if not mpi.parallel:
        return local_sampled(model, basis, weights)

    return mpi.call(
        mpi.function_call_manage, local_sampled,
        exadg_models_id(model), basis.impl.obj_id, weights,
    )


def _compile(builder):
    """Compile each rank's builder, keeping the result by ``ObjectId`` under MPI.

    Reads the weights the builder currently holds, so it has to happen after they are installed.
    """
    from pymor.tools import mpi

    if not mpi.parallel:
        return local_compiled(builder)

    return mpi.call(mpi.function_call_manage, local_compiled, builder)


def _evaluate(evaluator, method, coefficients):
    """Ask every rank's evaluator, and take the answer -- which each of them reduced already."""
    from pymor.tools import mpi

    if not mpi.parallel:
        if method == "n_selected":
            return local_selected(evaluator)[0]

        return np.array(getattr(evaluator, method)(list(coefficients)))

    if method == "n_selected":
        return mpi.call(mpi.function_call, local_selected, evaluator)[0]

    return mpi.call(mpi.function_call, local_evaluate, evaluator, method, list(coefficients))


class FullOrderMomentum:
    """The stabilisation and its Jacobian, over every face.

    Two halves, split by phase. The *builder* is a :class:`SampledOperator` the model hands out:
    it owns the basis and the weights, and reaching a face means walking the mesh. Compiling it
    gathers what the selected faces contribute -- weights, quadrature measures, normals, basis
    traces -- into flat arrays, and the :class:`CompiledOperator` that holds them refers to
    nothing else. Every online evaluation goes through that, so the discretisation is out of the
    loop and its cost is set by the faces kept rather than by the mesh.

    Each momentum owns its pair, which is the point: a reduced model and the one it is measured
    against cannot disturb one another. :class:`ECSWMomentum` differs only in the weights.
    """

    def __init__(self, model, basis, weights=None):
        self.basis = basis
        self.builder = _make_builder(model, basis, weights)
        self._compiled = None

    @property
    def compiled(self):
        """The evaluation half, built when first asked for and dropped when the weights change."""
        if self._compiled is None:
            self._compiled = _compile(self.builder)

        return self._compiled

    @property
    def n_batches(self):
        """Face batches visited, summed over ranks -- the honest cost.

        Matrix-free evaluates a batch of four to eight faces whole, so selecting one face in a
        batch costs the same as selecting all of them. Faces kept is the size of the fit; this is
        the size of the work.
        """
        return int(_evaluate(self.compiled, "n_selected", ()))

    def stabilisation(self, coefficients):
        """``V^T sum_f w_f S_f(V a)``."""
        return _evaluate(self.compiled, "projected", coefficients)

    def stabilisation_jacobian(self, coefficients):
        """``V^T (sum_f w_f S'_f(V a)) V``.

        ExaDG freezes lambda when it linearises -- it is not differentiable -- so ``S'`` is a
        linear face operator over the same faces, and the weights carry over unchanged. Together
        with the tensor and the viscous block it reproduces ``V^T A'(V a) V`` to machine precision
        at every refinement -- the decomposition is exact, not merely consistent.
        """
        r = len(self.basis)

        return _evaluate(self.compiled, "jacobian", coefficients).reshape(r, r)

    def contributions(self, coefficients):
        """``V^T S_f(V a)`` for every face, shape ``(n_faces, r)``. Offline: it touches the mesh."""
        flat = _evaluate(self.builder, "contributions", coefficients)

        return flat.reshape(-1, len(self.basis))

    def detach(self):
        """Drop the builder, keeping the compiled half. The reduced model becomes the deliverable.

        Compiles first, because the builder is what knows how. Afterwards this momentum still
        evaluates the residual and the Jacobian, but cannot be re-fitted, asked for training data
        or asked to draw its selection -- and nothing it holds refers to the full-order model, so
        the mesh, the matrix-free caches and the preconditioners can go.

        Worth doing before timing a reduced solve as well as before shipping one: a co-resident
        full-order model evicts the compiled arrays between calls, which costs more than the
        arithmetic does.
        """
        self.compiled

        self.builder = None

    def write_selection(self, filename):
        """Draw the faces the weights select, as a VTU/PVTU record, and return its path.

        Cell data: a face is not a cell, so what is drawn is the weight each cell *carries*, summed
        over the selected faces on its boundary. Goes through the builder, which is the half that
        still knows where in the mesh a face is.
        """
        from pathlib import Path
        from pymor.tools import mpi

        base = Path(filename)
        arguments = (self.builder, str(base.parent), base.name)

        if not mpi.parallel:
            return local_write_selection(*arguments)

        return mpi.call(mpi.function_call, local_write_selection, *arguments)


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
        """The exact derivative of :meth:`apply`, assembled from the same three pieces.

        The convective part comes from the tensor -- ``d/da C a a = 2 C a`` -- so the only piece
        that is evaluated is the linearised stabilisation, and hyper-reducing that hyper-reduces
        the Jacobian as well.
        """
        assert len(U) == 1

        a_u, _ = self._split(U.to_numpy()[:, 0])
        n, n_u = self.source.dim, self.n_velocity

        matrix = np.zeros((n, n))
        matrix[:n_u, :n_u] = (
            self.viscous
            + 2.0 * np.einsum("ikj,k->ij", self.tensor, a_u)
            + self.momentum.stabilisation_jacobian(a_u)
        )
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
            are what the weights have to reproduce. For a transient model these are the states of
            whole trajectories, so there are many more of them than there are parameters.
        tolerance: Relative residual at which the fit stops; larger means fewer faces.
        max_entries: Hard cap on the number of faces kept.
        sketch_rows: Fit on a Gaussian sketch of the training matrix's rows; see
            :func:`local_ecsw_weights`. ``None`` fits on the matrix itself.
        seed: Of the sketch.
        Everything else as for the base class.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, u_product=None, p_product=None,
                 training_states=None, tolerance=1.0e-2, max_entries=None, sketch_rows=None,
                 audit_rows=64, seed=0, **kwargs):
        super().__init__(fom, RB_u=RB_u, RB_p=RB_p, u_product=u_product, p_product=p_product,
                         **kwargs)

        self.training_states = training_states
        self.tolerance = tolerance
        self.max_entries = max_entries
        self.sketch_rows = sketch_rows
        self.audit_rows = audit_rows
        self.seed = seed

    def build_momentum(self, velocity):
        # The basis is orthonormal in u_product, so this is the projection of each snapshot onto
        # the enriched space -- the states the reduced model will actually be evaluated near.
        states = self.u_product.apply2(velocity, self.training_states).T

        return ECSWMomentum(self.fom, velocity, states, self.tolerance, self.max_entries,
                            sketch_rows=self.sketch_rows, audit_rows=self.audit_rows,
                            seed=self.seed)


class _Instationary:
    """The two extra pieces an instationary reduced model needs, and the model itself.

    Mixed in front of a stationary reductor rather than subclassed from it, so that the tensor
    and the hyper-reduced variants get it in the same way and neither has to know about the
    other. ``super()`` therefore reaches the stationary ``project_operators``, which does all the
    work that is not about time: the supremizer enrichment, the convective tensor, the viscous
    block and the divergence.

    What time adds is small, and that is the point of having carried the *step* through
    ``interface.h`` rather than a time integrator. The spatial operator is unchanged -- the mass
    term is not folded into it, because the stepper combines the two itself as
    ``LincombOperator([mass, A], [gamma_0/dt, 1])`` and must be able to vary the coefficient.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, u_product=None, p_product=None,
                 check_orthonormality=None, check_tol=None):
        """Set up the bases the way the stationary reductor does, without its type assertion.

        ``SupremizerGalerkinStokesReductor.__init__`` asserts ``isinstance(fom, SaddlePointModel)``
        -- which is a *stationary* model, by pyMOR's own class hierarchy. Its actual requirements
        are weaker and an instationary saddle point meets all of them: a two-block solution space,
        a (2,1) block to build supremizers from, and a velocity product to build them in. So the
        setup is reproduced here rather than the model being dressed up as something it is not.
        """
        RB_u = fom.solution_space.subspaces[0].empty() if RB_u is None else RB_u
        RB_p = fom.solution_space.subspaces[1].empty() if RB_p is None else RB_p

        assert RB_u in fom.solution_space.subspaces[0]
        assert RB_p in fom.solution_space.subspaces[1]

        self.u_product = u_product
        self.supremizers = fom.solution_space.subspaces[0].empty()

        ProjectionBasedReductor.__init__(
            self, fom, {"RB_u": RB_u, "RB_p": RB_p}, {"RB_u": u_product, "RB_p": p_product},
            check_orthonormality=check_orthonormality, check_tol=check_tol,
        )

    def project_operators(self):
        from pymor.algorithms.projection import project

        projected = super().project_operators()
        basis = self._block_basis

        projected["mass"] = project(self.fom.mass, basis, basis)
        projected["initial_data"] = project(self.fom.initial_data, basis, None)

        self._check_mass(projected["mass"])

        return projected

    def _check_mass(self, mass):
        """The reduced mass should be the identity on the velocity block. Verified, not assumed.

        It is, because the enriched velocity basis is orthonormalised in ``u_product`` and this
        application's ``u_product`` is the mass matrix. Both halves of that can change --
        somebody orthonormalises in H1, or an application's velocity product stops being its mass
        -- and neither would fail, they would just make the reduced time derivative wrong. So the
        projection is done properly and the identity is checked rather than exploited.
        """
        n_u = self.rom_velocity_dimension()
        block = mass.matrix[:n_u, :n_u]
        deviation = np.abs(block - np.eye(n_u)).max()

        if deviation > 1.0e-10:
            self.logger.warning(
                f"the reduced velocity mass deviates from the identity by {deviation:.2e}; the "
                f"basis is not orthonormal in the mass product. Not an error -- the projected "
                f"matrix is used either way -- but it means u_product is not the mass matrix."
            )

    def rom_velocity_dimension(self):
        """Velocity modes after supremizer enrichment."""
        return len(self.bases["RB_u"]) + len(self.supremizers)

    def build_rom(self, projected_operators, error_estimator):
        from pymor.models.basic import InstationaryModel

        fom = self.fom

        return InstationaryModel(
            T=fom.T,
            # The same stepper, minus the solver: a reduced step is a small dense system, so
            # pyMOR's Newton takes it. That one substitution is the whole difference between how
            # the two models are advanced, which is what makes them comparable.
            time_stepper=fom.time_stepper.with_(solver=None),
            num_values=fom.num_values,
            error_estimator=error_estimator,
            **projected_operators,
        )


class InstationaryTensorStokesReductor(_Instationary, TensorGalerkinStokesReductor):
    """:class:`TensorGalerkinStokesReductor` over a trajectory rather than a steady state.

    Same reduced spatial operator -- an exact convective tensor, the viscous block, and the
    stabilisation over every face -- stepped by the full-order model's own scheme.
    """


class InstationaryECSWStokesReductor(_Instationary, ECSWStokesReductor):
    """:class:`ECSWStokesReductor` over a trajectory.

    The training states are now the states of trajectories, so there are ``n_steps`` times as many
    of them as there are parameters and the fit's training matrix grows in its *row* dimension.
    That is what ``sketch_rows`` is for; see :func:`local_ecsw_weights`.
    """

    def __init__(self, fom, RB_u=None, RB_p=None, u_product=None, p_product=None,
                 training_states=None, tolerance=1.0e-2, max_entries=None, sketch_rows=None,
                 audit_rows=64, seed=0, **kwargs):
        # The ECSW settings are set here rather than through ECSWStokesReductor.__init__, because
        # that one chains into the stationary base whose type assertion this class exists to
        # sidestep. Same fields, same defaults; build_momentum is inherited and reads them.
        _Instationary.__init__(self, fom, RB_u=RB_u, RB_p=RB_p, u_product=u_product,
                               p_product=p_product, **kwargs)

        self.training_states = training_states
        self.tolerance = tolerance
        self.max_entries = max_entries
        self.sketch_rows = sketch_rows
        self.audit_rows = audit_rows
        self.seed = seed


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


def row_sketch(n_rows, n_sketch, seed=0):
    """A Gaussian sketch ``S`` of shape ``(n_sketch, n_rows)``, identical on every rank.

    ``S`` mixes the *rows* of the training matrix, which are training states, not faces. Every
    rank holds all of those rows for its own columns, so every rank must apply the same ``S`` --
    hence a seeded generator rather than a drawn one. A different sketch per rank would fit a
    different problem on each and the gathered matrix would be nonsense.

    Scaled by ``1/sqrt(n_sketch)`` so that ``||S x|| ~ ||x||``, which is what lets a residual
    measured on the sketch be read as an estimate of the real one.
    """
    return np.random.default_rng(seed).standard_normal((n_sketch, n_rows)) / np.sqrt(n_sketch)


def local_ecsw_weights(evaluator, states, tolerance, max_entries, sketch_rows=None,
                       audit_rows=64, seed=0):
    """Fit the weights on one rank and install them there.

    Each rank assembles the columns for the faces it owns, all ranks agree on the global fit, and
    each installs the weights belonging to its own faces -- which is where those faces are.

    The training matrix is ``(n_train * r) x n_faces``. Both dimensions grow, and for different
    reasons: refining the mesh widens it, and training on a *trajectory* rather than on a set of
    steady states lengthens it by the number of time steps. The second is the one that bites
    first -- 24 parameters at 640 BDF-2 steps and r = 8 is 122880 rows, or 7.6 GB at refinement 6
    in 2D -- and it is what ``sketch_rows`` is for.

    Args:
        sketch_rows: If given, fit on ``S G`` rather than on ``G``, with ``S`` a Gaussian sketch
            of that many rows. The sketch is applied to each state's block **as it is assembled**,
            so ``G`` is never formed: memory falls from ``(n_train r) x n_faces`` to
            ``sketch_rows x n_faces``. ``None`` fits on the matrix itself.
        audit_rows: Rows of a *second, independent* sketch, used only to measure the residual.
            Never fitted against, which is the point -- see below.
        seed: Of the sketches. Fixed so that a fit is reproducible and identical across ranks.

    Returns:
        ``(kept, candidates, residual, assembly_seconds, nnls_seconds, sketched)``, the residual
        relative and **measured out of sample** whenever a sketch is used.

    .. warning::
       **Never report the residual of a sketched fit on its own sketch.** NNLS *minimises* over
       ``S G``, so ``||S(G xi - b)||`` is an in-sample quantity and is biased low -- and the bias
       grows exactly where a warning would be wanted. Measured at refinement 4, 24 states, r = 8
       (192 rows), against the true residual on the unsketched matrix::

           sketch    faces    on its own sketch    true
           exact        27             9.63e-03    9.63e-03
           96           21             9.72e-03    1.52e-02
           24           14             7.03e-03    4.17e-02
           6             6             5.35e-16    2.77e-01

       The last row is the degenerate case: with six rows, six columns fit exactly and the fit
       reports machine zero while being 28% wrong. Johnson-Lindenstrauss bounds ``||Sx||`` for a
       *fixed* ``x``, and the minimiser is not fixed -- it is chosen after seeing ``S``.

       So a second sketch is drawn, accumulated in the same pass, and never fitted against. ``xi``
       is fixed by the time it is used, so the bound applies and the number is honest.

    .. warning::
       **The columns are still gathered whole on every rank.** Sketching fixes the row dimension,
       which is what a transient training set grows; it does not distribute the candidates, which
       is what a 3D mesh grows. Both are needed for a large 3D run -- see ``ExaDG ROM Next
       Steps.md`` in the vault, where the column architecture is written out.
    """
    import time

    from pymor.tools import mpi

    # Timed in two halves because they answer different questions. Assembling walks the mesh once
    # per training state and must scale with it; solving sees the mesh only as the width of the
    # matrix, and would not grow with it if the candidates were distributed. Rank 0's clock, which
    # is the whole of the fit since every rank does all of it.
    started = time.perf_counter()

    n_faces = evaluator.n_entities
    states = [list(state) for state in states]
    n_modes = len(states[0])

    if sketch_rows is None:
        local = np.vstack([
            np.array(evaluator.contributions(state)).reshape(n_faces, n_modes).T
            for state in states
        ])
        audit = None
    else:
        # Streamed: each state's block is sketched into both accumulators and discarded, so the
        # full matrix is never resident. That is the whole point -- forming G and multiplying by S
        # afterwards would need exactly the memory the sketch exists to avoid. The second sketch
        # rides along for free, since the expensive part is the face loop, not the multiply.
        n_rows = len(states) * n_modes
        sketch = row_sketch(n_rows, sketch_rows, seed)
        check = row_sketch(n_rows, audit_rows, seed + 1)

        local = np.zeros((sketch_rows, n_faces))
        audit = np.zeros((audit_rows, n_faces))
        for i, state in enumerate(states):
            block = np.array(evaluator.contributions(state)).reshape(n_faces, n_modes).T
            columns = slice(i * n_modes, (i + 1) * n_modes)
            local += sketch[:, columns] @ block
            audit += check[:, columns] @ block

    pieces = mpi.comm.allgather(local) if mpi.parallel else [local]
    matrix = np.hstack(pieces)

    # S(G 1) = (SG) 1, so the target is the fitted matrix's own column sum either way.
    target = matrix.sum(axis=1)

    if audit is not None:
        audit = np.hstack(mpi.comm.allgather(audit) if mpi.parallel else [audit])

    assembled = time.perf_counter()

    weights = sparse_nnls(matrix, target, tolerance, max_entries)

    solved = time.perf_counter()

    if audit is None:
        witness, witness_target = matrix, target
    else:
        witness, witness_target = audit, audit.sum(axis=1)

    kept = int((weights > 0.0).sum())
    if sketch_rows is not None and 2 * kept > sketch_rows:
        import warnings

        warnings.warn(
            f"the fit kept {kept} faces from a sketch of {sketch_rows} rows; below about twice "
            f"the support the sketch stops constraining the fit and the residual it reports "
            f"collapses towards zero while the true one grows. Raise sketch_rows.",
            stacklevel=2,
        )

    offset = sum(piece.shape[1] for piece in pieces[: mpi.rank]) if mpi.parallel else 0
    evaluator.set_weights(weights[offset : offset + n_faces].tolist())

    return np.array([
        kept,
        matrix.shape[1],
        np.linalg.norm(witness @ weights - witness_target) / np.linalg.norm(witness_target),
        assembled - started,
        solved - assembled,
        0.0 if sketch_rows is None else 1.0,
    ])


class ECSWMomentum(FullOrderMomentum):
    """:class:`FullOrderMomentum` with the stabilisation restricted to a weighted set of faces.

    A drop-in: same methods, same shapes. Only the weights differ, and residual and Jacobian are
    both evaluated over the faces they keep -- a frozen lambda makes ``S'`` a linear face operator
    over the same faces, so one fit serves both.

    ``sketch_rows`` fits on a Gaussian sketch of the training matrix's rows rather than on the
    matrix; see :func:`local_ecsw_weights` for what that costs and what it buys. Non-negativity is
    imposed on the weights either way, so the stability argument for ECSW is untouched -- it is
    the least-squares objective that is sketched, not the constraint.
    """

    def __init__(self, model, basis, states, tolerance=1.0e-2, max_entries=None,
                 sketch_rows=None, audit_rows=64, seed=0):
        super().__init__(model, basis)

        from pymor.tools import mpi

        arguments = (self.builder, [list(state) for state in states], tolerance, max_entries,
                     sketch_rows, audit_rows, seed)
        fitted = (
            local_ecsw_weights(*arguments) if not mpi.parallel
            else mpi.call(mpi.function_call, local_ecsw_weights, *arguments)
        )

        self.n_faces = int(fitted[0])
        self.n_candidates = int(fitted[1])
        self.training_residual = float(fitted[2])

        # What the fit cost, split where the two halves scale differently. See local_ecsw_weights.
        self.assembly_seconds = float(fitted[3])
        self.nnls_seconds = float(fitted[4])

        #: Whether `training_residual` was measured on a sketch rather than on the matrix itself.
        self.sketched = bool(fitted[5])
        self.n_training_states = len(states)


