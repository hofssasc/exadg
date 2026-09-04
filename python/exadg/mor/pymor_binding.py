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

"""pyMOR binding for the ExaDG thermal block.

Wraps the ``exadg_thermal_block`` extension module in the pyMOR interfaces so that pyMOR's
algorithms -- POD, Gram-Schmidt, greedy basis generation, empirical interpolation, the
projection-based reductors and their error estimators -- operate on ExaDG's full-order model
directly.

Degrees of freedom never enter Python. Following ``pymor/bindings/fenics.py``, a vector is an
opaque handle and every operation on it is forwarded to C++; ``to_numpy`` exists for tests and
debugging only. pyMOR's manual is explicit that this is the intended design: *"direct memory
access to the vector data from Python is not required to integrate a solver with pyMOR."*

**The constrained subspace.** The affine identity

    A(mu) = sum_p exp(mu_p) A_p

holds only for vectors whose Dirichlet-constrained entries vanish. On a constrained row each
affine component acts as the identity, so summing ``P`` of them scales that entry by
``sum_p exp(mu_p)`` rather than leaving it alone. Snapshots and reduced basis vectors satisfy
the constraints by construction, but vectors pyMOR creates itself -- random probes, empirical
interpolation candidates -- do not. This module therefore projects onto the constrained subspace
whenever it creates a vector, so that pyMOR only ever sees admissible vectors.

Currently single-rank. pyMOR's parallel story is the event loop in :mod:`pymor.tools.mpi`, with
Python on every rank and rank 0 dispatching; adding it does not change the interfaces here.
"""

import numpy as np
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.solvers.list import ListVectorArrayBasedSolver
from pymor.parameters.functionals import ConstantParameterFunctional, ParameterFunctional
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


class ExaDGVector(CopyOnWriteVector):
    """A single ExaDG degree-of-freedom vector, held by handle.

    Attributes:
        impl: The ``exadg_thermal_block.Vector`` this wraps.
    """

    def __init__(self, impl):
        """Wrap an ExaDG vector handle."""
        self.impl = impl

    @classmethod
    def from_instance(cls, instance):
        """Required by CopyOnWriteVector."""
        return cls(instance.impl)

    def _copy_data(self):
        """Deep-copy on write, so pyMOR's copy-on-write bookkeeping stays honest."""
        self.impl = self.impl.copy()

    def to_numpy(self, ensure_copy=False):
        """Copy into NumPy. Tests and debugging only; not meaningful in parallel."""
        # the C++ side always builds a fresh array, so ensure_copy needs no special handling
        return self.impl.to_numpy()

    def _scal(self, alpha):
        self.impl.scal(float(alpha))

    def _axpy(self, alpha, x):
        self.impl.axpy(float(alpha), x.impl)

    def inner(self, other):
        """Euclidean inner product. The C++ side reduces over the communicator."""
        return self.impl.inner(other.impl)

    def norm(self):
        return self.impl.norm()

    def norm2(self):
        return self.impl.norm2()

    def sup_norm(self):
        return self.impl.sup_norm()

    def dofs(self, dof_indices):
        """Selected entries, as empirical interpolation requires."""
        return np.array(self.impl.dofs([int(i) for i in dof_indices]))

    def amax(self):
        """Index and magnitude of the largest entry.

        Empirical interpolation selects its next interpolation point with this, so it is on the
        path after all -- ``deim`` and ``ei_greedy`` both call it. Single rank only; the C++ side
        raises otherwise, because a componentwise reduction would return a locally correct and
        globally wrong index.
        """
        return self.impl.amax()


class ExaDGVectorSpace(ListVectorSpace):
    """The space of ExaDG degree-of-freedom vectors for one full-order model.

    Attributes:
        fom: The ``exadg_thermal_block.ThermalBlockFOM*`` instance backing the space.
    """

    def __init__(self, fom, id="STATE"):
        """Create the space belonging to ``fom``."""
        self.fom = fom
        self.id = id

    @property
    def dim(self):
        """Global number of degrees of freedom."""
        return self.fom.n_dofs

    def __eq__(self, other):
        # identity of the underlying model, not of the wrapper
        return type(other) is ExaDGVectorSpace and other.fom is self.fom and other.id == self.id

    def __hash__(self):
        return hash((id(self.fom), self.id))

    def zero_vector(self):
        return ExaDGVector(self.fom.zero_vector())

    def make_vector(self, obj):
        """Wrap an existing ExaDG vector handle."""
        return ExaDGVector(obj)

    def vector_from_numpy(self, data, ensure_copy=False):
        """Build a vector from NumPy data, projected onto the constrained subspace.

        The projection is deliberate: pyMOR builds vectors this way for random probes and test
        data, and an unconstrained vector would silently violate the affine decomposition.
        """
        vector = self.fom.vector_from_numpy(np.ascontiguousarray(data, dtype=float))
        self.fom.zero_constrained(vector)

        return ExaDGVector(vector)

    def random_vector(self, distribution, random_state=None, **kwargs):
        """Random vector, projected onto the constrained subspace."""
        rng = np.random.default_rng(random_state)

        if distribution == "normal":
            data = rng.normal(kwargs.get("loc", 0.0), kwargs.get("scale", 1.0), self.dim)
        else:
            data = rng.uniform(kwargs.get("low", 0.0), kwargs.get("high", 1.0), self.dim)

        return self.vector_from_numpy(data)


class ExaDGBlockOperator(ListVectorArrayOperatorBase):
    """One affine component ``A_i`` of the thermal block operator.

    ``A(c) = sum_i c_i A_i`` exactly, so these are the objects a reductor projects; the parameter
    dependence is carried by the coefficient functionals of the enclosing
    :class:`~pymor.operators.constructions.LincombOperator`, not by this operator.

    Indexed by *parameter*, not by block. At degree zero those are the cells of the mesh and at
    degree one the coefficient's degrees of freedom; either way the index is the one
    ``n_parameters``, ``set_diffusivity`` and the SPDE prior all agree on. Indexing by
    lexicographic block instead would work on a Cartesian mesh with a piecewise constant
    coefficient and nowhere else, and would need a permutation nobody would remember to apply.
    """

    linear = True

    def __init__(self, space, block, name=None):
        """Wrap the affine component of the given parameter index."""
        # pyMOR's ImmutableObject requires every __init__ argument to be stored under the
        # same name, so that with_() can reconstruct the object
        self.space = space
        self.source = self.range = space
        self.block = block
        self.name = name or f"A_{block}"
        self.parameters_own = {}

    def _apply_one_vector(self, u, mu=None):
        result = self.range.fom.zero_vector()
        self.range.fom.apply_component_operator(self.block, result, u.impl)

        return result

    def restricted(self, dofs):
        """Restrict this affine component to the given output degrees of freedom.

        The component ``A_p`` is the operator assembled with the indicator of block ``p`` as its
        coefficient, which is why the raw-coefficient entry point is used: an indicator is not the
        exponential of any log diffusivity.
        """
        _require_piecewise_constant(self.space.fom)

        coefficients = np.zeros(self.space.fom.n_blocks)
        coefficients[self.block] = 1.0

        return _restricted_operator(self.space, coefficients.tolist(), dofs, f"{self.name}|dofs")

    def _assemble_lincomb(
        self, operators, coefficients, identity_shift=0.0, solver_options=None, name=None
    ):
        """Collapse ``sum_p c_p A_p`` into a single invertible operator.

        pyMOR calls this when a ``LincombOperator`` is assembled at a parameter. Returning an
        operator here rather than ``None`` is what gives the model an ``apply_inverse``: without
        it the assembled object stays a ``LincombOperator``, which pyMOR can only invert by
        converting to a NumPy matrix, and that is impossible for a vector type whose entries
        never enter Python.

        Returns ``None``, leaving pyMOR's generic path in place, whenever the combination is not
        one ExaDG can represent. The coefficients become diffusivities, so they have to be
        positive; a reductor forming a difference of operators lands here legitimately and must
        not be silently handed a wrong answer.
        """
        if identity_shift != 0.0:
            return None

        if not all(
            isinstance(operator, ExaDGBlockOperator) and operator.space is self.space
            for operator in operators
        ):
            return None

        coefficients = np.asarray(coefficients)
        if np.iscomplexobj(coefficients):
            return None

        weights = np.zeros(self.space.fom.n_parameters)
        for operator, coefficient in zip(operators, coefficients):
            weights[operator.block] += float(coefficient)

        if np.any(weights <= 0.0):
            return None

        return ExaDGAffineOperator(self.space, weights, name=name)


def _restricted_operator(space, coefficients, dofs, name, parameter=None):
    """Build the NumPy-space operator pyMOR's empirical interpolation expects.

    pyMOR's contract is that for any vector array ``U`` in the source space ::

        op.apply(U).dofs(dofs) == restricted.apply(source.from_numpy(U.dofs(source_dofs)))

    The stencil -- which cells touch these degrees of freedom, and which further degrees of
    freedom those cells carry -- is resolved in C++, because it is a question about the mesh and
    answering it in Python would mean exporting a full-order sized sparsity pattern.

    Args:
        space: An :class:`ExaDGVectorSpace`.
        coefficients: Fixed block diffusivities, or ``None`` when the operator is parametric.
        dofs: Output degrees of freedom.
        name: Name of the restricted operator.
        parameter: Name of the parameter carrying the log-diffusivity field, or ``None`` for a
            non-parametric restriction. The parametric form is what empirical interpolation of
            the *field* operator needs, since there the coefficient is only known at apply time.

    Returns:
        Tuple ``(restricted_operator, source_dofs)``.
    """
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    handle = space.fom.restricted([int(d) for d in dofs])
    source_dofs = np.array(handle.source_dofs, dtype=int)

    n_blocks = space.fom.n_parameters

    class _Restricted(Operator):
        """The operator on the stencil, evaluated one column at a time."""

        linear = True

        def __init__(self):
            self.source = NumpyVectorSpace(len(source_dofs))
            self.range = NumpyVectorSpace(len(dofs))
            self.handle = handle
            self.coefficients = coefficients
            self.parameter = parameter
            self.name = name
            self.parameters_own = {} if parameter is None else {parameter: n_blocks}

        def apply(self, U, mu=None):
            values = U.to_numpy()

            if self.parameter is None:
                columns = [
                    self.handle.apply_coefficients(self.coefficients, values[:, i].tolist())
                    for i in range(values.shape[1])
                ]
            else:
                assert mu is not None
                log_diffusivity = np.asarray(mu[self.parameter], dtype=float).tolist()

                columns = [
                    self.handle.apply(log_diffusivity, values[:, i].tolist())
                    for i in range(values.shape[1])
                ]

            return self.range.make_array(np.array(columns).T)

    return _Restricted(), source_dofs


class ExaDGFieldSolver(ListVectorArrayBasedSolver):
    """Solves ``A(mu) x = v`` for the parametric field operator.

    Differs from :class:`ExaDGSolver` only in where the coefficient comes from: there it is
    baked into the assembled operator, here it arrives with the parameter values.
    """

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        """Solve at the log-diffusivity field carried by ``mu``."""
        assert mu is not None

        operator.range.fom.set_diffusivity(
            np.exp(np.asarray(mu[operator.parameter], dtype=float)).tolist()
        )

        return operator.range.make_vector(operator.range.fom.apply_inverse_current(v.impl))

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        """The operator is symmetric, so this is the same solve."""
        return self._solve_one_vector(operator, u, mu, initial_guess, prepare_data)


class ExaDGFieldOperator(ListVectorArrayOperatorBase):
    """``A(mu)`` as a single parametric operator, rather than a sum of affine components.

    Mathematically this is the same operator :func:`affine_operator` builds. The difference is
    what it lets pyMOR do with it, and that matters once the coefficient is piecewise constant
    per cell.

    The affine form has one component per parameter. Projecting it costs ``P`` full-order
    applies for every basis vector and stores ``P`` reduced matrices, and assembling the reduced
    operator online costs ``P r^2`` -- all linear in a parameter dimension that now grows with
    the mesh. That is the hyper-reduction wall, and hitting it is the point of Tier 2.

    Presented as one operator instead, the parameter dependence is opaque to pyMOR, so pyMOR
    treats it the way it treats a genuinely non-affine operator: empirical interpolation. The
    interpolated operator evaluates ``A(mu) u`` at ``M`` degrees of freedom through
    :meth:`restricted`, and ``M`` is chosen by the interpolation, not by the mesh.

    Note what that costs in exchange. The restricted evaluation reads the coefficient only on
    the cells touching those ``M`` degrees of freedom, so the interpolated operator is *exactly
    insensitive* to every parameter outside that stencil. For a forward solve that is the
    intended approximation; for a Jacobian with respect to the parameters it is a statement that
    most of them do not matter, which is false. Whether it is false enough to matter is measured
    against the affine reference rather than assumed.
    """

    linear = True

    def __init__(self, space, parameter="mu", name=None):
        """Wrap the operator with its coefficient supplied at apply time."""
        self.space = space
        self.source = self.range = space
        self.parameter = parameter
        self.name = name or "A(mu)"
        self.parameters_own = {parameter: space.fom.n_parameters}

        self.solver = ExaDGFieldSolver()

    def _apply_one_vector(self, u, mu=None):
        assert mu is not None

        result = self.range.fom.zero_vector()

        self.range.fom.set_diffusivity(
            np.exp(np.asarray(mu[self.parameter], dtype=float)).tolist()
        )
        self.range.fom.apply_current(result, u.impl)

        return result

    def restricted(self, dofs):
        """Restrict to the given output degrees of freedom, keeping the parameter dependence."""
        _require_piecewise_constant(self.space.fom)

        return _restricted_operator(
            self.space, None, dofs, f"{self.name}|dofs", parameter=self.parameter
        )


def _require_piecewise_constant(fom):
    """Refuse the empirical-interpolation path unless the coefficient is piecewise constant.

    ``RestrictedLaplace`` precomputes each cell's contribution *per block*, which presumes the
    coefficient is constant on a cell. With a nodal field a cell carries several basis functions
    and the precomputation would have to be per local degree of freedom instead. Since §5 of the
    notes measured this path and rejected it for a field-valued parameter, the generalisation is
    deferred rather than guessed at -- and refusing beats returning a wrong operator.
    """
    if fom.coefficient_degree > 0:
        raise NotImplementedError(
            f"the restricted operator is implemented for a piecewise constant coefficient, but "
            f"this model's coefficient has degree {fom.coefficient_degree}"
        )


class ExaDGSolver(ListVectorArrayBasedSolver):
    """Solves linear systems by handing them to ExaDG's preconditioned conjugate gradients.

    This is pyMOR's intended extension point for an external solver: ``Operator.apply_inverse``
    delegates to ``operator.solver``, and :class:`ListVectorArrayBasedSolver` supplies the loop
    over the columns of a vector array, so only the single-vector solve has to be written.

    The thermal block operator is symmetric, so the adjoint solve is the same solve.
    """

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        """Solve ``A x = v`` for a single right-hand side."""
        operator.range.fom.set_diffusivity(operator.weights.tolist())

        return operator.range.make_vector(operator.range.fom.apply_inverse_current(v.impl))

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        """Solve ``A^T x = u``, which is the same system because ``A`` is symmetric."""
        return self._solve_one_vector(operator, u, mu, initial_guess, prepare_data)


class ExaDGAffineOperator(ListVectorArrayOperatorBase):
    """The operator ``sum_p w_p A_p`` at fixed, positive coefficients ``w``.

    This is the assembled counterpart of the parametric ``LincombOperator``, and it exists for
    one reason: it can be inverted. ``apply_inverse`` hands the system to ExaDG's own
    preconditioned conjugate gradient solver, so the pyMOR algorithms that need full-order
    solves -- greedy basis generation, residual-based error estimation, least-squares (LSPG)
    projection -- work against the real solver instead of failing with an ``InversionError``.

    Coefficients are stored as diffusivities and converted to logarithms at the interface,
    because that is the parameterisation the ExaDG application takes.
    """

    linear = True

    def __init__(self, space, weights, name=None):
        """Wrap the operator assembled at the given block diffusivities."""
        weights = np.array(weights, dtype=float)
        weights.flags.writeable = False

        # pyMOR's ImmutableObject requires every __init__ argument under the same name
        self.space = space
        self.source = self.range = space
        self.weights = weights
        self.name = name or "A(mu)"
        self.parameters_own = {}

        # the hook pyMOR's DefaultSolver looks for once the LincombOperator is assembled
        self.solver = ExaDGSolver()

    @property
    def log_weights(self):
        """Coefficients in the log parameterisation the ExaDG application expects."""
        return np.log(self.weights).tolist()

    def _apply_one_vector(self, u, mu=None):
        result = self.range.fom.zero_vector()

        self.range.fom.set_diffusivity(self.weights.tolist())
        self.range.fom.apply_current(result, u.impl)

        return result

    def restricted(self, dofs):
        """Restrict the assembled operator to the given output degrees of freedom."""
        _require_piecewise_constant(self.space.fom)

        return _restricted_operator(
            self.space, self.weights.tolist(), dofs, f"{self.name}|dofs"
        )


class ExaDGMassSolver(ListVectorArrayBasedSolver):
    """Inverts the mass matrix with ExaDG's preconditioned conjugate gradients.

    Required by any algorithm that forms a Riesz representative: least-squares (LSPG)
    projection, and residual-based error estimation for certified reduced models.
    """

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        """Solve ``M x = v``."""
        return operator.range.make_vector(operator.range.fom.apply_inverse_mass(v.impl))

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        """The mass matrix is symmetric, so this is the same solve."""
        return self._solve_one_vector(operator, u, mu, initial_guess, prepare_data)


class ExaDGMassOperator(ListVectorArrayOperatorBase):
    """The mass matrix, i.e. the L2 inner product of the finite element space.

    This is the product a proper orthogonal decomposition must be taken in. Using the Euclidean
    inner product of the coefficient vectors instead weights degrees of freedom by the local
    mesh size and does not give the L2-optimal basis the error analysis assumes.
    """

    linear = True

    def __init__(self, space, name="mass"):
        """Wrap the mass operator of the given space."""
        self.space = space
        self.source = self.range = space
        self.name = name
        self.parameters_own = {}
        self.solver = ExaDGMassSolver()

    def _apply_one_vector(self, u, mu=None):
        result = self.range.fom.zero_vector()
        self.range.fom.apply_mass(result, u.impl)

        return result


class ExaDGObservationOperator(ListVectorArrayOperatorBase):
    """Sensor evaluation, mapping a state to the observation vector.

    Used as the output functional of the model, so that ``B V`` comes out of the reductor along
    with the projected operators rather than having to be assembled separately.
    """

    linear = True

    def __init__(self, space, name="observation"):
        """Wrap the sensor operator of the given space."""
        from pymor.vectorarrays.numpy import NumpyVectorSpace

        self.space = space
        self.source = space
        self.range = NumpyVectorSpace(space.fom.n_sensors)
        self.name = name
        self.parameters_own = {}

    def _apply_one_vector(self, u, mu=None):
        return np.array(self.source.fom.observe(u.impl))

    def apply(self, U, mu=None):
        """Apply to every vector of the array, returning a NumPy array.

        Overridden because the base class assumes the range is also a list space, whereas the
        observations live in a NumpyVectorSpace and have to be stacked into one array.
        """
        assert U in self.source

        if len(U) == 0:
            return self.range.empty()

        # pyMOR's NumpyVectorArray stores each vector as a COLUMN, so the stacking axis is 1
        # and the resulting shape is (n_sensors, len(U)).
        return self.range.make_array(
            np.stack([self._apply_one_vector(u, mu=mu) for u in U.vectors], axis=1)
        )

    def apply_adjoint(self, V, mu=None):
        """``B^T v``, mapping sensor weights back to the state space.

        Overridden rather than inherited because the base class assumes both spaces are list
        spaces, whereas the source here is a NumpyVectorSpace. Without it pyMOR's output error
        estimator fails inside ``estimate_image``, which forms the Riesz representative of the
        output functional and therefore needs its adjoint.
        """
        assert V in self.range

        return self.source.make_array(
            [
                self.source.make_vector(self.source.fom.observe_transpose(column.tolist()))
                for column in V.to_numpy().T
            ]
        )


class ExponentialParameterFunctional(ParameterFunctional):
    """The coefficient functional ``mu -> exp(mu[index])`` of one affine component.

    Written out rather than assembled from an expression string, because the expression form
    does not survive the high-dimensional benchmark. ``ExpressionParameterFunctional`` takes the
    derivatives as one string per parameter component and the second derivatives as one string
    per pair, so a model with ``P`` components needs ``P`` functionals carrying ``P`` and ``P^2``
    strings each. At the eight-by-eight thermal block that is a few thousand strings and nobody
    notices; with one coefficient per cell, ``P`` is the cell count and the same construction
    asks for ``P^3`` -- a billion at refinement five -- before a single solve has happened.

    Here every derivative is known in closed form and costs nothing to return: the function is
    its own derivative in its own component, and zero in every other.
    """

    def __init__(self, index, size, parameter="mu"):
        """Build the functional of component ``index`` of a parameter of length ``size``."""
        self.index = index
        self.size = size
        self.parameter = parameter
        self.parameters_own = {parameter: size}
        self.name = f"exp(mu[{index}])"

    def evaluate(self, mu=None):
        """Value of the functional at the given parameter values."""
        assert mu is not None

        return float(np.exp(mu[self.parameter][self.index]))

    def d_mu(self, parameter, index=0):
        """Partial derivative, which is the functional itself or zero.

        Applying this twice returns the functional again, which is the correct second
        derivative: ``d^2/dmu_p^2 exp(mu_p) = exp(mu_p)``.
        """
        if parameter != self.parameter or index != self.index:
            return ConstantParameterFunctional(0, name=f"{self.name}_d_{parameter}_{index}")

        return self


def affine_operator(space):
    """Assemble ``A(mu) = sum_p exp(mu_p) A_p`` as a pyMOR LincombOperator.

    The coefficient functionals carry exact first and second derivatives, so pyMOR's sensitivity
    machinery gives exact parameter derivatives of the reduced model.

    Args:
        space: An :class:`ExaDGVectorSpace`.

    Returns:
        LincombOperator: The parametric operator.
    """
    size = space.fom.n_parameters

    return LincombOperator(
        [ExaDGBlockOperator(space, p) for p in range(size)],
        [ExponentialParameterFunctional(p, size) for p in range(size)],
    )


def stationary_model(fom, form="affine"):
    """Build the pyMOR StationaryModel for an ExaDG thermal block.

    This is the object to hand to pyMOR's reductors, greedy algorithms and empirical
    interpolation.

    Args:
        fom: An ``exadg_thermal_block.ThermalBlockFOM2D`` or ``...3D``.
        form: ``"affine"`` for the sum of one component per block, which is exact and gives
            pyMOR the full parametric structure, or ``"field"`` for the single parametric
            operator of :class:`ExaDGFieldOperator`, which is what empirical interpolation
            needs. The two solve the same equation; see :class:`ExaDGFieldOperator` for why
            both exist.

    Returns:
        Tuple ``(model, space)``.
    """
    from pymor.core.logger import set_log_levels
    from pymor.models.basic import StationaryModel

    # pyMOR logs one line per solve at INFO; a greedy run or a chain would emit a flood
    set_log_levels({"pymor": "WARNING"})

    space = ExaDGVectorSpace(fom)

    if form == "affine":
        operator = affine_operator(space)
    elif form == "field":
        operator = ExaDGFieldOperator(space)
    else:
        raise ValueError(f"form must be 'affine' or 'field', got {form!r}")

    rhs = space.make_vector(fom.rhs_vector())

    return (
        StationaryModel(
            operator=operator,
            rhs=VectorOperator(space.make_array([rhs])),
            output_functional=ExaDGObservationOperator(space),
            products={"mass": ExaDGMassOperator(space)},
        ),
        space,
    )
