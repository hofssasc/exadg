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

"""ExaDG's vectors and operators, dressed in pyMOR's interfaces.

Nothing here knows any physics. Every class wraps one of the abstract types of
``exadg/pymor/interface.h``, so the same wrappers serve every ExaDG application:

    ExaDGVector, ExaDGVectorSpace   distributed::Vector, FullOrderModel
    ExaDGOperator                   LinearOperator      (an ExaDG operator's vmult)
    ExaDGParametricOperator         ParametricOperator
    ExaDGFunctional                 Functional          (the quantity of interest)
    RestrictedExaDGOperator         RestrictedOperator  (hyper-reduction)
    ExaDGSolver                     apply_inverse       (the application's Krylov solve)

What a particular problem *is* enters in the application's C++ and in :mod:`exadg.mor.models`,
which assembles these into a pyMOR Model.

Degrees of freedom never enter Python. Following ``pymor/bindings/fenics.py``, a vector is an
opaque handle and every operation on it is forwarded to C++; ``to_numpy`` exists for tests and
debugging only.

Capabilities are declared by C++, never assumed here: whether an operator is symmetric, can be
inverted, or can be restricted. This module asks and reports.

Under MPI pyMOR runs Python on every rank with rank 0 dispatching, so every method here executes
on all ranks at once. Anything returning a value must return the *global* answer -- ``dofs`` and
``amax`` reduce in C++ -- and a vector pyMOR hands in arrives whole on every rank, so
``vector_from_numpy`` keeps only the part it owns.
"""

from pathlib import Path

import numpy as np
from pymor.core.base import ImmutableObject
from pymor.core.exceptions import InversionError
from pymor.operators.interface import Operator
from pymor.operators.list import ListVectorArrayOperatorBase
from pymor.solvers.list import ListVectorArrayBasedSolver
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


class ExaDGVector(CopyOnWriteVector):
    """A single ExaDG degree-of-freedom vector, held by handle.

    Attributes:
        impl: The ``exadg._core.Vector`` this wraps.
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
        path after all -- ``deim`` and ``ei_greedy`` both call it. Collective, and the index is
        global: C++ takes a maximum over the values and then a *minimum over the global indices*
        attaining it, so the tie-break does not depend on the rank count. ``MPI_MAXLOC`` would
        break ties by rank, and a collateral basis built on two ranks would then differ from one
        built on four.
        """
        return self.impl.amax()


class ExaDGVectorSpace(ListVectorSpace):
    """One ExaDG discrete function space.

    Args:
        impl: A ``PyMOR::Space`` handle. A ``FullOrderModel`` is one, and a saddle-point model
            hands out two -- velocity and pressure.
        id: pyMOR's label for the space; distinct ids keep two spaces of one model apart.
    """

    def __init__(self, impl, id="STATE"):
        self.impl = impl
        self.id = id

    @property
    def dim(self):
        """Global number of degrees of freedom."""
        return self.impl.n_dofs

    def __eq__(self, other):
        # identity of the underlying space, not of the wrapper
        return type(other) is ExaDGVectorSpace and other.impl is self.impl and other.id == self.id

    def __hash__(self):
        return hash((id(self.impl), self.id))

    def zero_vector(self):
        return ExaDGVector(self.impl.zero_vector())

    def make_vector(self, obj):
        """Wrap an existing ExaDG vector handle."""
        return ExaDGVector(obj)

    def vector_from_numpy(self, data, ensure_copy=False):
        """Build a vector from NumPy data, projected onto the admissible subspace.

        pyMOR builds vectors this way for random probes and test data, and a discretisation with
        eliminated Dirichlet rows needs those rows zeroed before its affine decomposition holds.
        """
        vector = self.impl.zero_vector()
        vector.assign_numpy(np.ascontiguousarray(data, dtype=float))
        self.impl.make_admissible(vector)

        return ExaDGVector(vector)

    def random_vector(self, distribution, random_state=None, **kwargs):
        """Random vector, projected onto the admissible subspace.

        Under MPI every rank runs this and keeps the part it owns, so the ranks must agree on
        the draw. ``random_state`` is therefore not optional there: with ``None`` each rank
        seeds itself from the OS and the assembled vector is a different field on every rank,
        which is not an error anywhere -- it just produces a vector that is not a vector.
        """
        from pymor.tools import mpi

        if mpi.parallel and random_state is None:
            raise ValueError(
                "random_vector() needs an explicit random_state under MPI, so that every rank "
                "draws the same field."
            )

        rng = np.random.default_rng(random_state)

        if distribution == "normal":
            data = rng.normal(kwargs.get("loc", 0.0), kwargs.get("scale", 1.0), self.dim)
        else:
            data = rng.uniform(kwargs.get("low", 0.0), kwargs.get("high", 1.0), self.dim)

        return self.vector_from_numpy(data)


class ExaDGSolver(ListVectorArrayBasedSolver):
    """Hands linear systems to the application's own solver.

    pyMOR's extension point for an external solver: ``Operator.apply_inverse`` delegates to
    ``operator.solver``, and the base class supplies the loop over a vector array's columns.

    Attached only to operators whose C++ side declares ``has_inverse``. The adjoint solve is a
    separate C++ entry point, because it is the same system only for a symmetric operator and
    that is the operator's claim to make.
    """

    def _solve_one_vector(self, operator, v, mu, initial_guess, prepare_data):
        """Solve ``A x = v`` for a single right-hand side."""
        operator._prepare(mu)

        result = operator.impl.apply_inverse(v.impl)
        if result is None:
            raise InversionError(f"{operator.name} declares has_inverse but returned nothing.")

        return operator.range.make_vector(result)

    def _solve_adjoint_one_vector(self, operator, u, mu, initial_guess, prepare_data):
        """Solve ``A^T x = u``."""
        operator._prepare(mu)

        result = operator.impl.apply_inverse_transpose(u.impl)
        if result is None:
            raise InversionError(f"{operator.name} has no adjoint solve.")

        return operator.range.make_vector(result)


class ExaDGOperator(ListVectorArrayOperatorBase):
    """Any ExaDG ``LinearOperator``, wrapped for pyMOR.

    Args:
        space: The :class:`ExaDGVectorSpace` the operator maps *from*.
        impl: A ``PyMOR::LinearOperator`` handle.
        name: pyMOR's name for the operator; defaults to the one C++ gives it.
        range_space: Where it maps *to*, when that is a different space -- the divergence block
            of a saddle point maps velocity to pressure. Defaults to ``space``.
        component: Index into the model's affine components when this operator *is* one, and
            ``None`` otherwise. What lets :meth:`_assemble_lincomb` recognise a combination the
            application can assemble.
        model: The model to ask for an assembled operator. Passed rather than reached for through
            the space, because a space is a space and knows nothing about parameters.
        n_components: How many affine components the model has in total, which is the length of
            the coefficient vector ``model.assemble`` expects.
    """

    linear = True

    def __init__(
        self, space, impl, name=None, range_space=None, component=None, model=None,
        n_components=0,
    ):
        # pyMOR's ImmutableObject requires every __init__ argument to be stored under the same
        # name, so that with_() can reconstruct the object
        self.space = space
        self.impl = impl
        self.name = name or impl.name
        self.range_space = range_space
        self.component = component
        self.model = model
        self.n_components = n_components

        self.source = space
        self.range = range_space if range_space is not None else space
        self.parameters_own = {}

        if impl.has_inverse:
            self.solver = ExaDGSolver()

    def _prepare(self, mu):
        """Install whatever this operator needs before an apply. Nothing, unless parametric."""

    def _apply_one_vector(self, u, mu=None):
        self._prepare(mu)

        result = self.range.impl.zero_vector()
        self.impl.apply(result, u.impl)

        return result

    def _apply_adjoint_one_vector(self, v, mu=None):
        """``A^T v``. The C++ side refuses unless it can actually do it."""
        self._prepare(mu)

        result = self.source.impl.zero_vector()
        self.impl.apply_transpose(result, v.impl)

        return result

    def restricted(self, dofs):
        """Restrict to the given output degrees of freedom, for empirical interpolation.

        ``NotImplementedError`` when the application declines, because that is what pyMOR checks
        for: interpolation then degrades to evaluating the full operator instead of producing a
        wrong one. The reasons for declining belong with the discretisation and live there.
        """
        handle = self.impl.restricted([int(d) for d in dofs])

        if handle is None:
            raise NotImplementedError(
                f"{self.name} offers no restricted evaluation in this configuration."
            )

        return (
            RestrictedExaDGOperator(handle, len(dofs), f"{self.name}|dofs"),
            np.array(handle.source_dofs, dtype=int),
        )

    def _assemble_lincomb(
        self, operators, coefficients, identity_shift=0.0, solver_options=None, name=None
    ):
        """Collapse ``sum_i c_i A_i`` into a single operator the application can also invert.

        pyMOR calls this when a ``LincombOperator`` is assembled at a parameter. Returning an
        operator rather than ``None`` is what gives the model an ``apply_inverse``: otherwise the
        assembled object stays a ``LincombOperator``, which pyMOR can only invert by building a
        NumPy matrix -- impossible for a vector whose entries never enter Python.

        Whether a combination *can* be assembled is the application's call: the coefficients mean
        something there and nothing here.
        """
        if identity_shift != 0.0:
            return None

        if self.model is None:
            return None

        if not all(
            isinstance(operator, ExaDGOperator)
            and operator.space == self.space
            and operator.component is not None
            for operator in operators
        ):
            return None

        coefficients = np.asarray(coefficients)
        if np.iscomplexobj(coefficients):
            return None

        weights = np.zeros(self.n_components)
        for operator, coefficient in zip(operators, coefficients):
            weights[operator.component] += float(coefficient)

        impl = self.model.assemble(weights.tolist())
        if impl is None:
            return None

        return ExaDGOperator(self.space, impl, name=name)


class ExaDGParametricOperator(ExaDGOperator):
    """An ExaDG ``ParametricOperator``: one operator whose coefficients arrive with the parameter.

    The same equation as the affine form, presented so that pyMOR cannot see its structure and
    reaches for empirical interpolation instead of projecting each component exactly.

    The coefficients are the *same functionals* the affine form uses, evaluated here rather than
    by pyMOR, so a change to the parameterisation is made in one place and both forms follow.

    **Do not train an interpolation on** ``operator.apply(model.solve(mu), mu)``, which is what
    :func:`pymor.algorithms.ei.interpolate_operators` builds by default. For a linear stationary
    problem it is degenerate: ``A(mu) u(mu) = f`` for *every* parameter, so the evaluation set is
    rank one and the greedy converges to 1e-15 having learnt the right-hand side and nothing
    about the operator. Train on ``A(mu) V`` for the reduced basis ``V`` instead.
    """

    def __init__(self, space, impl, coefficients, name=None):
        super().__init__(space, impl, name=name)


        # stored as an __init__ argument so that pyMOR collects the functionals' parameters into
        # this operator's own, exactly as it does for a LincombOperator
        self.coefficients = coefficients

    def _prepare(self, mu):
        assert mu is not None

        self.impl.set_coefficients([float(c.evaluate(mu)) for c in self.coefficients])

    def restricted(self, dofs):
        handle = self.impl.restricted([int(d) for d in dofs])

        if handle is None:
            raise NotImplementedError(
                f"{self.name} offers no restricted evaluation in this configuration."
            )

        return (
            RestrictedExaDGOperator(
                handle, len(dofs), f"{self.name}|dofs", coefficients=self.coefficients
            ),
            np.array(handle.source_dofs, dtype=int),
        )


class RestrictedExaDGOperator(Operator):
    """An ExaDG operator evaluated on a stencil, in pyMOR's NumPy spaces.

    pyMOR's contract is an identity rather than an approximation::

        op.apply(U, mu).dofs(dofs) == restricted.apply(source.from_numpy(U.dofs(source_dofs)), mu)

    A violation does not raise -- interpolation just converges to a slightly different operator --
    so it is checked explicitly, in C++ by ``tests/pymor/restricted_operator.cc`` and through
    pyMOR's own call path by ``python/examples/thermal_block_ei.py``.
    """

    linear = True

    def __init__(self, handle, n_output_dofs, name, coefficients=None):
        # every __init__ argument is stored under its own name because pyMOR's parameter
        # collection reads them back with a plain getattr, and because with_() rebuilds from them
        self.handle = handle
        self.n_output_dofs = n_output_dofs
        self.name = name
        self.coefficients = coefficients

        self.source = NumpyVectorSpace(len(handle.source_dofs))
        self.range = NumpyVectorSpace(n_output_dofs)
        self.parameters_own = {}

    def apply(self, U, mu=None):
        if self.coefficients is not None:
            assert mu is not None
            self.handle.set_coefficients([float(c.evaluate(mu)) for c in self.coefficients])

        values = U.to_numpy()

        return self.range.make_array(
            np.array(
                [self.handle.apply(values[:, i].tolist()) for i in range(values.shape[1])]
            ).T
        )


class ExaDGFunctional(ListVectorArrayOperatorBase):
    """An ExaDG ``Functional``: the map from a state to the quantities of interest.

    Used as a pyMOR Model's output functional, so its projection onto the reduced basis comes out
    of the reductor with the projected operators.
    """

    linear = True

    def __init__(self, space, impl, name="output"):
        self.space = space
        self.source = space
        self.range = NumpyVectorSpace(impl.n_outputs)
        self.impl = impl
        self.name = name
        self.parameters_own = {}

    def _apply_one_vector(self, u, mu=None):
        return np.array(self.impl.apply(u.impl))

    def apply(self, U, mu=None):
        """Apply to every vector of the array, returning a NumPy array.

        Overridden because the base class assumes the range is also a list space, whereas the
        outputs live in a NumpyVectorSpace and have to be stacked into one array.
        """
        assert U in self.source

        if len(U) == 0:
            return self.range.empty()

        # pyMOR's NumpyVectorArray stores each vector as a COLUMN, so the stacking axis is 1 and
        # the resulting shape is (n_outputs, len(U)).
        return self.range.make_array(
            np.stack([self._apply_one_vector(u, mu=mu) for u in U.vectors], axis=1)
        )

    def apply_adjoint(self, V, mu=None):
        """``B^T v``, mapping output weights back to the state space.

        Overridden rather than inherited because the base class assumes both spaces are list
        spaces, whereas the range here is a NumpyVectorSpace. Without it pyMOR's output error
        estimator fails inside ``estimate_image``, which forms the Riesz representative of the
        output functional and therefore needs its adjoint.
        """
        assert V in self.range

        return self.source.make_array(
            [
                self.source.make_vector(self.impl.apply_transpose(column.tolist()))
                for column in V.to_numpy().T
            ]
        )


class ExaDGVisualizer(ImmutableObject):
    """Writes vector arrays as VTU/PVTU records, as pyMOR's ``visualizer`` hook.

    A file writer rather than a plot window: the model may be on a compute node, and under MPI
    each rank holds a piece of the field. deal.II writes one piece per rank plus a ``.pvtu`` that
    ParaView opens as one field, so the parallel case needs no separate path.

    ``reductor.reconstruct(rom.solve(mu))`` lands in the same space as ``fom.solve(mu)``, so the
    usual three-way comparison works::

        fom.visualize((u_fom, u_rom, u_fom - u_rom),
                      legend=("fom", "rom", "error"), filename="output/compare")
    """

    def __init__(self, space, directory="output/pymor"):
        """Write into ``directory`` unless ``visualize`` is given a filename."""
        self.space = space
        self.directory = directory

    def visualize(self, U, title=None, legend=None, filename=None, block=None, **kwargs):
        """Write the given fields.

        Args:
            U: A VectorArray, or a tuple of them to write as separate fields of one record.
            title: Accepted and ignored; there is no window to title.
            legend: Field name per entry of ``U``. Defaults to ``field_0``, ``field_1``, ...
            filename: Path without extension. Defaults to ``<directory>/<title or 'solution'>``.
            block: Accepted and ignored; nothing blocks.

        Returns:
            str: Path of the written ``.pvtu``.
        """
        arrays = U if isinstance(U, tuple) else (U,)

        vectors, names = [], []
        for position, array in enumerate(arrays):
            assert array in self.space

            label = (
                legend[position]
                if legend is not None and not isinstance(legend, str)
                else (legend if isinstance(legend, str) else f"field_{position}")
            )

            # A time series would be several records rather than several fields; until an
            # instationary model exists, refusing beats writing only the first vector.
            if len(array) != 1:
                raise NotImplementedError(
                    f"visualize() writes one vector per field, got {len(array)}."
                )

            vectors.append(array.vectors[0].impl)
            names.append(label if len(arrays) > 1 or legend is not None else "solution")

        if filename is None:
            directory, basename = self.directory, (title or "solution")
        else:
            path = Path(filename)
            directory, basename = str(path.parent), path.stem

        return self.space.impl.write_vtu(directory, basename, vectors, names)
