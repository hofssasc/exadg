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

"""pyMOR ``StationaryModel`` over any ExaDG model that declares an affine operator.

    A(mu) u = f(mu),   A(mu) = sum_i c_i(mu) A_i,   f(mu) = f_0 + sum_j d_j(mu) f_j

Everything structural is read from the model: how many affine components there are, which
parameter entry each belongs to, whether the right-hand side is parametric, which products and
output functional exist, and whether the operator can also be presented as a single parametric
object for empirical interpolation. No application needs a file of its own.

Naming and parameterisation are *not* read from C++. ``parameter_shape`` says there are sixteen
coefficients; it does not say they are called ``mu`` or that the operator depends on them through
an exponential. Those are modelling choices, and keeping them here means changing one is not a
recompile.
"""

import functools

import numpy as np
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ConstantParameterFunctional, ParameterFunctional

from exadg.mor.binding import (
    ExaDGFunctional,
    ExaDGOperator,
    ExaDGParametricOperator,
    ExaDGVectorSpace,
    ExaDGVisualizer,
)


class ExponentialParameterFunctional(ParameterFunctional):
    """The coefficient functional ``mu -> exp(mu[index])``.

    Written out rather than built from an expression string: ``ExpressionParameterFunctional``
    carries ``P`` and ``P^2`` derivative strings per functional, so ``P`` components cost ``P^3``
    strings -- a billion once there is one coefficient per cell. Here every derivative is known
    in closed form and costs nothing.
    """

    def __init__(self, index, size, parameter="mu"):
        """Build the functional of component ``index`` of a parameter of length ``size``."""
        self.index = index
        self.size = size
        self.parameter = parameter
        self.parameters_own = {parameter: size}
        self.name = f"exp({parameter}[{index}])"

    def evaluate(self, mu=None):
        """Value of the functional at the given parameter values."""
        assert mu is not None

        return float(np.exp(mu[self.parameter][self.index]))

    def d_mu(self, parameter, index=0):
        """Partial derivative, which is the functional itself or zero.

        Applying this twice returns the functional again, which is the correct second derivative:
        ``d^2/dmu_p^2 exp(mu_p) = exp(mu_p)``.
        """
        if parameter != self.parameter or index != self.index:
            return ConstantParameterFunctional(0, name=f"{self.name}_d_{parameter}_{index}")

        return self


def parameter_names(shape, parameters=None):
    """Names for the parameter groups a model declares.

    A single group defaults to ``"mu"``. Several are not named automatically: generated names like
    ``mu_0`` would appear in every ``Mu(...)`` a caller writes, and a one-time error here is
    cheaper than that confusion.
    """
    shape = list(shape)

    if parameters is not None:
        parameters = (parameters,) if isinstance(parameters, str) else tuple(parameters)

        if len(parameters) != len(shape):
            raise ValueError(
                f"the model declares {len(shape)} parameter group(s) of sizes {shape}, but "
                f"{len(parameters)} name(s) were given: {parameters}"
            )

        return parameters

    if len(shape) <= 1:
        return ("mu",) * len(shape)

    raise ValueError(
        f"this model declares {len(shape)} parameter groups of sizes {shape}; pass "
        f"parameters=(...) to name them, since the names appear in every Mu(...) you write"
    )


def default_coefficients(components, shape, names):
    """One coefficient functional per affine component: ``exp(mu_p)``.

    The log parameterisation the thermal block's prior is written in -- a modelling choice, not a
    structural one. An application whose operator is linear in its parameters should pass
    ``ParameterFunctional``s of its own.
    """
    return [
        ExponentialParameterFunctional(c.index, shape[c.slot], names[c.slot]) for c in components
    ]


def stationary_model(
    fom, parameters=None, coefficients=None, form="affine", directory="output/pymor"
):
    """Build the pyMOR ``StationaryModel`` of an ExaDG full-order model.

    Args:
        fom: Any bound ``PyMOR::FullOrderModel``.
        parameters: Name per parameter group; see :func:`parameter_names`.
        coefficients: One ``ParameterFunctional`` per affine component, in the order the model
            declares them. Defaults to :func:`default_coefficients`.
        form: ``"affine"`` for the sum of the model's affine components, which is exact and gives
            pyMOR the full parametric structure, or ``"field"`` for the single parametric operator
            that empirical interpolation needs. The two solve the same equation; see
            :class:`~exadg.mor.binding.ExaDGParametricOperator` for why both exist.
        directory: Where the visualizer writes when it is not given a filename.

    Returns:
        Tuple ``(model, space)``.
    """
    from pymor.models.basic import StationaryModel

    space = ExaDGVectorSpace(fom)

    shape = list(fom.parameter_shape)
    names = parameter_names(shape, parameters)

    components = fom.operator_components()
    if coefficients is None:
        coefficients = default_coefficients(components, shape, names)
    elif len(coefficients) != len(components):
        raise ValueError(
            f"the model declares {len(components)} affine components but "
            f"{len(coefficients)} coefficient functionals were given"
        )

    if form == "affine":
        if not components:
            raise ValueError("this model declares no affine components; try form='field'")

        operator = LincombOperator(
            [
                ExaDGOperator(
                    space,
                    component.op,
                    component=position,
                    model=fom,
                    n_components=len(components),
                )
                for position, component in enumerate(components)
            ],
            coefficients,
        )
    elif form == "field":
        impl = fom.parametric_operator()
        if impl is None:
            raise ValueError("this model offers no parametric operator; try form='affine'")

        operator = ExaDGParametricOperator(space, impl, coefficients)
    else:
        raise ValueError(f"form must be 'affine' or 'field', got {form!r}")

    products = {
        name: ExaDGOperator(space, impl, name=name) for name, impl in fom.products().items()
    }

    output = fom.output_functional()

    return (
        StationaryModel(
            operator=operator,
            rhs=_rhs(fom, space, shape, names),
            output_functional=None if output is None else ExaDGFunctional(space, output),
            products=products,
            visualizer=ExaDGVisualizer(space, directory=directory),
        ),
        space,
    )


def _rhs(fom, space, shape, names):
    """The right-hand side as a pyMOR operator, parametric or not.

    A model may declare a constant part, affine components, or both, and the sum is the
    right-hand side. An application whose parameters live entirely in the forcing declares no
    constant part and lands in the ``LincombOperator`` branch without anything here knowing.
    """
    def as_operator(vector):
        return VectorOperator(space.make_array([space.make_vector(vector)]))

    constant = fom.rhs()
    components = fom.rhs_components()

    if not components:
        if constant is None:
            raise ValueError("this model declares no right-hand side")

        return as_operator(constant)

    operators = [as_operator(component.vector) for component in components]
    coefficients = default_coefficients(components, shape, names)

    if constant is not None:
        operators.insert(0, as_operator(constant))
        coefficients.insert(0, ConstantParameterFunctional(1.0))

    return LincombOperator(operators, coefficients)


def _build_model(module_name, class_name, args, kwargs, model_kwargs):
    """Construct the per-rank model. Module level so that it survives pickling to the ranks."""
    import importlib

    module = importlib.import_module(f"exadg.{module_name}")
    fom = getattr(module, class_name)(*args, **kwargs)

    return stationary_model(fom, **model_kwargs)[0]


def mpi_stationary_model(module_name, class_name, *args, **kwargs):
    """Build the model, wrapped for MPI when the interpreter is running in parallel.

    The model has to be constructed *on every rank*, because each rank owns a piece of the mesh,
    so what is shipped is a recipe rather than a model. The application is named by string for
    the same reason: a recipe has to pickle, and a compiled class does not.

    Run it as ::

        mpirun -n 4 python -m pymor.tools.mpi reduce.py

    which starts pyMOR's event loop on ranks 1..n-1 and the script on rank 0. Running the script
    directly under ``mpirun`` without ``-m pymor.tools.mpi`` executes it once per rank instead,
    and the ranks will deadlock on the first collective call.

    Args:
        module_name: Application module inside the ``exadg`` package, e.g. ``"thermal_block"``.
        class_name: Model class in it, e.g. ``"ThermalBlockFOM3D"``.
        *args: Passed to that class.
        **kwargs: Passed to that class, except ``parameters``, ``coefficients``, ``form`` and
            ``directory``, which go to :func:`stationary_model`.

    Returns:
        Tuple ``(model, space)``. Serially these are exactly what :func:`stationary_model`
        returns; in parallel the model is an MPI-wrapped one and the space is its
        ``solution_space``, an :class:`~pymor.vectorarrays.mpi.MPIVectorSpace`.
    """
    from pymor.tools import mpi

    model_kwargs = {
        key: kwargs.pop(key)
        for key in ("parameters", "coefficients", "form", "directory")
        if key in kwargs
    }

    factory = functools.partial(_build_model, module_name, class_name, args, kwargs, model_kwargs)

    if not mpi.parallel:
        model = factory()

        return model, model.solution_space

    from pymor.models.mpi import mpi_wrap_model

    # pickle_local_spaces=False because an ExaDGVectorSpace holds a handle to the C++ model and
    # cannot be pickled; pyMOR then refers to the local spaces by an id registered on each rank.
    model = mpi_wrap_model(factory, use_with=True, pickle_local_spaces=False)

    return model, model.solution_space
