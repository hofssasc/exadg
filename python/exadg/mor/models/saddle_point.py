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

"""pyMOR ``SaddlePointModel`` over any ExaDG model that declares velocity/pressure blocks.

    [ A   B* ] [u]   [f]
    [ B   0  ] [p] = [g]

pyMOR assembles that block operator itself, forming the (1,2) block as ``AdjointOperator(B)``, so
``B`` has to carry a real adjoint rather than borrow its own apply. The application declares that
and the binding passes it through; nothing here assumes it.

Two spaces rather than one, because that is what the reductor needs.
:class:`~pymor.reductors.stokes.SupremizerGalerkinStokesReductor` enriches the velocity basis with
``u_product^-1 B^T p`` for each pressure mode, which is what restores the inf-sup condition that
independently reduced velocity and pressure spaces would otherwise lose -- leaving a reduced
system that is singular, or a pressure that is noise.
"""

import numpy as np
from pymor.core.exceptions import InversionError
from pymor.models.saddle_point import SaddlePointModel
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.parameters.functionals import ConstantParameterFunctional, ProjectionParameterFunctional
from pymor.solvers.interface import Solver
from pymor.vectorarrays.block import BlockVectorSpace

from exadg.mor.binding import ExaDGOperator, ExaDGVectorSpace
from exadg.mor.models.stationary import parameter_names


class ExaDGCoupledSolver(Solver):
    """Hands the coupled system to the application's own solver.

    Attached to the block operator, so ``model.solve(mu)`` runs ExaDG's GMRES with its block
    preconditioner -- and, once the equation is Navier-Stokes, ExaDG's Newton iteration around it.

    The right-hand side is taken from the vector pyMOR passes, not recomputed from the parameter.
    That matters: ``apply_inverse`` may legitimately be asked to solve with any vector, and a
    solver that quietly answered a different question would be wrong in exactly the places nobody
    checks.
    """

    def __init__(self, fom):
        self.fom = fom

    def _solve(self, operator, V, mu, initial_guess):
        velocity_space, pressure_space = operator.source.subspaces
        velocity_rhs, pressure_rhs = V.blocks

        velocities, pressures = [], []
        for i in range(len(V)):
            result = self.fom.solve(velocity_rhs.vectors[i].impl, pressure_rhs.vectors[i].impl)

            if result is None:
                raise InversionError("the application declined to solve this system")

            u, p = result
            velocities.append(velocity_space.make_vector(u))
            pressures.append(pressure_space.make_vector(p))

        solution = operator.source.make_array(
            [velocity_space.make_array(velocities), pressure_space.make_array(pressures)]
        )

        # pyMOR's Solver contract is (solution, info); the info dict is what return_info exposes
        return solution, {}


class ExaDGSaddlePointVisualizer:
    """Writes the two fields of a block vector as two VTU records.

    Separate records rather than one, because velocity and pressure live on different DoF
    handlers -- a mixed-order pair has different polynomial degrees, so there is no single set of
    patches to write them on.
    """

    def __init__(self, directory="output/pymor"):
        self.directory = directory

    def visualize(self, U, title=None, legend=None, filename=None, block=None, **kwargs):
        base = filename or f"{self.directory}/{title or 'solution'}"
        velocity, pressure = U.blocks

        return tuple(
            array.space.impl.write_vtu(
                str(base).rsplit("/", 1)[0],
                str(base).rsplit("/", 1)[-1] + suffix,
                [array.vectors[0].impl],
                [name],
            )
            for array, suffix, name in (
                (velocity, "_velocity", "velocity"),
                (pressure, "_pressure", "pressure"),
            )
        )


def saddle_point_model(fom, parameters=None, coefficients=None, directory="output/pymor"):
    """Build the pyMOR ``SaddlePointModel`` of an ExaDG velocity/pressure model.

    Args:
        fom: Any bound ``PyMOR::SaddlePointModel``.
        parameters: Name per parameter group; see
            :func:`~exadg.mor.models.stationary.parameter_names`.
        coefficients: One ``ParameterFunctional`` per right-hand-side component. The default is
            ``ProjectionParameterFunctional``, i.e. the right-hand side is linear in its
            parameters -- which is what a body force expanded in modes with those amplitudes is.
        directory: Where the visualizer writes when it is not given a filename.

    Returns:
        Tuple ``(model, (velocity_space, pressure_space))``.
    """
    from pymor.core.logger import set_log_levels

    set_log_levels({"pymor": "WARNING"})

    velocity = ExaDGVectorSpace(fom.velocity_space(), id="VELOCITY")
    pressure = ExaDGVectorSpace(fom.pressure_space(), id="PRESSURE")

    A = ExaDGOperator(velocity, fom.momentum(), name="A")
    B = ExaDGOperator(velocity, fom.divergence(), name="B", range_space=pressure)

    shape = list(fom.parameter_shape)
    names = parameter_names(shape, parameters)

    components = fom.velocity_rhs_components()
    if coefficients is None:
        coefficients = [
            ProjectionParameterFunctional(names[c.slot], shape[c.slot], c.index)
            for c in components
        ]
    elif len(coefficients) != len(components):
        raise ValueError(
            f"the model declares {len(components)} right-hand-side components but "
            f"{len(coefficients)} coefficient functionals were given"
        )

    products = {}
    for name, impl, space in (
        ("u_product", fom.velocity_product(), velocity),
        ("p_product", fom.pressure_product(), pressure),
    ):
        products[name] = None if impl is None else ExaDGOperator(space, impl, name=name)

    return (
        SaddlePointModel(
            A,
            B,
            _velocity_rhs(velocity, fom, components, coefficients),
            g=_pressure_rhs(pressure, fom),
            visualizer=ExaDGSaddlePointVisualizer(directory=directory),
            solver=ExaDGCoupledSolver(fom),
            **products,
        ),
        (velocity, pressure),
    )


def _as_operator(space, vector):
    return VectorOperator(space.make_array([space.make_vector(vector)]))


def _velocity_rhs(space, fom, components, coefficients):
    """f = f_0 + sum_i c_i(mu) f_i, dropping f_0 when it is exactly zero.

    The constant part is the right-hand side at zero parameters: the boundary terms of the
    gradient and viscous operators. Homogeneous boundary conditions make it vanish, and this
    application's do, but that is checked here rather than assumed -- a lifted Dirichlet
    condition would put a non-zero constant term in exactly this place.
    """
    constant = fom.velocity_rhs()

    operators = [_as_operator(space, component.vector) for component in components]
    functionals = list(coefficients)

    if constant is not None and constant.norm() != 0.0:
        operators.insert(0, _as_operator(space, constant))
        functionals.insert(0, ConstantParameterFunctional(1.0))

    return LincombOperator(operators, functionals)


def _pressure_rhs(space, fom):
    """g, or None when it is zero -- pyMOR then builds the zero vector itself."""
    pressure_rhs = fom.pressure_rhs()

    if pressure_rhs is None or pressure_rhs.norm() == 0.0:
        return None

    return _as_operator(space, pressure_rhs)
