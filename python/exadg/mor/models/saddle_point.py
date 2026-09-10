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

from pathlib import Path

from pymor.core.exceptions import InversionError
from pymor.models.saddle_point import SaddlePointModel
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.parameters.functionals import ConstantParameterFunctional, ProjectionParameterFunctional
from pymor.solvers.interface import Solver
from pymor.vectorarrays.interface import VectorArray

from exadg.mor.binding import ExaDGOperator, ExaDGVectorSpace, level_times, write_pvd_record
from exadg.mor.models.stationary import parameter_names

#: The step this module is on. ``interface.h`` takes ``(mass_scaling, time)`` at every method
#: that evaluates or solves the momentum equation, because a transient model steps in time and a
#: steady one does not -- and it never defaults them, since a silent ``0.0`` would mean "steady".
#: This factory builds pyMOR's *stationary* saddle point, so the answer is the same everywhere and
#: is written down once. An instationary model is a different factory and passes the step it is on.
STEADY = dict(mass_scaling=0.0, time=0.0)


class ExaDGNonlinearMomentum(Operator):
    """A(u), the momentum block of a Navier-Stokes system.

    Nonlinear in the velocity through the convective term, which is what makes the whole model
    nonlinear -- B stays linear, so the Jacobian of the block system is [[A'(u), B*], [B, 0]] and
    only this block changes with the state.

    A(u) is evaluated as N(u, p = 0). The pressure enters the momentum equation only through
    B* p, so dropping it leaves exactly the momentum operator; checked at 3e-17 relative rather
    than assumed, since the whole point of handing pyMOR A and B separately is that it assembles
    the same residual ExaDG solves.
    """

    linear = False

    def __init__(self, space, pressure_space, fom, name="A(u)"):
        self.space = space
        self.pressure_space = pressure_space
        self.fom = fom
        self.name = name

        self.source = self.range = space
        self.parameters_own = {}

    def apply(self, U, mu=None):
        assert U in self.source

        zero_pressure = self.pressure_space.impl.zero_vector()

        return self.range.make_array([
            self.range.make_vector(self.fom.apply_nonlinear(u.impl, zero_pressure, **STEADY)[0])
            for u in U.vectors
        ])

    def jacobian(self, U, mu=None):
        """A'(u), which pyMOR's Newton iteration asks for at each step.

        Not the exact derivative of :meth:`apply`, and the cause is the Lax-Friedrichs term
        alone. Its lambda is a maximum of absolute normal velocities, which is not
        differentiable, so ExaDG freezes it at the linearisation point rather than differentiate
        it. Measured: a finite difference agrees to 4.5e-08 at upwind_factor = 0 and only to
        5.1e-03 at upwind_factor = 1. A Newton iteration on it converges linearly rather than
        quadratically, which costs iterations and nothing else -- the solution is defined by the
        residual.
        """
        assert len(U) == 1

        return ExaDGOperator(
            self.space,
            self.fom.jacobian_momentum(U.vectors[0].impl, STEADY["mass_scaling"]),
            name="A'(u)",
        )


def _solve_blocks(fom, velocity_space, pressure_space, f, g):
    """Solve the coupled system once per right-hand side, returning the two blocks.

    Shared by the serial and the MPI solver below. Those two differ only in where the model comes
    from and whether the call is dispatched to every rank -- never in what is solved, so what is
    solved is written once.
    """
    velocities, pressures = [], []
    for i in range(len(f)):
        result = fom.solve(f.vectors[i].impl, g.vectors[i].impl, **STEADY)

        if result is None:
            raise InversionError("the application declined to solve this system")

        u, p = result
        velocities.append(velocity_space.make_vector(u))
        pressures.append(pressure_space.make_vector(p))

    return velocity_space.make_array(velocities), pressure_space.make_array(pressures)


def _visualize_arguments(U, title, legend, filename, directory):
    """The part of visualize() that does not depend on how the write is dispatched.

    Returns ``(arrays, names, base, levels)``: the fields to write, one name each, the output path
    without its suffix, and how many time levels each field has.

    One level is one record; several are a *series* -- one record per level plus a ``.pvd``
    collection -- because they are the same field at different times rather than different fields.
    """
    # Any sequence of arrays, or one array on its own. A VectorArray is itself iterable over its
    # vectors, so it has to be recognised before the sequence case rather than after it.
    arrays = (U,) if isinstance(U, VectorArray) else tuple(U)

    levels = {len(array) for array in arrays}
    if len(levels) != 1:
        raise ValueError(
            f"all fields must have the same number of time levels, got {sorted(levels)}."
        )

    names = [
        legend[i] if legend is not None and not isinstance(legend, str) else f"field_{i}"
        for i in range(len(arrays))
    ]
    base = Path(filename) if filename else Path(directory) / (title or "solution")

    return arrays, names, base, levels.pop()


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
        f, g = V.blocks

        blocks = _solve_blocks(self.fom, velocity_space, pressure_space, f, g)

        # pyMOR's Solver contract is (solution, info); the info dict is what return_info exposes
        return operator.source.make_array(list(blocks)), {}


class ExaDGSaddlePointVisualizer:
    """Writes the two fields of a block vector as two VTU records.

    Separate records rather than one, because velocity and pressure live on different DoF
    handlers -- a mixed-order pair has different polynomial degrees, so there is no single set of
    patches to write them on.

    Takes a tuple of arrays the way :class:`~exadg.mor.binding.ExaDGVisualizer` does, so the usual
    three-way comparison works and produces two records::

        model.visualize((U_fom, U_rom, U_fom - U_rom),
                        legend=("fom", "rom", "error"), filename="output/compare")
    """

    def __init__(self, directory="output/pymor"):
        self.directory = directory

    def visualize(self, U, title=None, legend=None, filename=None, block=None, times=None,
                  **kwargs):
        """Write the fields, as two records or as two time series.

        Args:
            times: One time per level, for a series. Defaults to the level index.
        """
        arrays, names, base, levels = _visualize_arguments(
            U, title, legend, filename, self.directory
        )
        moments = level_times(levels, times)

        written = []
        for position, suffix in enumerate(("velocity", "pressure")):
            blocks = [array.blocks[position] for array in arrays]
            space = blocks[0].space.impl

            if levels == 1:
                written.append(
                    space.write_vtu(
                        str(base.parent), f"{base.name}_{suffix}",
                        [array.vectors[0].impl for array in blocks], names,
                    )
                )
                continue

            entries = [
                (
                    time,
                    space.write_vtu(
                        str(base.parent), f"{base.name}_{suffix}_t{level:04d}",
                        [array.vectors[level].impl for array in blocks], names,
                    ),
                )
                for level, time in enumerate(moments)
            ]
            written.append(write_pvd_record(str(base.parent / f"{base.name}_{suffix}"), entries))

        return tuple(written)


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
    velocity = ExaDGVectorSpace(fom.velocity_space(), id="VELOCITY")
    pressure = ExaDGVectorSpace(fom.pressure_space(), id="PRESSURE")

    if fom.is_nonlinear:
        A = ExaDGNonlinearMomentum(velocity, pressure, fom)
    else:
        A = ExaDGOperator(velocity, fom.momentum(STEADY["mass_scaling"]), name="A")
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
    """f = f_0 + sum_i c_i(mu) f_i, with the constant part f_0 only where it belongs.

    f_0 is the right-hand side at zero parameters: the inhomogeneous boundary terms of the
    gradient and viscous operators, moved to the right. Whether it belongs on the right depends
    on which form of A the model exposes, and the two branches genuinely differ.

    ``momentum()`` is ExaDG's ``vmult``, the *homogeneous* operator, so those terms are not in it
    and have to appear here. ``apply_nonlinear()`` is ExaDG's residual, assembled from
    ``evaluate()``, which already carries them -- adding f_0 as well would count them twice, and
    the reduced model would converge to a solution ExaDG's own Newton does not.

    Both are zero under homogeneous boundary conditions, which is what every application here has
    and what the norm check below detects. The branch costs nothing and removes a defect that
    would otherwise surface the first time somebody lifts a Dirichlet condition.
    """
    constant = None if fom.is_nonlinear else fom.velocity_rhs()

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


def _local_coupled_solve(model, f, g):
    """Solve on every rank. Called through mpi.call, so the arguments arrive as local objects."""
    velocity_space, pressure_space = model.operator.source.subspaces

    return _solve_blocks(model.operator.solver.fom, velocity_space, pressure_space, f, g)


def _take(pair, index):
    """Split the solve's result, so each block can be managed as its own MPI object."""
    return pair[index]


class MPIExaDGCoupledSolver(Solver):
    """The coupled solve, dispatched to every rank.

    :class:`ExaDGCoupledSolver` holds one model, which is rank 0's; calling it in parallel would
    enter ExaDG's GMRES on rank 0 alone and hang, since that solve is collective. This one holds
    the :class:`~pymor.tools.mpi.ObjectId` of the local models instead and drives them all through
    ``mpi.call``.

    The solve happens once and returns both blocks; the two follow-up calls only split that pair,
    because an MPIVectorArray needs an ObjectId of its own.
    """

    def __init__(self, models_id):
        self.models_id = models_id

    def _solve(self, operator, V, mu, initial_guess):
        from pymor.tools import mpi

        f, g = V.blocks
        pair = mpi.call(
            mpi.function_call_manage, _local_coupled_solve, self.models_id,
            f.impl.obj_id, g.impl.obj_id,
        )

        velocity_space, pressure_space = operator.source.subspaces

        solution = operator.source.make_array([
            velocity_space.make_array(mpi.call(mpi.function_call_manage, _take, pair, 0)),
            pressure_space.make_array(mpi.call(mpi.function_call_manage, _take, pair, 1)),
        ])

        return solution, {}


def _local_write_block(model, position, array_ids, directory, basename, names, level=0):
    """Write one block's fields at one time level, on every rank. The pvtu record is collective.

    The ids are resolved here rather than by mpi.function_call, which only maps arguments that
    are themselves ObjectIds and not lists of them.
    """
    from pymor.tools import mpi

    space = model.operator.source.subspaces[position]
    arrays = [mpi.get_object(array_id) for array_id in array_ids]

    return space.impl.write_vtu(
        directory, basename, [array.vectors[level].impl for array in arrays], names
    )


class MPIExaDGSaddlePointVisualizer:
    """The block visualizer, dispatched to every rank.

    pyMOR's own MPIVisualizer cannot wrap this one: it assumes each array is an MPIVectorArray and
    reads ``u.impl.obj_id``, which a BlockVectorArray does not have. Here the blocks are taken
    apart first and each is dispatched on its own, which is what deal.II wants anyway -- velocity
    and pressure go to separate records.
    """

    def __init__(self, models_id, directory="output/pymor"):
        self.models_id = models_id
        self.directory = directory

    def visualize(self, U, title=None, legend=None, filename=None, block=None, times=None,
                  **kwargs):
        from pymor.tools import mpi

        arrays, names, base, levels = _visualize_arguments(
            U, title, legend, filename, self.directory
        )
        moments = level_times(levels, times)
        ids = {
            position: [array.blocks[position].impl.obj_id for array in arrays]
            for position in (0, 1)
        }

        def record(position, basename, level):
            return mpi.call(
                mpi.function_call, _local_write_block, self.models_id, position, ids[position],
                str(base.parent), basename, names, level,
            )

        written = []
        for position, suffix in enumerate(("velocity", "pressure")):
            if levels == 1:
                written.append(record(position, f"{base.name}_{suffix}", 0))
                continue

            entries = [
                (time, record(position, f"{base.name}_{suffix}_t{level:04d}", level))
                for level, time in enumerate(moments)
            ]

            # Each record above is collective; the collection is one small file and is written
            # here, which under pyMOR's event loop is rank 0 alone.
            written.append(write_pvd_record(str(base.parent / f"{base.name}_{suffix}"), entries))

        return tuple(written)


def exadg_model(model):
    """The bound ExaDG model behind a *rank-local* pyMOR model built by this package.

    A pyMOR model is meant to hide what discretised it, so there is no general route back. The
    route here is a solver, because a coupled solve is a method on the application rather than an
    operator's inverse, and a solver is where a pyMOR model is allowed to keep one.

    **Which** solver depends on what is being inverted, and the two model shapes differ:

    - stationary: the block operator itself, so ``model.operator.solver``;
    - instationary: the *step*, ``gamma/dt M + A``, which the time stepper assembles and owns --
      the block operator has no solver at all, so ``model.time_stepper.solver``.

    Both are checked here rather than at each call site, so a reductor works on either shape and a
    model built some other way fails with a sentence rather than an ``AttributeError`` three
    frames down.

    Under MPI this is the *local* model on whichever rank is asking; see :func:`exadg_models_id`
    for the handle that addresses all of them.
    """
    return _handle(model, "fom", "a local ExaDG model")


def exadg_models_id(model):
    """The ``ObjectId`` addressing every rank's ExaDG model, from the MPI-wrapped one.

    A different object from :func:`exadg_model`, not a parallel spelling of it: that one answers
    "which model is on this rank", this one answers "how do I reach all of them at once". Only
    the MPI solvers carry it, because only they need to dispatch.
    """
    return _handle(model, "models_id", "an MPI handle to the ExaDG models")


def _handle(model, attribute, what):
    solvers = [
        getattr(getattr(model, owner, None), "solver", None)
        for owner in ("operator", "time_stepper")
    ]

    for solver in solvers:
        handle = getattr(solver, attribute, None)
        if handle is not None:
            return handle

    found = " and ".join(type(solver).__name__ for solver in solvers)
    raise TypeError(
        f"{model.name} does not carry {what}: the solvers on its operator and time stepper are "
        f"{found}, so it was not built by exadg.mor.models."
    )


def _build_saddle_point_model(module_name, class_name, args, kwargs, model_kwargs):
    """Construct the per-rank model. Module level so that it survives pickling to the ranks."""
    import importlib

    module = importlib.import_module(f"exadg.{module_name}")
    fom = getattr(module, class_name)(*args, **kwargs)

    return saddle_point_model(fom, **model_kwargs)[0]


def mpi_saddle_point_model(module_name, class_name, *args, **kwargs):
    """Build the model, wrapped for MPI when the interpreter is running in parallel.

    The counterpart of :func:`~exadg.mor.models.stationary.mpi_stationary_model`, and it exists
    for the same reason: the full-order model has to be constructed *on every rank*, because each
    owns a piece of the mesh and deal.II partitions the triangulation collectively. Called on rank
    0 alone while the others wait in pyMOR's event loop, the constructor hangs with no output.

    Run it as ::

        mpirun -n 4 python -m pymor.tools.mpi reduce.py

    Args:
        module_name: Application module inside the ``exadg`` package, e.g. ``"forced"``.
        class_name: Model class in it, e.g. ``"ForcedFOM2D"``.
        *args: Passed to that class.
        **kwargs: Passed to that class, except ``parameters``, ``coefficients`` and
            ``directory``, which go to :func:`saddle_point_model`.

    Returns:
        Tuple ``(model, (velocity_space, pressure_space))``.
    """
    import functools

    from pymor.tools import mpi

    model_kwargs = {
        key: kwargs.pop(key)
        for key in ("parameters", "coefficients", "directory")
        if key in kwargs
    }

    factory = functools.partial(
        _build_saddle_point_model, module_name, class_name, args, kwargs, model_kwargs
    )

    if not mpi.parallel:
        model = factory()

        return model, model.solution_space.subspaces

    from pymor.models.mpi import mpi_wrap_model

    # The models are managed first so that the solver below can address them: mpi_wrap_model
    # accepts the ObjectId as well as a factory, and only the former is reusable afterwards.
    models_id = mpi.call(mpi.function_call_manage, factory)

    model = mpi_wrap_model(
        models_id,
        mpi_spaces=(ExaDGVectorSpace,),
        use_with=True,
        pickle_local_spaces=False,
    )

    model = model.with_(
        solver=MPIExaDGCoupledSolver(models_id),
        visualizer=MPIExaDGSaddlePointVisualizer(
            models_id, directory=model_kwargs.get("directory", "output/pymor")
        ),
    )

    return model, model.solution_space.subspaces
