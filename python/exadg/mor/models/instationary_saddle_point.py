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

"""pyMOR ``InstationaryModel`` over an ExaDG velocity/pressure model that carries a mass term.

    [ M  0 ] d  [u]     [ A   B* ] [u]   [f]
    [ 0  0 ] dt [p]  +  [ B   0  ] [p] = [g]

A differential-algebraic system, not an ordinary differential equation: the pressure has no time
derivative, because it is a Lagrange multiplier for the constraint rather than a state. Everything
below follows from that.

The counterpart of :mod:`~exadg.mor.models.saddle_point`, and built the same way -- pyMOR's
``SaddlePointModel`` is a ``StationaryModel`` with a ``BlockOperator`` and saddle-point accessors
bolted on, and this is an ``InstationaryModel`` with the same. That parallel is the reason for
the design: everything ``ProjectionBasedReductor`` knows how to do, everything ``mpi_wrap_model``
knows how to wrap, and everything ``reduction_error_analysis`` knows how to measure keeps working,
and what is saddle-point-specific stays in one class.

**The loop is here; the step is ExaDG's.** ``interface.h`` carries one implicitly discretised step
and nothing about time integration, so a reduced model and the full-order model it is measured
against run the *same* scheme -- which they must, or the comparison measures two discretisations.
:class:`BDFTimeStepper` is that scheme, and it drives a reduced model through pyMOR's Newton and a
full-order one through :class:`ExaDGStepSolver` with no other difference.
"""

import numpy as np
from pymor.algorithms.timestepping import TimeStepper
from pymor.core.exceptions import InversionError
from pymor.models.basic import InstationaryModel
from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockOperator
from pymor.operators.constructions import (
    AdjointOperator,
    IdentityOperator,
    LincombOperator,
    VectorOperator,
    ZeroOperator,
)
from pymor.operators.interface import Operator
from pymor.solvers.interface import Solver

from exadg.mor.binding import ExaDGOperator, ExaDGVectorSpace
from exadg.mor.models.saddle_point import (
    ExaDGNonlinearMomentum,
    ExaDGSaddlePointVisualizer,
    MPIExaDGSaddlePointVisualizer,
    exadg_model,
    exadg_models_id,
    install_coefficients,
    local_coefficient_names,
)
from exadg.mor.models.stationary import parameter_names


def bdf_coefficients(order):
    """Coefficients of the backward differentiation formula of the given order.

    Returns ``(gamma, alpha)`` for

        gamma/dt * u^{n+1}  =  sum_i alpha_i/dt * u^{n-i}  +  F(u^{n+1}),

    which is ExaDG's convention: ``gamma`` is what multiplies the mass matrix in the step operator
    and ``alpha`` weights the history that goes into the right-hand side.

    Derived rather than tabulated. BDF-p is ``sum_{j=1..p} (1/j) nabla^j u^{n+1} = dt F``, and
    expanding the backward differences gives ``c_i = sum_{j>=i} (1/j) (-1)^i binom(j, i)``, so
    ``gamma = c_0`` and ``alpha_i = -c_{i+1}``. Checked against the textbook values: order 1 gives
    (1, [1]), order 2 gives (3/2, [2, -1/2]), order 3 gives (11/6, [3, -3/2, 1/3]).

    Only 1 and 2 are unconditionally stable. BDF-3 is A(86 degrees)-stable rather than A-stable,
    which is usually harmless and is not always -- hence the warning rather than a refusal.
    """
    if order < 1:
        raise ValueError(f"BDF order must be at least 1, got {order}")

    from math import comb

    c = np.array([
        sum((-1) ** i * comb(j, i) / j for j in range(max(i, 1), order + 1))
        for i in range(order + 1)
    ])

    return c[0], -c[1:]


class BDFTimeStepper(TimeStepper):
    """Backward differentiation of order ``order``, starting from rest with lower orders.

    Solves ``M du/dt + A(u, t) = F(t)`` as a sequence of steps

        gamma/dt * M u^{n+1} + A(u^{n+1}) = M sum_i alpha_i/dt u^{n-i} + F(t^{n+1}),

    each of which is exactly the operator ``interface.h`` carries -- the steady problem plus a
    mass term, with the history in the right-hand side. The step operator is assembled here, as
    ``LincombOperator([mass, A], [gamma/dt, 1])``, so the same expression serves a full-order
    model and a reduced one and only the solver differs.

    **Order ramps up over the first steps**, because BDF-p needs p previous levels and at
    ``t = 0`` there is one. Step n therefore uses order ``min(order, n + 1)``: the first step is
    implicit Euler whatever ``order`` says. That costs one step of first-order error, which for a
    fixed order is O(dt) in a single step and does not degrade the global rate.

    **Why order matters here more than usual.** For a degree-k discontinuous Galerkin velocity the
    spatial error is O(h^(k+1)), so keeping the time error subordinate needs dt <~ h^((k+1)/p).
    At k = 2 and h = 1/16 that is dt <~ 2e-4 for BDF-1 and 2e-2 for BDF-2 -- forty thousand steps
    against six hundred. BDF-1 is a startup and debugging scheme; it is not a scheme to produce
    snapshots with.

    Args:
        nt: Number of steps over the interval.
        order: BDF order. 2 is the default and the lowest that is usually affordable.
        solver: :class:`~pymor.solvers.interface.Solver` for each step, or ``None`` to let the
            step operator resolve its own -- which for a reduced model is pyMOR's Newton.
    """

    def __init__(self, nt, order=2, solver=None):
        assert nt >= 1
        assert order >= 1

        if order > 2:
            self.logger.warning(
                f"BDF-{order} is not A-stable; only orders 1 and 2 are. Fine for a mildly stiff "
                f"problem, not something to adopt without measuring."
            )

        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(
        self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None,
        num_values=None,
    ):
        """Step from ``initial_time`` to ``end_time``, yielding ``(U, t)`` per stored level."""
        from pymor.parameters.base import Mu

        assert operator.source == operator.range
        assert len(initial_data) == 1

        if mu is None:
            mu = Mu()
        if mass is None:
            mass = IdentityOperator(operator.source)

        nt = self.nt
        dt = (end_time - initial_time) / nt

        num_values = num_values or nt + 1
        stride = (end_time - initial_time) / (num_values - 1)

        # The history, newest first. Only `order` levels are ever needed, so it is trimmed rather
        # than kept -- a trajectory of full-order block vectors is the largest thing in the room.
        history = [initial_data.copy()]

        t = initial_time
        stored = 1
        yield initial_data, t

        for n in range(nt):
            t += dt
            mu_t = mu.at_time(t)

            order = min(self.order, n + 1)
            gamma, alpha = bdf_coefficients(order)

            # M sum_i alpha_i/dt u^{n-i}: the pressure block of `mass` is zero, so this is the
            # velocity history alone and the pressure carries no memory -- which is the whole
            # content of "the pressure is a multiplier, not a state".
            weighted = history[0] * (alpha[0] / dt)
            for i in range(1, order):
                weighted.axpy(alpha[i] / dt, history[i])

            step_rhs = mass.apply(weighted, mu=mu_t)
            if rhs is not None:
                step_rhs += rhs.as_range_array(mu_t)

            step_operator = LincombOperator([mass, operator], [gamma / dt, 1.0])

            solver = None if self.solver is None else self.solver.at_step(gamma / dt, t)
            U = step_operator.apply_inverse(
                step_rhs, mu=mu_t, initial_guess=history[0], solver=solver
            )

            history.insert(0, U)
            del history[self.order:]

            while stored * stride <= t - initial_time + 0.5 * min(dt, stride):
                stored += 1
                yield U, t


def _velocity_of(U):
    """The velocity block of a block array, or the array itself if it has no blocks."""
    return U.blocks[0] if hasattr(U, "blocks") else U


class ExaDGStepSolver(Solver):
    """Hands one implicit step to ExaDG, at the mass scaling the stepper chose.

    Attached to the step operator by :class:`BDFTimeStepper` rather than to the model, because
    ``gamma/dt`` is a property of the step and not of the problem. ``at_step`` is what the stepper
    calls; the scaling therefore travels one way, from the object that computed it, and there is
    no second place that has to agree about ``dt``.

    The right-hand side is taken from the vector pyMOR passes, never recomputed from the
    parameter: it carries the history, and a solver that rebuilt it from ``mu`` would silently
    solve the first step over and over.
    """

    def __init__(self, fom, mass_scaling=0.0, time=0.0):
        self.__auto_init(locals())

    def at_step(self, mass_scaling, time):
        """This solver, fixed to one step. Immutable, so it is a copy rather than a mutation."""
        return self.with_(mass_scaling=mass_scaling, time=time)

    def _solve(self, operator, V, mu, initial_guess):
        velocity_space, pressure_space = operator.source.subspaces
        f, g = V.blocks

        guess = None if initial_guess is None else _velocity_of(initial_guess).vectors[0].impl

        # The parameter does reach the application after all, but only the part of it the
        # application owns: a coefficient of the operator, which its Newton iteration reads from
        # its own objects. The right-hand side is still taken from the vector pyMOR passes.
        install_coefficients(self.fom, mu)

        velocities, pressures = [], []
        for i in range(len(f)):
            result = self.fom.solve(
                f.vectors[i].impl, g.vectors[i].impl, self.mass_scaling, self.time, guess
            )
            if result is None:
                raise InversionError(
                    f"the application declined to solve the step at t = {self.time}"
                )

            u, p = result
            velocities.append(velocity_space.make_vector(u))
            pressures.append(pressure_space.make_vector(p))

        return operator.source.make_array([
            velocity_space.make_array(velocities), pressure_space.make_array(pressures)
        ]), {}


def _local_time_step_for_cfl(model, cfl):
    """Ask one rank's model. The criterion reduces over the communicator, so all must ask."""
    return exadg_model(model).time_step_for_cfl(cfl)


def time_step_for_cfl(model, cfl):
    """The time step this discretisation admits at that CFL number.

    ExaDG's own criterion, read off the mesh -- see ``ForcedFOM::time_step_for_cfl``. Collective,
    because the minimum is taken over every element and therefore over every rank, so it is
    dispatched rather than asked of rank 0 alone.

    A step count is what a time stepper wants, and deriving it from this rather than fixing it is
    what keeps a refinement study honest: a count that does not follow the mesh silently changes
    the CFL number when the mesh changes, and then two things are varying at once.
    """
    from pymor.tools import mpi

    if not mpi.parallel:
        return _local_time_step_for_cfl(model, cfl)

    return mpi.call(mpi.function_call, _local_time_step_for_cfl, exadg_models_id(model), cfl)


def steps_for_cfl(model, cfl, T):
    """Steps over ``[0, T]`` at that CFL number, rounded up so the last one lands on ``T``.

    The rounding is ``adjust_time_step_to_hit_end_time`` from ExaDG's time integrator, which is
    what it does with the same number.
    """
    from math import ceil

    return max(1, ceil(T / time_step_for_cfl(model, cfl)))


def _local_step_solve(model, f, g, mass_scaling, time, guess, coefficients):
    """Solve one step on every rank. Called through mpi.call, so arguments arrive as objects."""
    fom = exadg_model(model)
    velocity_space, pressure_space = model.operator.source.subspaces

    for name, value in coefficients.items():
        fom.set_coefficient(name, value)

    velocities, pressures = [], []
    for i in range(len(f)):
        result = fom.solve(
            f.vectors[i].impl, g.vectors[i].impl, mass_scaling, time,
            None if guess is None else _velocity_of(guess).vectors[i].impl,
        )
        if result is None:
            return None

        u, p = result
        velocities.append(velocity_space.make_vector(u))
        pressures.append(pressure_space.make_vector(p))

    return velocity_space.make_array(velocities), pressure_space.make_array(pressures)


def _take(pair, index):
    """Split the solve's result, so each block can be managed as its own MPI object."""
    return pair[index]


class MPIExaDGStepSolver(Solver):
    """The step, dispatched to every rank.

    The counterpart of :class:`ExaDGStepSolver` and it exists for the same reason
    :class:`~exadg.mor.models.saddle_point.MPIExaDGCoupledSolver` does: ExaDG's solve is
    collective, so entering it on rank 0 alone hangs.
    """

    def __init__(self, models_id, coefficients=(), mass_scaling=0.0, time=0.0):
        self.__auto_init(locals())

    def at_step(self, mass_scaling, time):
        return self.with_(mass_scaling=mass_scaling, time=time)

    def _solve(self, operator, V, mu, initial_guess):
        from pymor.tools import mpi

        f, g = V.blocks

        # Resolved here rather than inside the call: mu is rank 0's, and the names come from the
        # model, so what crosses to the other ranks is a plain dict of numbers.
        coefficients = {} if mu is None else {
            name: float(mu[name][0]) for name in self.coefficients if name in mu
        }

        pair = mpi.call(
            mpi.function_call_manage, _local_step_solve, self.models_id,
            f.impl.obj_id, g.impl.obj_id, self.mass_scaling, self.time,
            None if initial_guess is None else _velocity_of(initial_guess).impl.obj_id,
            coefficients,
        )

        velocity_space, pressure_space = operator.source.subspaces

        return operator.source.make_array([
            velocity_space.make_array(mpi.call(mpi.function_call_manage, _take, pair, 0)),
            pressure_space.make_array(mpi.call(mpi.function_call_manage, _take, pair, 1)),
        ]), {}


class InstationarySaddlePointModel(InstationaryModel):
    """A velocity/pressure problem with a mass term, as pyMOR's ``InstationaryModel`` wants it.

    Args:
        A: The (1,1) block. Linear for Stokes, an
            :class:`~exadg.mor.models.saddle_point.ExaDGNonlinearMomentum` for Navier-Stokes.
        B: The (2,1) block; its adjoint is the (1,2) block, as in the stationary model.
        velocity_mass: The velocity mass matrix -- ``velocity_mass()``, which the application
            declares rather than letting anything infer it from ``velocity_product()``. Named for
            the block it occupies and *not* ``mass``, which is a ``InstationaryModel`` argument:
            ``__auto_init`` assigns only attributes that are not already set, so a subclass
            argument of the same name wins silently and the base class's value never lands.
        f, g: Right-hand sides, as operators with a scalar source.
        initial_velocity: ``u(0)``. A one-element ``VectorArray`` on the velocity space, or a
            ``VectorOperator`` over one. Stored as the operator whichever is given, because
            ``mpi_wrap_model`` wraps an ``Operator`` and passes a ``VectorArray`` through
            untouched -- so a bare array would leave the parallel model starting from rank 0's
            slice of the initial condition.
        T: End time; the interval is ``[0, T]``.
        time_stepper: Usually a :class:`BDFTimeStepper`.

    The **initial pressure is not an argument**, and the zero vector is used. That is not a
    simplification: the mass operator's pressure block is zero, so the stepper's ``M u^n`` term
    annihilates whatever is there, and the pressure at every level -- including the first -- is
    determined by the constraint inside that step's solve. Asking a caller for a quantity that
    cannot affect the answer would invite them to compute one.
    """

    def __init__(
        self, A, B, velocity_mass, f, g, initial_velocity, T, time_stepper,
        u_product=None, p_product=None, num_values=None, error_estimator=None,
        visualizer=None, name=None,
    ):
        velocity, pressure = A.source, B.range

        assert A.source == A.range == B.source
        assert velocity_mass.source == velocity_mass.range == velocity

        if not isinstance(initial_velocity, Operator):
            assert len(initial_velocity) == 1 and initial_velocity in velocity
            initial_velocity = VectorOperator(initial_velocity, name="u_0")

        assert initial_velocity.range == velocity and initial_velocity.source.is_scalar

        operator = BlockOperator([[A, AdjointOperator(B)], [B, None]])
        block_mass = BlockDiagonalOperator([velocity_mass, ZeroOperator(pressure, pressure)])

        rhs = BlockColumnOperator(
            [f, ZeroOperator(pressure, f.source) if g is None else g], name="rhs"
        )

        initial_data = BlockColumnOperator(
            [initial_velocity, VectorOperator(pressure.zeros(1))], name="initial_data"
        )

        products = {
            "mixed": BlockDiagonalOperator([
                u_product if u_product else IdentityOperator(velocity),
                p_product if p_product else IdentityOperator(pressure),
            ])
        }

        self.__auto_init(locals())
        super().__init__(
            T=T, initial_data=initial_data, operator=operator, rhs=rhs, mass=block_mass,
            time_stepper=time_stepper, num_values=num_values, products=products,
            error_estimator=error_estimator, visualizer=visualizer, name=name,
        )


def instationary_saddle_point_model(
    fom, T, nt, order=2, parameters=None, coefficients=None, initial_velocity=None,
    num_values=None, directory="output/pymor", solver=None,
):
    """Build the pyMOR ``InstationaryModel`` of an ExaDG velocity/pressure model.

    The instationary counterpart of
    :func:`~exadg.mor.models.saddle_point.saddle_point_model`, and it reads the same declared
    structure. What it needs beyond that is the time interval, the number of steps and the BDF
    order -- none of which a discretisation can know.

    Args:
        fom: Any bound ``PyMOR::SaddlePointModel`` whose ``velocity_mass()`` is not ``None``.
        T: End time.
        nt: Number of steps.
        order: BDF order; see :class:`BDFTimeStepper`.
        initial_velocity: ``u(0)``. Defaults to rest.
        solver: The step solver. Defaults to :class:`ExaDGStepSolver` over ``fom``.

    Returns:
        Tuple ``(model, (velocity_space, pressure_space))``.
    """
    from pymor.parameters.functionals import ProjectionParameterFunctional

    if fom.velocity_mass() is None:
        raise ValueError(
            "this model declares no velocity_mass(), so it has no operator on du/dt; see "
            "PyMOR::SaddlePointModel::velocity_mass"
        )

    velocity = ExaDGVectorSpace(fom.velocity_space(), id="VELOCITY")
    pressure = ExaDGVectorSpace(fom.pressure_space(), id="PRESSURE")

    A = (
        ExaDGNonlinearMomentum(velocity, pressure, fom)
        if fom.is_nonlinear
        else ExaDGOperator(velocity, fom.momentum(0.0), name="A")
    )
    B = ExaDGOperator(velocity, fom.divergence(), name="B", range_space=pressure)
    mass = ExaDGOperator(velocity, fom.velocity_mass(), name="M")

    shape = list(fom.parameter_shape)
    names = parameter_names(shape, parameters)

    components = fom.velocity_rhs_components()
    if coefficients is None:
        coefficients = [
            ProjectionParameterFunctional(names[c.slot], shape[c.slot], c.index)
            for c in components
        ]

    # A model may have no body force at all -- a flow driven through an inhomogeneous Dirichlet
    # boundary has none, and on a discontinuous space that boundary data is already inside the
    # residual rather than beside it. An empty LincombOperator is not a zero operator.
    if components:
        f = LincombOperator(
            [VectorOperator(velocity.make_array([velocity.make_vector(c.vector)]))
             for c in components],
            coefficients,
            name="f",
        )
    else:
        f = VectorOperator(velocity.zeros(1), name="f")

    return (
        InstationarySaddlePointModel(
            A=A,
            B=B,
            velocity_mass=mass,
            f=f,
            g=None,
            initial_velocity=velocity.zeros(1) if initial_velocity is None else initial_velocity,
            T=T,
            time_stepper=BDFTimeStepper(
                nt, order=order, solver=ExaDGStepSolver(fom) if solver is None else solver
            ),
            u_product=ExaDGOperator(velocity, fom.velocity_product(), name="u_product"),
            p_product=ExaDGOperator(pressure, fom.pressure_product(), name="p_product"),
            num_values=num_values,
            visualizer=ExaDGSaddlePointVisualizer(directory=directory),
        ),
        (velocity, pressure),
    )


def _build_instationary_model(module_name, class_name, args, kwargs, model_kwargs):
    """Construct the per-rank model. Module level so that it survives pickling to the ranks."""
    import importlib

    module = importlib.import_module(f"exadg.{module_name}")
    fom = getattr(module, class_name)(*args, **kwargs)

    return instationary_saddle_point_model(fom, **model_kwargs)[0]


def mpi_instationary_saddle_point_model(module_name, class_name, *args, **kwargs):
    """Build the transient model, wrapped for MPI when the interpreter is running in parallel.

    The counterpart of :func:`~exadg.mor.models.saddle_point.mpi_saddle_point_model`, and it
    exists for the same reason: the full-order model has to be constructed on *every* rank, so
    what is shipped is a picklable recipe naming the application by string.

    Two of the model's parts need explicit parallel counterparts, because ``mpi_wrap_model`` wraps
    the model's ``Operator`` arguments and passes everything else through as rank 0's. The step
    solver holds a model rather than an operator -- ExaDG's coupled solve is a method -- so it is
    replaced with :class:`MPIExaDGStepSolver`; and pyMOR's own ``MPIVisualizer`` reads
    ``u.impl.obj_id``, which a block array has not got.

    The initial velocity *is* wrapped, and only because it is stored as a ``VectorOperator``
    rather than as the array it is usually given as -- see :class:`InstationarySaddlePointModel`.

    Args:
        module_name: Application module inside the ``exadg`` package, e.g. ``"forced"``.
        class_name: Model class in it, e.g. ``"ForcedFOM2D"``.
        *args, **kwargs: Passed to that class, except ``T``, ``nt``, ``order``, ``parameters``,
            ``coefficients``, ``initial_velocity``, ``num_values`` and ``directory``, which go to
            :func:`instationary_saddle_point_model`.

    Returns:
        Tuple ``(model, (velocity_space, pressure_space))``.
    """
    import functools

    from pymor.tools import mpi

    model_kwargs = {
        key: kwargs.pop(key)
        for key in ("T", "nt", "order", "parameters", "coefficients", "initial_velocity",
                    "num_values", "directory")
        if key in kwargs
    }

    factory = functools.partial(
        _build_instationary_model, module_name, class_name, args, kwargs, model_kwargs
    )

    if not mpi.parallel:
        model = factory()

        return model, model.solution_space.subspaces

    from pymor.models.mpi import mpi_wrap_model

    models_id = mpi.call(mpi.function_call_manage, factory)

    model = mpi_wrap_model(
        models_id,
        mpi_spaces=(ExaDGVectorSpace,),
        use_with=True,
        pickle_local_spaces=False,
    )

    stepper = model.time_stepper
    model = model.with_(
        time_stepper=BDFTimeStepper(
            stepper.nt,
            order=stepper.order,
            solver=MPIExaDGStepSolver(
                models_id,
                # Asked of the application rather than read off the wrapped operator, whose
                # blocks are no longer reachable once mpi_wrap_model has been through it.
                coefficients=mpi.call(mpi.function_call, local_coefficient_names, models_id),
            ),
        ),
        visualizer=MPIExaDGSaddlePointVisualizer(
            models_id, directory=model_kwargs.get("directory", "output/pymor")
        ),
    )

    return model, model.solution_space.subspaces
