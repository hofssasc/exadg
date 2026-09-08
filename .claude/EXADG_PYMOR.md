# ExaDG ↔ pyMOR: where everything is

Orientation for a session picking up the `exaDG-pyMOR` branch. Facts and pointers only — the
reasoning lives in the vault notes at the bottom.

## Run environment

```bash
PYTHONPATH=/home/hofstetter/Code/exadg/build/python
~/Code/miniforge3/envs/queens/bin/python           # 3.11; the .so files are built for that ABI
```

The base miniforge python is 3.13 and fails with a bare `ModuleNotFoundError: No module named
'exadg.forced'`, which reads like a missing build rather than a version mismatch.

**Editing `python/exadg/**/*.py` is not enough.** CMake *copies* them into `build/python/exadg/`,
so re-run `cmake .` in `build/` before an import sees the change. C++ needs
`make -j8 forced` / `thermal_block` / `_core`.

Examples read paths relative to the repository root — run them from there, not from `build/`.

```bash
python python/examples/stokes_rb.py
mpirun -n 4 python -m pymor.tools.mpi python/examples/stokes_rb.py
```

`-m pymor.tools.mpi` is not optional: it starts pyMOR's event loop on ranks 1…n−1 and the script
on rank 0. Plain `mpirun` runs the script once per rank and deadlocks on the first collective.

## The four layers

| layer | file | knows about |
|---|---|---|
| vocabulary | `include/exadg/pymor/interface.h` | shapes only — abstract base classes, no physics |
| binding | `python/core_bindings.cpp` → module `exadg._core` | binds the vocabulary **once** |
| application | `applications/<app>/python_bindings.cpp` | the physics; implements the vocabulary |
| pyMOR shim | `python/exadg/mor/binding.py` | pyMOR's interfaces; **no physics at all** |
| assembly | `python/exadg/mor/models/{stationary,saddle_point}.py` | which pyMOR `Model`; naming, parameterisation |

`exadg._core` is **forced, not tidy**: pybind11's type registry is process-global and keyed by
`std::type_index`, so a second module binding `distributed::Vector` aborts with
`generic_type: type "Vector" is already registered!`. Every application module must
`py::module_::import("exadg._core")` at the top of its `PYBIND11_MODULE`.

## Two models

| | `FullOrderModel` | `SaddlePointModel` |
|---|---|---|
| application | `applications/poisson/thermal_block/` | `applications/incompressible_navier_stokes/forced/` |
| Python | `models/stationary.py` | `models/saddle_point.py` |
| examples | `thermal_block_rb.py`, `thermal_block_ei.py` | `stokes_rb.py`, `navier_stokes_rb.py` |
| spaces | is a `Space` — one | hands out two |
| operator | affine components $\sum_i c_i(\mu) A_i$ | three fixed blocks (A, B, and B's transpose) |
| parameters | in the operator | in the right-hand side only |
| nonlinear | no | `is_nonlinear` / `apply_nonlinear` / `jacobian_momentum` |

The `forced` application is one application for both Stokes and Navier–Stokes: the `Equation`
setting in the input file selects them, the convective term is the only difference, and
`input.json` / `input_navier_stokes.json` are the two configurations.

## Rules that are load-bearing

1. **Nothing is assumed.** Symmetry, invertibility, a restricted evaluation — each is a virtual
   method defaulting to "no", whose transposed form aborts rather than improvising. A wrong guess
   is invisible: an adjoint solve that quietly solves the wrong system returns plausible numbers.
2. **Structure is declared in C++, naming is not.** `parameter_shape()` says there are sixteen
   coefficients; that they are called `mu` and enter through an exponential is a modelling choice
   and lives in Python, where changing it is not a recompile.
3. **Composite objects need explicit MPI counterparts.** `mpi_wrap_model` wraps *leaf* operators
   in `MPIOperator` and attaches `MPISolver`. Anything pyMOR builds by composing those leaves — a
   `BlockOperator`, its solver, a block visualizer — is assembled on rank 0 and gets nothing. Hence
   `MPIExaDGCoupledSolver` and `MPIExaDGSaddlePointVisualizer`.
4. **A model constructor is collective.** deal.II partitions the triangulation across the
   communicator, so the FOM must be built on *every* rank. That is why `mpi_*_model` ships a
   picklable recipe naming the application by string rather than a model.
5. **Everything returning data must return the global answer.** Every wrapped method runs on all
   ranks and pyMOR keeps rank 0's. `dofs()` and `amax()` reduce in C++.

## Traps, each of which cost real time

- **`set_velocity_ptr` keeps a pointer.** `MomentumOperator::set_solution_linearization` forwards
  to it, so ExaDG dereferences the linearisation velocity long after pyMOR has discarded the
  `Jacobian` that owned it → segfault in `update_ghost_values`. `set_velocity_copy` does *not* fix
  it (a different nil-pointer crash in `ConvectiveKernel::reinit_cell`). The **model** owns it, via
  `ForcedFOM::install_linearization`.
- **`solve_nonlinear_problem` resets the mass scaling to 1.0 on every call.** A steady residual
  carries no mass term, so any operator that called `set_scaling_factor_mass_operator(0.0)` once at
  construction silently becomes `A + M` after the first solve. Set it inside `apply`.
- **`A'(u)` is not the exact derivative of `A(u)`.** ExaDG over-integrates the convective term
  (`quad_index_nonlinear`) and its linearisation (`quad_index_linearized`) differently; a finite
  difference plateaus at ~3.4e-4. Newton converges linearly, not quadratically. Deliberate, not a
  bug — the solution is defined by the residual.
- **`SolverControl::NoConvergence` escaping `solve()` aborts the interpreter.** Caught; the model
  declines, which is what lets a greedy skip an unreachable training parameter.
- **A snapshot velocity is discretely divergence-free**, so `Bu ≈ 0` and an adjoint check probed
  at a snapshot divides roundoff by roundoff (reads ~1e-5, means nothing). Probe at `Bᵀp`.
- **pyMOR names a `BasicObject`'s logger after its class's *module*.** `ExaDGNonlinearMomentum`
  logs under `exadg`, never under `pymor`.
- **`restricted()` returns `nullptr` above one rank** (the stencil is gathered from locally owned
  cells only). The binding turns that into `NotImplementedError`, which pyMOR treats as "fall back
  to the full operator" — correct, but no speed-up.
- **`mpi.call` returns `None` outright when pyMOR was built without mpi4py**
  (`pymor/tools/mpi.py:73` sets `finished = True`). Never route a serial path through it.
- **pyMOR bug**: `mpi_wrap_model` asserts `isinstance(base_type, Model)` (an instance) then does
  `class MPIWrappedModel(MPIModel, base_type)` (needs a class). No value satisfies both; pass an
  `ObjectId` instead. Not reported upstream.

## The regression surface

Run all of these before claiming a change is safe. Every printed quantity is global, so **1 and 4
ranks must agree to nine significant digits** — the twelfth digit moves with the partitioning
because reductions sum in a different order.

| check | what it pins |
|---|---|
| `thermal_block_rb.py` (1, 4 ranks) | affine path, certified estimator, `dofs`/`amax` |
| `thermal_block_ei.py` (serial) | the restriction contract through pyMOR's own call path |
| `stokes_rb.py` (1, 4 ranks) | `⟨Bu,p⟩` vs `⟨u,Bᵀp⟩`; block system vs ExaDG's solve; exactness at P modes |
| `navier_stokes_rb.py` (1, 4 ranks) | the nonlinear path; error falls with the basis |
| `ctest -R pymor` | DoF-numbering stability at 1/2/4 ranks; the restricted operator |

Known rank-dependent values: the Stokes last row (~1e-10 vs ~7e-11) and the NS residual
(6.556e-07 vs 1.244e-07) are both at their solver's tolerance floor, not defects.

## Current state

Reduction works end to end for the thermal block, Stokes and Navier–Stokes. The NS ROM is
**correct but not fast** — every reduced Newton step evaluates the residual at full order.

Next step: **hyper-reduction, ECSW rather than DEIM.** DEIM needs `restricted()`, i.e. a second
implementation of the physics on a stencil (`include/exadg/pymor/restricted_laplace.h` is that for
the Laplace operator, and it is matrix-based on purpose). ECSW keeps the matrix-free loop and only
attaches per-cell weights, which is what `RestrictedLaplace`'s per-cell contribution layout is
already shaped for. pyMOR has DEIM/EIM but **no ECSW**.

Also outstanding: switch the thermal block from `ExponentialParameterFunctional` to
`ProjectionParameterFunctional` and drop the P³ workaround (deferred deliberately, so the examples
stayed a byte-identical regression test through the interface rewrite).

## Vault

`~/Documents/Dissertation/Literature/40-Reference/`

- `ExaDG pyMOR Interface.md` — the interface, then each model in detail
- `ExaDG Operators and Solvers.md` — ExaDG itself: operators, solvers, and the two setups
- `ExaDG Build System.md` — build and linking
