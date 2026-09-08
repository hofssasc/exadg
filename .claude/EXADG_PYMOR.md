# ExaDG ↔ pyMOR

Entry point for a session picking up the `exaDG-pyMOR` branch. Facts and pointers; the reasoning
lives in the vault notes at the bottom.

## Run it

```bash
PYTHONPATH=/home/hofstetter/Code/exadg/build/python
~/Code/miniforge3/envs/queens/bin/python          # 3.11; the .so files are built for that ABI
```

The base miniforge python is 3.13 and fails with a bare `ModuleNotFoundError`, which reads like a
missing build rather than a version mismatch.

**Editing `python/exadg/**/*.py` is not enough.** CMake *copies* them into `build/python/exadg/`,
so re-run `cmake .` in `build/` before an import sees the change — and a **new** file must be added
to the list in `python/CMakeLists.txt`. C++ needs `make -j8 forced` / `thermal_block` / `_core`.

Examples take paths relative to the repository root; run them from there.

```bash
python python/examples/navier_stokes_ecsw.py
mpirun -n 4 python -m pymor.tools.mpi python/examples/navier_stokes_ecsw.py
```

`-m pymor.tools.mpi` is not optional: it starts pyMOR's event loop on ranks 1…n−1 and the script on
rank 0. Plain `mpirun` runs the script once per rank and deadlocks on the first collective.

## Layers

| layer | file | knows |
|---|---|---|
| vocabulary | `include/exadg/pymor/interface.h` | shapes only — abstract base classes |
| binding | `python/core_bindings.cpp` → `exadg._core` | binds the vocabulary **once** |
| application | `applications/<app>/python_bindings.cpp` | the physics |
| pyMOR shim | `python/exadg/mor/binding.py` | pyMOR's interfaces; no physics |
| models | `python/exadg/mor/models/{stationary,saddle_point}.py` | which pyMOR `Model`; naming |
| reductors | `python/exadg/mor/reductors.py` | the tensor and ECSW reductions |

`exadg._core` is forced, not tidy: pybind11's type registry is process-global, so a second module
binding `distributed::Vector` aborts with `generic_type: type "Vector" is already registered!`.
Every application module must `py::module_::import("exadg._core")`.

## The two applications

| | `poisson/thermal_block` | `incompressible_navier_stokes/forced` |
|---|---|---|
| interface | `FullOrderModel` | `SaddlePointModel` |
| discretisation | CG, Dirichlet rows eliminated | DG (L2), nothing constrained |
| parameters | in the operator, $P$ affine components | in the right-hand side only |
| equation | linear | Stokes or Navier–Stokes, by input file |

`forced` serves both flow equations: `Equation` selects them and the convective term is the only
difference, so one is a controlled comparison for the other.

## The Navier–Stokes reduction, in one picture

$$N(u) = \underbrace{B(u,u)}_{\text{trilinear} \to \text{tensor } C_{ijk}} + \underbrace{S(u)}_{\text{Lax–Friedrichs} \to \text{ECSW}}$$

Everything hard is in `S`. Its $\lambda = \texttt{uf}\cdot 2\max(|u_M\!\cdot\!n|,|u_P\!\cdot\!n|)$ is
a maximum of absolute values, so it is **(a)** not a polynomial — no tensor, **(b)** not
differentiable — ExaDG freezes it, so the Jacobian is only first-order accurate, and **(c)** the
mechanism that stabilises under-resolved flow, so it cannot be dropped.

`B` is projected exactly; `S` is sampled on a weighted subset of faces. The same weights serve the
Jacobian, because a frozen λ makes `S'` a *linear* face operator.

## Rules that are load-bearing

1. **Nothing is assumed.** Symmetry, invertibility, a restricted evaluation — each defaults to
   "no" and its transposed form aborts rather than improvising.
2. **Structure is declared in C++, naming is not.** `parameter_shape()` says how many parameters;
   what they are called and how the operator depends on them live in Python.
3. **Composite objects need explicit MPI counterparts.** `mpi_wrap_model` wraps *leaf* operators.
   Anything pyMOR composes from them — a `BlockOperator`, its solver, a block visualizer — gets
   nothing. Hence `MPIExaDGCoupledSolver`, `MPIExaDGSaddlePointVisualizer`, `reductors.dispatch`.
4. **A model constructor is collective**, so the FOM is built on every rank from a picklable
   recipe naming the application by string.
5. **Anything returning data must return the global answer.** Every wrapped method runs on all
   ranks and pyMOR keeps rank 0's.

## Traps, each of which cost real time

- **ExaDG hands out pointers; the caller owns the lifetime.** Three bugs of this shape, all silent
  serially and fatal under MPI: `set_velocity_ptr` (`solve()` left the kernel pointing at its own
  local solution), `OperatorBase::reinit` keeping a `lazy_ptr` to a stack-local `AffineConstraints`,
  and a `ConvectiveKernel` with no velocity storage whose `update_ghost_values_velocity()`
  dereferenced nothing. **Suspect lifetimes first when MPI corrupts the heap.**
- **Vectors from Python must be copied and un-ghosted** before `MatrixFree::loop`, which does its
  own exchange — `ForcedFOM::owned()`.
- **`solve_nonlinear_problem` resets the mass scaling to 1.0**, so a steady operator must zero it
  inside `apply`, not once at construction.
- **`A'(u)` is not the exact derivative**: λ is frozen. Measured 4.5e-08 at `upwind_factor = 0`
  against 5.1e-03 at 1.0. Not a quadrature effect — both indices are 2.
- **On a boundary face the test and trial jumps differ.** The residual is integrated against the
  interior test function alone; the increment still has an exterior value, so the trial jump is
  the mirrored one. Conflating them makes the boundary term twice too large — which then hides as
  a plausible-looking $O(h^5)$ residue.
- **ExaDG's linearly-implicit operator is not the polarisation of its nonlinear one.** Both are
  trilinear; they differ at discretisation level (~$h^5$). Build tensors from the nonlinear one.
- **A snapshot velocity is discretely divergence-free**, so an adjoint check probed at a snapshot
  divides roundoff by roundoff. Probe at $B^\top p$.
- **pyMOR names a `BasicObject`'s logger after its class's module** — `ExaDGNonlinearMomentum` logs
  under `exadg`, never `pymor`.
- **`MPISolver` assembles before dispatching**, so a parametric operator must return something its
  own solver can prepare — see `ExaDGParametricOperator.assemble`.
- **pyMOR bug**: `mpi_wrap_model` asserts `isinstance(base_type, Model)` then subclasses it. Pass
  an `ObjectId`.
- **Teardown noise is benign**: `Rank0ObjectId.__del__` raising `TypeError` after a clean exit.

## Regression surface

Every printed quantity is global, so **1 and 4 ranks must agree to nine significant digits**.

| script | pins |
|---|---|
| `thermal_block_rb.py` | affine path, certified estimator, `dofs`/`amax` |
| `thermal_block_ei.py` | the restriction contract through pyMOR's own call path |
| `stokes_rb.py` | adjoint identity, block system vs ExaDG's solve, exactness at $P$ modes |
| `navier_stokes_rb.py` | the nonlinear path |
| `convective_split.py` | the split is exact; the face sum adds up; the Jacobian's frozen λ |
| `navier_stokes_tensor.py` | the tensor reproduces a plain Galerkin ROM |
| `navier_stokes_ecsw.py` | sampling does not move the error |
| `ctest -R pymor` | DoF-numbering stability at 1/2/4 ranks; the restricted operator |

Legitimately rank-dependent: Stokes' last row and the NS residual sit at their solver's tolerance
floor; `thermal_block_ei`'s first row is a singular reduced operator that raises on one rank and
returns ~1e17 on four.

## Known defects, not yet fixed

**The ECSW weight fit is redundant across ranks.** `local_ecsw_weights` has every rank gather the
whole training matrix — $(n_\text{train}\, r) \times n_\text{faces}$, on *every* rank — and solve
the same NNLS. Scales in neither memory nor work, and its width grows with the mesh. Offline, so it
bounds the size of problem that can be trained rather than the cost of a reduced solve.

The architecture for the fix is in
`~/Documents/Dissertation/Literature/40-Reference/ExaDG ROM Next Steps.md`. **Read that first.**
Two defects listed there are now fixed: the evaluator owns its basis and weights (no tokens), and
it speaks reduced coefficients (no full-order reconstruction).

## Vault

`~/Documents/Dissertation/Literature/40-Reference/`

- `ExaDG pyMOR Interface.md` — the interface, then each model in detail
- `ExaDG Operators and Solvers.md` — ExaDG itself: operators, solvers, the two setups
- `ExaDG ROM Next Steps.md` — the three defects above, and how to fix them properly
- `ExaDG Build System.md` — build and linking
