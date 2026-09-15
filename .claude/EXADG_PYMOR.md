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

## The three applications

| | `poisson/thermal_block` | `incompressible_navier_stokes/forced` | `.../flow_past_cylinder` |
|---|---|---|---|
| interface | `FullOrderModel` | `SaddlePointModel` | `SaddlePointModel` |
| discretisation | CG, Dirichlet rows eliminated | DG (L2), nothing constrained | same |
| parameters | in the operator, $P$ affine components | in the right-hand side only | in the operator: viscosity, and the inflow amplitude |
| drive | body force | body force | inhomogeneous Dirichlet inflow |
| equation | linear | Stokes or Navier–Stokes, by input file | Navier–Stokes |
| dynamics | — | relaxes to steady | von Kármán shedding |

The two flow applications share `include/exadg/pymor/incompressible_flow.h`
(`IncNSSaddlePoint<dim, ApplicationType>`), which holds everything that is about ExaDG rather than
about the case: spaces, blocks, the split, the face loops, the CFL step. A binding supplies only
its parameter shape and its right-hand side — the cylinder returns `{}` / `nullptr` for all of
them, because it has no body force at all.

`forced` serves both flow equations and both regimes: `Equation` selects Stokes or Navier-Stokes,
`Regime` selects Steady or Unsteady, and each changes exactly one term -- the convective one and the
mass one. `input_navier_stokes_transient.json` is the unsteady file. Verified: at `mass_scaling = 0`
the unsteady model reproduces the steady one **bit-exactly**.

**`forced` is creeping flow.** Re = 2-6 at the amplitudes the examples draw (0.5-1.5), ceiling
Re ~ 54 before the velocity-block preconditioner -- which ignores convection -- stops converging. A
transient run relaxes monotonically to steady in t ~ 10; there is no shedding and, in 2D, no
turbulence at any Re. To make time matter there, give `ForcingModes` a time dependence (it is a
`dealii::Function` and has `get_time()`), not a larger Reynolds number.

**`flow_past_cylinder` is where time matters.** `Formulation = Coupled` swaps the shipped splitting
scheme for `BDFCoupledSolution` with an implicit convective term and no penalty terms -- a splitting
scheme's substeps do not compose into one residual, and a penalty term makes the momentum block
depend on the state outside the convective term. `TestCase 1` has a *time-independent* inflow;
`MaxInflow` and `Viscosity` are overridable, so test case 1's steady inflow can be run at test case
2's Reynolds number. Verified against Schäfer–Turek 2D-2: at refinement 1, degree 2, Re = 100 the
wake sheds at **St = 0.300** (published 0.295-0.305); refinement 0 reaches 0.266 on 900 velocity
dofs. No symmetry-breaking perturbation is used or needed -- the cylinder is off-centre.

## The step, not the loop

`interface.h` carries one implicitly discretised step — `s M u + N(u, p) = f`, `s = gamma_0/dt` — so
`momentum`, `apply_nonlinear`, `jacobian_momentum` and `solve` each take `(mass_scaling, time)`, and
`solve` also takes an initial guess (`None` = cold start, which is what a snapshot needs). `s = 0` is
the steady problem; `models/saddle_point.py` writes that down once as `STEADY`.

**The time loop belongs to pyMOR, for both models.** A reduced model has no ExaDG object to step it,
and a FOM stepped by `TimeIntBDF` against a ROM stepped by pyMOR would be two discretisations rather
than a measurement. The history reaches ExaDG inside `f`.

`models/instationary_saddle_point.py` is that loop: `InstationarySaddlePointModel(InstationaryModel)`
-- the same relation pyMOR's own `SaddlePointModel` has to `StationaryModel` -- plus `BDFTimeStepper`
(orders 1 and 2 measured at 1.01 and 2.11). Each step is assembled as
`LincombOperator([mass, A], [gamma_0/dt, 1])` and handed a solver: `ExaDGStepSolver` for the FOM,
`None` for a ROM so pyMOR's Newton takes it. That is what makes "same scheme both sides" structural.

The mass is `blockdiag(M, 0)`, so there is **no initial pressure argument** -- it would be
annihilated -- and the BDF history is velocity alone. The handle route differs from the stationary
model's, and `exadg_model()` knows both: the stationary model keeps its solver on the block
operator, the instationary one on the **time stepper**, because what is inverted is the step.

`InstationaryTensorStokesReductor` / `InstationaryECSWStokesReductor` are a `_Instationary` mixin in
front of the stationary reductors. They project `mass` and `initial_data` and build an
`InstationaryModel` whose stepper is `fom.time_stepper.with_(solver=None)` -- that one substitution
is the whole difference in how FOM and ROM are advanced. The reduced spatial operator is unchanged;
the mass term stays *outside* it, because the stepper varies its coefficient.

pyMOR asserts more than it needs: `SupremizerGalerkinStokesReductor.__init__` demands a
`SaddlePointModel` (stationary by its own hierarchy) while only using two subspaces, `blocks[1,0]`
and a velocity product. The mixin reproduces that setup and calls `ProjectionBasedReductor.__init__`
directly.

**Transient needs `SchurComplementPreconditioner::CahouetChabard`.** With `InverseMassMatrix` the
cost per step *grows* with s (0.91x, 1.35x, 3.32x a steady solve at s = 2, 8, 32); with
Cahouet-Chabard it falls (0.69x, 0.45x, 0.38x). The Schur complement tends to a pressure Laplacian
scaled by 1/s, not a mass matrix.

`velocity_mass()` is declared separately from `velocity_product()` even though `forced` returns the
same handle: a velocity product may legitimately be the H1 product, and reading the time
derivative's operator off the inner product would then be wrong without failing.

**A nonzero `mass_scaling` on a steady model is refused.** `MomentumOperatorData::unsteady_problem`
comes from `SolverType` at setup, so without it the mass kernel is never built and
`set_scaling_factor_mass_operator()` accepts any value and changes nothing — a steady solve returned
as a step. `ForcedFOM::require_mass_admissible` aborts instead.

## Both right-hand sides are easy to under-read

`f` carries the **body force alone**. For a nonlinear model `A` is the application's own residual,
so the momentum equation's inhomogeneous boundary contribution is already inside it — which is why
`_velocity_rhs` takes `constant = None if fom.is_nonlinear`.

`g` is **not zero in general**: it is the divergence operator's boundary term, and it is zero
exactly when the Dirichlet data is homogeneous. `IncNSSaddlePoint::pressure_rhs()` assembles it the
way `OperatorCoupled::rhs_linear_problem` does, sign and scaling included.

The asymmetry is not an inconsistency — one row's inhomogeneity is inside the operator because that
operator is a residual, the other's is beside it because that operator is a plain linear block.

## Two kinds of parameter

A parameter in the **right-hand side** never reaches ExaDG. The model holds its affine components
(`velocity_rhs_components()`) and combines them in Python, so the application is only ever handed
an assembled vector and never told which parameter it is solving at. That is the `forced` case.

A parameter of the **operator** has to go the other way: the residual, the Jacobian and the solve
all belong to the application, and its Newton iteration reads the coefficient out of objects Python
does not own. `SaddlePointModel` therefore declares

```cpp
coefficients()                  // names; empty by default
get_coefficient(name)
set_coefficient(name, value)
```

and `install_coefficients(fom, mu)` in `models/saddle_point.py` pushes them before every evaluate
and every solve — before each, not on change, because two models can share one discretisation and
the last one to solve is the one whose value is installed. Under MPI the names are resolved once
and a plain dict of numbers crosses to the ranks.

**The operator must be a polynomial of declared degree in each coefficient**, which is what lets
the reduced model project once per monomial instead of once per parameter value.
`coefficient_degree(name)` says which; one is the default.

| coefficient | degree | why |
|---|---|---|
| viscosity | 1 | every viscous flux carries one factor of it (affine to **1.9e-14**) |
| inflow amplitude | **2** | the convective flux is quadratic in the velocity, and prescribed Dirichlet data *is* part of that velocity — so the scalar appears in the flux's linear part and again, squared, in its constant |

The reductor probes on a tensor-product grid with `degree + 1` nodes per coefficient and solves one
Vandermonde. Declare too low and the fit is silently wrong away from the nodes; too high costs
probes and is harmless. Verified on the cylinder at inflow 0.35 and 0.77 against nodes at 0, 1, 2:
reduced residual **5e-16**.

**One thing resists decomposition.** A Lax-Friedrichs λ is a maximum of absolute values of a
velocity that *includes* the boundary data, so it is not a polynomial in the amplitude at all. The
sampled operator is therefore compiled at an amplitude of one and told where on the schedule it is
— `set_boundary_amplitude` on both halves, the detached one that evaluates and the builder that
trains. `boundary_amplitude_coefficient()` names which coefficient that is.

**Setting it reaches three copies**, and all three matter:

| copy | read when | reached by |
|---|---|---|
| `Parameters::viscosity` | at apply time, by the Schur preconditioners | `IncNS::ApplicationBase::set_viscosity` |
| `ViscousKernel::data` | every operator evaluation | `SpatialOperatorBase::set_viscosity` |
| one per multigrid level | inside the preconditioner | `MultigridPreconditioner::update()`, which now syncs it and re-initialises the smoothers |

**Two copies are not reached.** The divergence and continuity penalty kernels cache it at setup,
so setting the viscosity with those active now asserts rather than quietly using the old value. The
*pressure-block* preconditioner also caches it: `PressureConvectionDiffusion` copies
`param.viscosity` into its diffusive kernel at setup, and `update_block_preconditioner()` refreshes
the pressure block only under `ale_formulation` or `viscosity_is_variable()` — neither of which a
constant-viscosity sweep sets. So the cylinder's Schur preconditioner stays built at the viscosity
the model was constructed with. Preconditioner only, so a sweep converges to the right answer, but
it degrades as the parameter moves away from that value and is part of why the sweep is less
robust at large steps. Fixing it means either refreshing that block on a viscosity change or
choosing a Schur preconditioner that reads the parameter at apply time, as Cahouet-Chabard does.

Whether the viscosity *is* a parameter is the application's choice, not the binding's —
`viscosity_is_parameter()` defaults to false, and only the cylinder overrides it. Declaring it
would otherwise make every flow model demand a value for a parameter it never varies.

## The Navier–Stokes reduction, in one picture

$$N(u) = \underbrace{B(u,u)}_{\text{trilinear} \to \text{tensor } C_{ijk}} + \underbrace{L u + c}_{\text{inflow} \to \text{affine block}} + \underbrace{S(u)}_{\text{Lax–Friedrichs} \to \text{ECSW}}$$

**The polynomial half is quadratic *plus affine*, not quadratic.** An inhomogeneous Dirichlet
inflow carries the prescribed value into the convective flux, so $Q(u) = B(u,u) + Lu + c$ with
$c = Q(0)$. Two consequences, both of which are identities for homogeneous data and wrong the
moment there is an inflow:

- the polarisation needs $Q(0)$: $B(a,b) = \tfrac12[Q(a{+}b) - Q(a) - Q(b) + Q(0)]$;
- the diagonal is no longer free — $B(a,a) \ne Q(a)$, so it costs $Q(2a)$.

`convective_tensor` returns $(C, L, c)$ projected, and `local_momentum_blocks` folds $L$ and $c$
into the affine block, which is where they belong. What that block then represents is
$M(v) = A(v) - N(v) + Q(v) - B(v,v)$, affine in $v$ (measured: $A - N$ is affine to 1.7e-15), so
$r+1$ evaluations determine it and the reduced residual $B(a,a) + M(a) + S(a)$ is *exactly* the
projected full-order one. Verified on the cylinder at **4e-16**, at viscosities the probes never
used.

Everything hard is in `S`. Its $\lambda = \texttt{uf}\cdot 2\max(|u_M\!\cdot\!n|,|u_P\!\cdot\!n|)$ is
a maximum of absolute values, so it is **(a)** not a polynomial — no tensor, **(b)** not
differentiable — ExaDG freezes it, so the Jacobian is only first-order accurate, and **(c)** the
mechanism that stabilises under-resolved flow, so it cannot be dropped.

`B` is projected exactly, and its tensor is coefficient-free — so are the divergence block and
the stabilisation, whose $\lambda$ is built from the velocity alone. The whole viscosity
dependence of the reduced system is therefore the affine block: two small dense matrices, `base +
value * slope`, assembled online at a cost independent of the mesh. The decomposition is recovered
by **probing at two values and differencing**, not by naming terms — which needs no knowledge of
which term is which or what the boundary condition contributes to each, and is exact for anything
genuinely affine.

`S` is sampled on a weighted subset of faces. The same weights serve the
Jacobian, because a frozen λ makes `S'` a *linear* face operator.

Both halves reach `reductors.py` through **declared** vocabulary and nothing else:
`split_momentum()` returns a `SplitOperator` with `apply` (= N) and `apply_polynomial` (= Q), and
`sampled_momentum(basis)` returns the `SampledOperator` for S. The C++ handle behind a pyMOR model
comes from `exadg_model()` / `exadg_models_id()` in `models/saddle_point.py` — the first is the
rank-local model, the second the ObjectId addressing all of them, and they are different objects
rather than two spellings. Nothing in Python calls a method only one application binds.

`S` is sampled through **two objects split by phase**, and the split is load-bearing:

| | `SampledOperator<VectorType>` (builder) | `CompiledOperator` |
|---|---|---|
| holds | the FOM, the basis, the weights | arrays: weights, `JxW`, normals, lifts, basis traces; a comm; the upwind factor |
| answers | `n_entities`, `set_weights`, `contributions`, `compiled` | `n_selected`, `projected`, `jacobian` |
| when | offline — `contributions` walks the mesh | online — every residual and Jacobian |

`SampledOperator::write_selection(directory, basename)` draws what the weights kept: a **surface
mesh of the selected faces**, one cell per face carrying its weight. deal.II has no "number on a
face" API — `DataOutFaces` evaluates DoF *fields* on faces — so it derives from `DataOutInterface`
and hands each face over as its own `Patch` (set `reference_cell`, or the writer aborts). Rank-
independent for free: matrix-free gives a shared face to exactly one rank.
`FullOrderMomentum.write_selection(filename)` dispatches it.

**Cost model, measured by `navier_stokes_scaling.py`** (degree 2, refinements 3–6, 64x in dofs):
offline exponents `d log t / d log n` are FOM solve 0.89, POD 0.72, projection 0.85, ECSW data
0.97 — and the ECSW **fit 0.60**, the known defect. Online: reduced solve 0.04, sampled S 0.07,
S' 0.12.

Two things make those online numbers honest, and both were found the hard way:
- **Batches, not faces, are the unit of online work** — matrix-free evaluates 4–8 faces at once.
  They go 15 → 23 → 27 → 27, *saturating* at the face count; per-batch cost is flat (1.56 → 1.19
  µs). Wall clock at tens of µs otherwise looks like mesh dependence when it is batch growth.
- **Detach before timing.** With the FOM co-resident its working set evicts the compiled arrays;
  the *identical* work ran 38% slower at refinement 6 and unchanged at refinement 5.
  `FullOrderMomentum.detach()` drops the builder, and using it moved the exponents from
  0.13/0.17/0.17 to 0.04/0.07/0.12.

`compiled()` does the pass; `set_weights` only records and invalidates, so a fit's discarded faces
are never gathered. The compiled half is **not templated on a vector type** and is bound once in
`core_bindings.cpp`'s module body, not per vector type. λ is reached through the static
`ConvectiveKernel::lambda_of`, so nothing on the online path holds a kernel — or a mesh.
`FullOrderMomentum` in `reductors.py` mirrors this: `self.builder`, and a lazy `compiled` property.

## Rules that are load-bearing

1. **Nothing is assumed.** Symmetry, invertibility, a restricted evaluation — each defaults to
   "no" and its transposed form aborts rather than improvising.
2. **Structure is declared in C++, naming is not.** `parameter_shape()` says how many parameters;
   what they are called and how the operator depends on them live in Python. The exception proves
   it: a coefficient the *operator* depends on is named in C++, because the application's own
   solver has to look it up — see "Two kinds of parameter".
3. **Composite objects need explicit MPI counterparts.** `mpi_wrap_model` wraps *leaf* operators.
   Anything pyMOR composes from them — a `BlockOperator`, its solver, a block visualizer — gets
   nothing. Hence `MPIExaDGCoupledSolver`, `MPIExaDGSaddlePointVisualizer`, `reductors.dispatch`.
4. **A model constructor is collective**, so the FOM is built on every rank from a picklable
   recipe naming the application by string.
5. **Anything returning data must return the global answer.** Every wrapped method runs on all
   ranks and pyMOR keeps rank 0's.

## Traps, each of which cost real time

- **A velocity has two opposite ghost requirements.** As the `src` of `MatrixFree::loop` it must
  *not* be ghosted — the loop exchanges ghosts itself, hence `ForcedFOM::owned()`. As the installed
  **transport velocity** it *must* be, because only `evaluate_nonlinear_operator` calls
  `update_ghost_values_velocity()`; `apply`/`vmult` assume the caller did it (ExaDG's own time
  integrator does, at the call site). Hence `ForcedFOM::transported()`. Using `owned()` for both
  made `C(w,w)` **50% wrong on four ranks** while every nonlinear path agreed to eight digits —
  and it hid because a wrong Jacobian does not move a converged Newton solve, only its iteration
  count. Found by running `convective_split.py` under MPI.
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
- **The norm of a velocity is not a shedding diagnostic.** A limit cycle's fluctuation is a
  quadrature pair, so $\|u(t)\|$ is nearly constant while the flow oscillates fully: on a cylinder
  wake whose fluctuation is 16% of the mean it read a peak-to-peak of 3.5e-03 and looked steady.
  Take the deviation from the *time mean*, and get the frequency off a scalar probe of it — the
  norm of that deviation is nearly constant too, for the same reason, so counting its mean
  crossings measures nothing.
- **pyMOR hides a time-dependent parameter value until `at_time()` is called.** `'inflow' in mu`
  is `False` for one, and `mu['inflow']` raises — so code that installs coefficients by testing
  membership skips it and solves at whatever was installed last, silently and with a plausible
  answer. `install_coefficients` checks `mu.time_dependent_values` and refuses. The stepper passes
  `mu.at_time(t)`; anything that does not has to be told.
- **ECSW weights are fitted to a term that carries the boundary data.** Training every state at one
  amplitude while the states came from a schedule fits a different operator — and converges, so
  nothing looks wrong. The reductor now takes `training_amplitudes`, one per state, and refuses
  without them when the application declares a boundary coefficient.
- **A block-Jacobi multigrid smoother is serial-only.** It builds its block diagonal from separate
  cell and face loops, and ExaDG aborts in parallel asking for `use_cell_based_face_loops` instead
  — which is *not* a free switch here, because the hyper-reduction samples face batches directly
  and changing how faces are traversed changes what it counts. `PreconditionerSmoother::PointJacobi`
  needs no block diagonal, costs iterations rather than correctness, and leaves the face machinery
  alone. Chebyshev is not the alternative it is for `forced`: a velocity block that keeps its
  convective term is not symmetric, so Chebyshev's eigenvalue estimate has nothing to work with.
- **A missing `g` does not look like a missing constant.** When the continuity right-hand side was
  left out (see above), the reduced velocity collapsed to 2.6 % of the full-order one, the reduced
  pressure sat at a constant 17x too large, **doubling the basis changed nothing**, and
  root-finding on the reduced step from the projected full-order state converged — to a state 140
  away from a trajectory of norm 1.08. What identified it was the *shape* of the residual, not its
  size: split by row, the continuity row was 1.437 at every level, identical to four digits, while
  the momentum row varied. A closure error moves with the state; a constant does not.
- **The convective operator vanishes at zero velocity only for homogeneous data.** With an inflow,
  $N(0) \ne 0$ and $Q(0) \ne 0$; anything that treats the polynomial half as a pure quadratic form,
  or takes the affine block's constant from $A(0)$ rather than $A(0) - N(0)$, leaves a term
  weighted by $\sum_j a_j$ instead of by 1. It then cancels at exactly one point of the reduced
  space, so a ROM checked only at its training point looks right. Cost: 1.5e-02 relative, hidden
  behind `forced`'s homogeneous boundary.
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
| `navier_stokes_transient.py` | BDF coefficients and rates, s=0 vs steady, relaxation, step cost |
| `navier_stokes_transient_rom.py` | streamed offline phase, timings, FOM error; `--vtu` / `--chunks` / `--sketches` |
| `cylinder_transient_rom.py` | a parameter of the *operator*, an inhomogeneous inflow, a shedding wake; `--vtu` (~20 min) — identical at 1 and 4 ranks |
| `navier_stokes_scaling.py` | the cost model: offline ~n, online ~0 (sweep, minutes) |
| `ctest -R pymor` | DoF-numbering stability at 1/2/4 ranks; the restricted operator |

Legitimately rank-dependent: Stokes' last row and the NS residual sit at their solver's tolerance
floor; `navier_stokes_tensor`'s `tensor vs Galerkin` is a cancellation and sits at roundoff (4e-16
on one rank, 7e-13 on four); `thermal_block_ei`'s first row is a singular reduced operator that
raises on one rank and returns ~1e17 on four; face **counts** grow with the rank count because
matrix-free pads its face batches per rank (144 → 184 at four). A **sketched** fit is
rank-dependent beyond that: the Gaussian rows are drawn over a differently padded candidate set,
so the selection itself moves (13 faces at one rank, 11 at four) and with it the fit residual. The
reduced error still agrees to four digits — which is the right expectation for two different fits,
not the nine that holds where the computation is the same one reduced differently.

## Known defects, not yet fixed

**A plain Galerkin ROM of the cylinder wake is closure-limited.** At refinement 0, Reynolds 80-170
and 66+18 modes it reaches worst-level errors of a few tens of percent at untrained Reynolds
numbers (0.28 and 0.23), against a projection floor of 1.4e-02 and 7.3e-03 — the error is **20-30x
what the basis can do**, so more modes will not close it. That is the expected behaviour of
POD-Galerkin on convection-dominated flow and is a *basis-and-closure* problem (problem 4 in the
vault's road map), not a defect in the projection: every operator is exact to machine precision and
the residual reproduces the full-order one at 4e-16. The online speed-up is correspondingly modest
— 7.6x, because a 102-dimensional reduced space makes the tensor contraction $O(r^3)$ the dominant
online cost, where the forced problem's 35 dimensions gave 76x.

Read those errors as an order of magnitude, not a pinned number. Switching the multigrid smoother
— which cannot change the converged full-order solution, only the iterations to reach it — moved
the ECSW selection by one face (29 → 30, fit residual 1.4e-01 → 1.1e-01) and with it the worst
level from 0.15 to 0.28. A trajectory perturbed at the solver tolerance flips a discrete face
selection near a threshold, and a closure-limited reduced model is sensitive to that. The
projection floor, which is the basis alone, moved by 12 %.

**The sampled Jacobian is 3.5e-04 from ExaDG's with an inhomogeneous inflow**, against 9.8e-16
with homogeneous data. Localised by elimination: the tensor half and the affine half were each
checked against a central difference of their own operator — exact, one being quadratic and the
other affine — and the residual `S` matches `N - Q` at 2.2e-16, so the gap is `CompiledStabilisation
::jacobian` alone. It does not move a solution: the reduced solution is defined by its residual,
which is exact, and ExaDG's own Jacobian is already not the derivative of its own residual — both
sit 22.7% from a finite difference of it, because λ is frozen. The cost is Newton iterations,
already linear rather than quadratic for that reason.

**The ROM is not yet a deliverable.** `CompiledStabilisation` holds a communicator and allreduces on
every `projected()` / `jacobian()`, and nothing serialises. `detach()` drops the mesh, not the
communicator.

**The ECSW weight fit still gathers the columns on every rank.** The *rows* are handled --
`sketch_rows` streams a Gaussian sketch as the matrix is assembled, measured at a twelfth of the
memory for the same fit over trajectories. The columns are not, and that is what a 3D mesh grows
(5.7 GB per rank at 10^6 faces).

**Never read a sketched fit's residual off its own sketch.** NNLS minimises over it, so the number
is in-sample and biased low -- a six-row sketch reported 5e-16 while being 28% wrong. A second,
independent sketch rides along in the same pass and is what gets reported; it tracks the truth to
about 10% over four orders of magnitude. `local_ecsw_weights` warns when the support exceeds half
the sketch.

**The basis is a hierarchical POD.** `pod`'s method of snapshots forms an N x N Gramian -- 1.76 GB
per rank and a ~6 min eigensolve at 24 parameters x 641 levels -- and its eigensolve is serial and
replicated, exactly like the ECSW fit. `inc_hapod` compresses trajectory by trajectory under a
*certified* l2-mean bound; at matched accuracy it gives the same basis from a 64x smaller Gramian.
`eps` is absolute, so scale it by the rms snapshot norm. It can also stream, at the price of a
second pass of FOM solves -- ECSW needs the same states once the basis exists.

**The offline phase streams** -- `exadg/mor/basis.py`. The blocker was never ECSW: *any*
hyper-reduction trains at `a_i = V^T M u_i`, and V is unknown until every trajectory is seen, so
the naive ordering needs the snapshots twice (9 GB at production, or a second solve pass).
Projection is linear, so coefficients are *carried* through each basis update -- `c' = (V^T M W) c`
-- and no snapshot is ever needed. `streaming_basis` returns `CompressedSnapshots` and **not**
coefficients on the global basis: supremizers are directions no velocity POD contains, so a
projection made before the enrichment cannot be extended to it. The reductor takes it as
`training_snapshots=`; `training_states=` is unchanged.

Measured against keeping everything (6 trajectories x 33 levels): 26 vectors held against 396, the
same basis dimensions, faces 18-21 against 18, and ROM error 3.9483e-02 against 3.9434e-02 -- 0.12%.
The face selection *does* shift by a face or two and it does not matter. Accumulation over 129
updates costs 7x the error for 21x less memory, and stays at the order of the tolerance asked for:
compress often, ask for one decade more than you need.

Still mesh-scale: `contributions()` walks the mesh once per training state. Streaming removes the
storage and the second solve pass, not the face loops.

**The three tolerances control different things.** `BASIS_TOLERANCE` is a *projection* error on
the training snapshots, and the reduced error comes out 2.5-15x larger -- so a ROM error below 1e-2
needs about 3e-3, below 1e-3 needs about 1e-4. `TOLERANCE` (ECSW) is nearly free to tighten: at a
basis tolerance of 3e-4, going 1e-2 -> 3e-4 takes the fit from 38 faces to 89 and moves the ROM
error by 0.02%. What *does* become visible below ~1e-3 is the **basis**: two compression routes
meeting the same certified bound land on different subspaces, and the stored/streamed gap goes from
0.3% to 8.8% -- with the streamed one ahead, so it is a difference and not a penalty.

**Derive the step count from a CFL number, never fix it.** `ForcedFOM::time_step_for_cfl` calls
ExaDG's own `calculate_time_step_cfl_global()` -- element sizes from the MatrixFree, plus
`max_velocity` and `cfl_exponent_fe_degree_velocity` -- and `steps_for_cfl` rounds it into a count.
Collective, so it is dispatched. A count held fixed across meshes silently changes the Courant
number when the mesh changes, and then a refinement study varies two things at once.
`max_velocity` is an a-priori scale (an input parameter), not a measurement: a step has to be
chosen before there is a solution.

**Time series come out as a `.pvd`.** Both visualizers take a trajectory: one record per level plus
a ParaView collection, `times=` optional. Written in Python -- `write_vtu_with_pvtu_record` appends a
counter of its own, and a collection makes the record names irrelevant. `float()` the timestep or
ParaView chokes on `np.float64(...)`. Write at a time interval, not per step:
`binding.output_levels(times, interval)` picks the levels ExaDG's `TimeControl` would, and the
transient examples take `U[levels]` *before* `reconstruct` (`VTU_INTERVAL`).

**A view shares its `obj_id` with the whole array.** `basis[k]` sent to the ranks as `impl.obj_id`
alone is all of `basis`. `MPIExaDGSaddlePointVisualizer` passes `ind` along; before it did, every
POD mode was written as mode 0, bitwise.

Status, order and the architecture for each are in
`~/Documents/Dissertation/Literature/40-Reference/ExaDG ROM Next Steps.md`. **Read that first.**

## Vault

`~/Documents/Dissertation/Literature/40-Reference/`

- `ExaDG pyMOR Interface.md` — the interface, then each model in detail
- `ExaDG Operators and Solvers.md` — ExaDG itself: operators, solvers, the two setups
- `ExaDG ROM Next Steps.md` — the three defects above, and how to fix them properly
- `ExaDG Build System.md` — build and linking
