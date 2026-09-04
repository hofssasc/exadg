# ExaDG Python bindings

Optional pybind11 bindings that expose ExaDG full-order models to Python, primarily so that
[pyMOR](https://pymor.org) can drive model order reduction while every degree-of-freedom-sized
object stays in C++.

Disabled by default. Nothing in the standard ExaDG build changes unless
`-D EXADG_WITH_PYTHON=ON` is passed.

## How it is put together

CMake builds each extension module directly into a staged Python package in the build tree:

```
<build>/python/
├── pyproject.toml                  generated from python/pyproject.toml.in
└── exadg/
    ├── __init__.py                 generated from python/exadg/__init__.py.in
    └── thermal_block...so          built by EXADG_ADD_PYTHON_MODULE()
```

That directory is a normal installable package, so it is installed with `pip` rather than
placed on `PYTHONPATH`. An **editable** install keeps the extension in the build tree, so
recompiling picks up automatically without reinstalling.

Applications opt in with one line in their `CMakeLists.txt`:

```cmake
IF (EXADG_WITH_PYTHON)
  EXADG_ADD_PYTHON_MODULE(thermal_block python_bindings.cpp)
ENDIF()
```

which produces `from exadg import thermal_block`.

---

## Local workflow

Prerequisites: a deal.II installation ExaDG already builds against, plus `pybind11` in the
Python environment you intend to use.

```bash
# 0. the Python environment that will import the bindings
conda activate queens          # or: python -m venv .venv && source .venv/bin/activate
python -m pip install pybind11 numpy

# 1. configure ExaDG with the bindings enabled
cd $EXADG
cmake -S . -B build \
      -D deal.II_DIR=$DEAL_II_DIR \
      -D EXADG_WITH_PYTHON=ON \
      -D pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# 2. build the library and the extension modules
cmake --build build -j $(nproc) --target thermal_block

# 3. install the staged package, once
python -m pip install -e build/python
```

Afterwards:

```python
from exadg import thermal_block
fom = thermal_block.ThermalBlockFOM2D("input.json", degree=2, refinements=5)
```

**Rebuilding.** Because step 3 is an editable install, later code changes need only

```bash
cmake --build build -j $(nproc) --target thermal_block
```

The reinstall is *not* repeated. Repeat step 3 only when a new module is added or the package
metadata changes.

**Important:** build against the *same* interpreter that will import the result. Configuring
with one environment active and importing from another gives an `ImportError` about a missing
or mismatched Python ABI. `pybind11_DIR` derived from `python -c ...` as above pins this
correctly.

---

## Cluster workflow

The difference from the local case is that the environment is set up by modules, and the
package is installed into a shared location rather than an editable build tree, so that compute
jobs do not depend on the build directory remaining intact.

```bash
# ---- build (login node or a build job) -------------------------------------
module load gcc openmpi cmake python        # site specific
source $HOME/venvs/exadg/bin/activate       # a venv on a shared filesystem
python -m pip install pybind11 numpy

cd $EXADG
cmake -S . -B build \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=$HOME/opt/exadg \
      -D deal.II_DIR=$DEAL_II_DIR \
      -D EXADG_WITH_PYTHON=ON \
      -D pybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake --build build -j 16 --target thermal_block
cmake --install build                       # stages the package under $PREFIX/python

# non-editable install, so the build tree can be removed afterwards
python -m pip install $HOME/opt/exadg/python
```

A batch script then needs no `PYTHONPATH`:

```bash
#!/bin/bash
#SBATCH --job-name=exadg_rom
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

module load gcc openmpi python
source $HOME/venvs/exadg/bin/activate

python reduce.py                   # imports `from exadg import thermal_block`
```

### Where each stage runs

The bindings are imported by the reduction script only. A typical run is a single job: pyMOR
solves the full-order model at the training parameters, computes the basis, projects the
operators and evaluates the reduced model, all in one process.

### MPI

Supported through pyMOR's event loop. Build the model with `mpi_stationary_model` and launch
with pyMOR's runner, which starts the event loop on ranks 1..n-1 and the script on rank 0:

```bash
mpirun -n 4 python -m pymor.tools.mpi reduce.py
```

`python -m pymor.tools.mpi` is not optional. Running the script directly under `mpirun` executes
it once per rank instead of once with n workers, and the ranks deadlock on the first collective.

Requires `mpi4py`, built against **the same MPI** ExaDG is linked to. Installing a wheel that
bundles a different MPI gives a process that starts and then hangs or crashes inside the first
communication:

```bash
MPICC=$(which mpicc) python -m pip install --no-binary=mpi4py mpi4py
```

`python/examples/thermal_block_rb.py` runs unchanged on any number of ranks and prints
quantities that must be identical on all of them.

**What is and is not parallel.** Solves, the POD, the projection, the products and the output
functional are. Empirical interpolation is not: `Operator.restricted()` raises
`NotImplementedError` under MPI, because the stencil is built from locally owned cells only.
pyMOR handles that gracefully — it falls back to evaluating the full operator, which is correct
but not a speed-up.

Two invariants make the parallel path meaningful, and both are checked rather than assumed:

- Global degree-of-freedom indices do not depend on the partitioning, since
  `VectorArray.dofs()` and `Operator.restricted()` address entries by global index.
  `tests/pymor/dof_numbering_stability.cc` asserts this for one, two and four ranks.
- `Vector.amax` returns a global index and breaks ties by the smallest one, so an interpolation
  point does not depend on the rank count. It is a max-reduction over values followed by a
  min-reduction over indices, rather than one `MPI_MAXLOC`, which would break ties by rank.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'exadg'` | the `pip install` step was skipped, or a different interpreter is active |
| `ImportError: ...undefined symbol...` | built against a different Python or deal.II than the one being used |
| `static_assert` failure in `vectorization.h` while compiling | deal.II's compile flags were lost; the module needs `-march=native` because deal.II hard-codes the vectorization width detected at *its* configure time |
| CMake: `Target links to pybind11::module but the target was not found` | `find_package(pybind11)` ran in a subdirectory; imported targets are scoped to that subtree, so it must run at top level |
| CMake: cannot mix keyword and plain `target_link_libraries` | `DEAL_II_SETUP_TARGET()` uses the plain signature and `pybind11_add_module()` the keyword one; `EXADG_ADD_PYTHON_MODULE` applies deal.II's flags by hand instead |
