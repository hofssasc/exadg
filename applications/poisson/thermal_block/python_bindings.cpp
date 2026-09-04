/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

/*
 * Python bindings for the thermal block full-order model.
 *
 * The purpose is to let pyMOR drive model reduction while every degree-of-freedom-sized object
 * stays on the C++ side. pyMOR is explicitly designed for this: its manual states that "direct
 * memory access to the vector data from Python is not required to integrate a solver with
 * pyMOR". Accordingly the surface exposed here is deliberately narrow -- essentially the
 * operations pymor.vectorarrays.list.ListVectorSpace and pymor.operators.interface.Operator
 * require:
 *
 *     Vector    scal, axpy, inner, norm, norm2, copy, dofs, to_numpy
 *     Space     zero_vector, make_vector (via the model below)
 *     Operator  apply
 *
 * to_numpy() exists for debugging only; it refuses on more than one rank, because gathering the
 * whole vector onto every rank is exactly what this interface exists to avoid.
 *
 * MPI: supported through pyMOR's event loop (pymor.tools.mpi), where Python runs on every rank
 * and rank 0 dispatches. Every call below therefore executes simultaneously on all ranks, and
 * pyMOR keeps rank 0's return value -- so anything that returns data has to return the global
 * answer rather than this rank's slice. dofs() and amax() do their own reductions accordingly.
 * RestrictedLaplace is still serial: it collects the stencil from locally owned cells only, so
 * the Python layer raises NotImplementedError under MPI, which is what pyMOR expects.
 */

// C/C++
#include <iostream>
#include <sstream>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// deal.II
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>

// ExaDG
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/poisson/driver.h>
#include <exadg/pymor/block_coefficient.h>
#include <exadg/pymor/sensor_operator.h>
#include <exadg/utilities/general_parameters.h>

// application
#include "application.h"

namespace py = pybind11;

namespace ExaDG
{
using Number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

namespace
{
/**
 * Initializes deal.II's external libraries exactly once per interpreter, starting MPI only if
 * nobody else has.
 *
 * Under pyMOR's event loop mpi4py is imported first and has already called MPI_Init by the time
 * this module loads. deal.II's MPI_InitFinalize asserts MPI_Initialized() == 0 and would abort
 * with "MPI error. You can only start MPI once!". Kokkos, PETSc, Zoltan and p4est still have to
 * be brought up, so the fix is to initialize the same set of libraries as MPI_InitFinalize with
 * the MPI bit cleared -- deal.II guards only the MPI branch with that assertion.
 *
 * The object is deliberately leaked. Its destructor finalizes MPI and the libraries, and
 * running that during interpreter teardown -- after an arbitrary amount of other cleanup, and
 * possibly after deal.II statics have gone -- is a reliable source of crashes at exit.
 */
void
ensure_mpi_initialized()
{
  static bool initialized = false;

  if(initialized)
    return;

  static char   program_name[] = "exadg";
  static char * argv_storage[] = {program_name, nullptr};

  int     argc = 1;
  char ** argv = argv_storage;

  int mpi_already_started = 0;
  MPI_Initialized(&mpi_already_started);

  // the set MPI_InitFinalize uses, so that behaviour is identical either way
  auto libraries = dealii::InitializeLibrary::Kokkos | dealii::InitializeLibrary::SLEPc |
                   dealii::InitializeLibrary::PETSc | dealii::InitializeLibrary::Zoltan |
                   dealii::InitializeLibrary::P4EST | dealii::InitializeLibrary::PSBLAS;

  if(not mpi_already_started)
    libraries = libraries | dealii::InitializeLibrary::MPI;

  new dealii::InitFinalize(argc, argv, libraries, 1);

  initialized = true;
}

/**
 * Silences C++ standard output for its lifetime.
 *
 * ExaDG announces the grid, the discretization and the parameter list whenever a solver is set
 * up. That is useful for a command line run and pure noise when a sampler constructs the model
 * -- and the driver's is_test flag only suppresses the timing report, not the banner.
 */
class SuppressOutput
{
public:
  explicit SuppressOutput(bool const active) : buffer(nullptr)
  {
    if(active)
      buffer = std::cout.rdbuf(null_stream.rdbuf());
  }

  ~SuppressOutput()
  {
    if(buffer != nullptr)
      std::cout.rdbuf(buffer);
  }

  SuppressOutput(SuppressOutput const &) = delete;
  SuppressOutput &
  operator=(SuppressOutput const &) = delete;

private:
  std::ostringstream null_stream;
  std::streambuf *   buffer;
};
} // namespace

/**
 * The thermal block operator restricted to a small set of output degrees of freedom.
 *
 * This is the object empirical interpolation needs. pyMOR's contract is
 *
 *     A.apply(U).dofs(dofs) == restricted.apply(U.dofs(source_dofs))
 *
 * that is: given the values of a vector on a small set of *source* degrees of freedom, reproduce
 * exactly the rows of the operator belonging to the requested *output* degrees of freedom.
 * Everything hyperreduction promises follows from that being cheap -- once the residual can be
 * evaluated at a few points without touching the whole mesh, DEIM and ECSW become possible.
 *
 * The source set is the stencil: every degree of freedom sharing a cell with a requested one.
 * For a continuous Q2 element in two dimensions it is bounded by 25 entries per row, and that
 * bound is a property of the element, not of the mesh -- which is what makes the cost independent
 * of how fine the discretisation is.
 *
 * The evaluation is deliberately matrix-based, assembling small cell matrices with FEValues
 * rather than driving the matrix-free loop. Matrix-free processes cells in vectorised batches of
 * four to eight, so a handful of scattered cells would waste most of the lanes. This form also
 * happens to be the shape ECSW wants, since it accumulates per cell.
 */
template<int dim>
class RestrictedLaplace
{
public:
  RestrictedLaplace(Poisson::Operator<dim, 1, Number> const &            pde_operator,
                    unsigned int const                                   blocks_per_dim,
                    std::vector<dealii::types::global_dof_index> const & output_dofs)
    : blocks_per_dim(blocks_per_dim), output_dofs(output_dofs)
  {
    auto const & dof_handler = pde_operator.get_dof_handler();
    auto const & fe          = dof_handler.get_fe();

    constraints = &pde_operator.get_matrix_free()->get_affine_constraints(
      pde_operator.get_dof_index());

    std::set<dealii::types::global_dof_index> requested(output_dofs.begin(), output_dofs.end());
    AssertThrow(requested.size() == output_dofs.size(),
                dealii::ExcMessage("The requested degrees of freedom must be distinct."));

    unsigned int const dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> local(dofs_per_cell);

    // Collect the cells that touch a requested degree of freedom. A row of the operator only
    // receives contributions from cells containing that degree of freedom, so this set is
    // exactly what is needed and nothing more.
    std::set<dealii::types::global_dof_index> source_set;
    for(auto const & cell : dof_handler.active_cell_iterators())
    {
      if(not cell->is_locally_owned())
        continue;

      cell->get_dof_indices(local);

      bool touches = false;
      for(auto const index : local)
        if(requested.count(index))
        {
          touches = true;
          break;
        }

      if(touches)
      {
        cells.push_back(cell);
        source_set.insert(local.begin(), local.end());
      }
    }

    source_dofs.assign(source_set.begin(), source_set.end());

    std::map<dealii::types::global_dof_index, unsigned int> source_position;
    for(unsigned int i = 0; i < source_dofs.size(); ++i)
      source_position[source_dofs[i]] = i;

    std::map<dealii::types::global_dof_index, unsigned int> output_position;
    for(unsigned int i = 0; i < output_dofs.size(); ++i)
      output_position[output_dofs[i]] = i;

    // Precompute, per cell, where each local degree of freedom reads from and writes to. The
    // gather and scatter maps do not depend on the parameter, so they are built once.
    for(auto const & cell : cells)
    {
      cell->get_dof_indices(local);

      CellMap map;
      map.gather.resize(dofs_per_cell);
      map.scatter.resize(dofs_per_cell);
      map.constrained.resize(dofs_per_cell);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        map.gather[i]      = source_position.at(local[i]);
        map.constrained[i] = constraints->is_constrained(local[i]);

        auto const found = output_position.find(local[i]);
        map.scatter[i]   = (found == output_position.end())
                             ? dealii::numbers::invalid_unsigned_int
                             : found->second;
      }

      cell_maps.push_back(map);
    }

    // Precompute each cell's contribution, split by block.
    //
    // The operator is affine in the block diffusivities, and that affinity holds cell by cell:
    // the contribution of a cell to block p is a fixed matrix, independent of the parameter. So
    // the whole geometric part -- the quadrature, the shape gradients, the Jacobians -- is
    // evaluated once here and never again. Applying the operator afterwards is dense arithmetic
    // on matrices of size (dofs per cell) squared.
    //
    // This is also precisely the structure ECSW assumes: a per-element contribution scaled by a
    // weight. Building it here means the DEIM and ECSW routes share an implementation.
    dealii::QGauss<dim> const quadrature(fe.degree + 1);

    dealii::FEValues<dim> fe_values(*pde_operator.get_mapping(),
                                    fe,
                                    quadrature,
                                    dealii::update_gradients | dealii::update_JxW_values |
                                      dealii::update_quadrature_points);

    unsigned int const n_blocks = dealii::Utilities::pow(blocks_per_dim, dim);

    for(unsigned int c = 0; c < cells.size(); ++c)
    {
      fe_values.reinit(cells[c]);

      std::map<unsigned int, std::vector<double>> per_block;

      for(unsigned int q = 0; q < quadrature.size(); ++q)
      {
        unsigned int const block =
          BlockCoefficient<dim>::block_index_of(blocks_per_dim, fe_values.quadrature_point(q));

        AssertThrow(block < n_blocks, dealii::ExcMessage("Block index out of range."));

        auto & matrix = per_block.try_emplace(block, dofs_per_cell * dofs_per_cell, 0.0)
                          .first->second;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            matrix[i * dofs_per_cell + j] +=
              (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)) * fe_values.JxW(q);
      }

      for(auto const & entry : per_block)
        cell_maps[c].contributions.push_back({entry.first, entry.second});
    }

    this->dofs_per_cell = dofs_per_cell;
  }

  std::vector<dealii::types::global_dof_index> const &
  get_source_dofs() const
  {
    return source_dofs;
  }

  unsigned int
  n_cells() const
  {
    return static_cast<unsigned int>(cells.size());
  }

  /**
   * The blocks whose coefficient the restricted operator actually reads, ascending.
   *
   * Every other block is invisible to it: no cell of the stencil lies in one, so changing its
   * coefficient cannot change any output value. That is the whole economy of hyper-reduction
   * and, when the blocks *are* the parameters, also its limitation -- so the set is exposed
   * rather than left implicit, both to build the reduced operator from and to measure how much
   * of the parameter space an interpolation actually sees.
   */
  std::vector<unsigned int>
  get_blocks() const
  {
    std::set<unsigned int> blocks;
    for(auto const & map : cell_maps)
      for(auto const & contribution : map.contributions)
        blocks.insert(contribution.block);

    return std::vector<unsigned int>(blocks.begin(), blocks.end());
  }

  /**
   * Applies the restricted operator.
   *
   * @param log_diffusivity One value per block.
   * @param source_values The vector's entries on the source degrees of freedom.
   * @return The operator's values on the output degrees of freedom.
   */
  std::vector<double>
  apply(std::vector<double> const & log_diffusivity,
        std::vector<double> const & source_values) const
  {
    std::vector<double> diffusivity(log_diffusivity.size());
    for(unsigned int p = 0; p < log_diffusivity.size(); ++p)
      diffusivity[p] = std::exp(log_diffusivity[p]);

    return apply_coefficients(diffusivity, source_values);
  }

  /**
   * Applies the restricted operator with the block coefficients given directly.
   *
   * Needed because a single affine component is the operator with the indicator of one block as
   * its coefficient, and an indicator cannot be expressed as the exponential of a log
   * diffusivity. This is the entry point pyMOR's empirical interpolation uses.
   */
  std::vector<double>
  apply_coefficients(std::vector<double> const & diffusivity,
                     std::vector<double> const & source_values) const
  {
    AssertThrow(source_values.size() == source_dofs.size(),
                dealii::ExcMessage("Expected " + std::to_string(source_dofs.size()) +
                                   " source values, got " +
                                   std::to_string(source_values.size()) + "."));

    std::vector<double> result(output_dofs.size(), 0.0);
    std::vector<double> local_values(dofs_per_cell);

    for(unsigned int c = 0; c < cell_maps.size(); ++c)
    {
      auto const & map = cell_maps[c];

      // The operator reads zero from constrained columns and acts as the identity on constrained
      // rows. Replicating both is what makes the restriction agree with the full apply exactly
      // rather than merely closely.
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        local_values[i] = map.constrained[i] ? 0.0 : source_values[map.gather[i]];

      for(auto const & contribution : map.contributions)
      {
        double const coefficient = diffusivity[contribution.block];

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          if(map.scatter[i] == dealii::numbers::invalid_unsigned_int or map.constrained[i])
            continue;

          double value = 0.0;
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            value += contribution.matrix[i * dofs_per_cell + j] * local_values[j];

          result[map.scatter[i]] += coefficient * value;
        }
      }
    }

    for(unsigned int i = 0; i < output_dofs.size(); ++i)
      if(constraints->is_constrained(output_dofs[i]))
      {
        auto const position =
          std::lower_bound(source_dofs.begin(), source_dofs.end(), output_dofs[i]);
        result[i] = source_values[position - source_dofs.begin()];
      }

    return result;
  }

private:
  struct Contribution
  {
    unsigned int        block;
    std::vector<double> matrix;
  };

  struct CellMap
  {
    std::vector<unsigned int> gather;
    std::vector<unsigned int> scatter;
    std::vector<bool>         constrained;
    std::vector<Contribution> contributions;
  };

  unsigned int                                 blocks_per_dim;
  std::vector<dealii::types::global_dof_index> output_dofs;
  std::vector<dealii::types::global_dof_index> source_dofs;

  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> cells;
  std::vector<CellMap>                                               cell_maps;

  dealii::AffineConstraints<Number> const * constraints;
  unsigned int                              dofs_per_cell;
};

/**
 * Full-order thermal block model, kept alive across calls so that the mesh, the matrix-free
 * data and the preconditioner are set up once.
 */
template<int dim>
class ThermalBlockFOM
{
public:
  ThermalBlockFOM(std::string const & input_file,
                  unsigned int const  degree,
                  unsigned int const  refinements,
                  bool const          verbose = false)
    : mpi_comm(MPI_COMM_WORLD)
  {
    ensure_mpi_initialized();

    SuppressOutput const suppress(not verbose);

    application = std::make_shared<Poisson::Application<dim, 1, Number>>(input_file, mpi_comm);
    application->set_parameters_convergence_study(degree, refinements);

    driver = std::make_unique<Poisson::Driver<dim, Number>>(mpi_comm,
                                                            application,
                                                            true /* is_test, silences output */,
                                                            false);
    driver->setup();

    pde_operator = driver->get_pde_operator();

    MassOperatorData<dim, Number> mass_data;
    mass_data.dof_index  = pde_operator->get_dof_index();
    mass_data.quad_index = pde_operator->get_quad_index();

    mass_operator.initialize(
      *pde_operator->get_matrix_free(),
      pde_operator->get_matrix_free()->get_affine_constraints(pde_operator->get_dof_index()),
      mass_data);

    // The finite element space is continuous, so the mass matrix is not block diagonal and the
    // element-local inverse does not apply. A global conjugate gradient solve with a point
    // Jacobi preconditioner is the right choice: the mass matrix is well conditioned, so this
    // converges in a handful of iterations independently of the mesh size.
    InverseMassOperatorData<Number> inverse_mass_data;
    inverse_mass_data.dof_index                     = pde_operator->get_dof_index();
    inverse_mass_data.quad_index                    = pde_operator->get_quad_index();
    inverse_mass_data.parameters.implementation_type = InverseMassType::GlobalKrylovSolver;
    inverse_mass_data.parameters.preconditioner      = PreconditionerMass::PointJacobi;

    // The default relative tolerance of 1e-6 is loose enough to limit the accuracy of any
    // residual based error estimator built on top of this. The mass solve is cheap, so it is
    // driven down to solver level instead.
    inverse_mass_data.parameters.solver_data.rel_tol = 1.e-12;

    inverse_mass_operator.initialize(
      *pde_operator->get_matrix_free(),
      inverse_mass_data,
      &pde_operator->get_matrix_free()->get_affine_constraints(pde_operator->get_dof_index()));

    sensors.setup(pde_operator->get_dof_handler().get_triangulation(),
                  *pde_operator->get_mapping(),
                  SensorOperator<dim, Number>::interior_cartesian_grid(
                    application->get_sensors_per_dim()));
  }

  unsigned int
  n_dofs() const
  {
    return pde_operator->get_number_of_dofs();
  }

  unsigned int
  n_blocks() const
  {
    return application->n_blocks();
  }

  /**
   * Degree of the diffusivity field: 0 for one value per block, 1 for a nodal field.
   */
  unsigned int
  coefficient_degree() const
  {
    return application->get_coefficient_degree();
  }

  /**
   * Number of parameters, i.e. of affine components.
   *
   * The blocks for a piecewise constant coefficient, the coefficient degrees of freedom
   * otherwise. This -- not n_blocks -- is what indexes apply_component_operator().
   */
  unsigned int
  n_parameters() const
  {
    return coefficient_degree() == 0 ? n_blocks()
                                     : static_cast<unsigned int>(
                                         pde_operator->n_coefficient_dofs());
  }

  /**
   * Support points of the coefficient degrees of freedom, flattened.
   *
   * What relates the parameter vector back to geometry when there is no block index to read it
   * off: an unstructured mesh has no lexicographic ordering, and a nodal field has no cells.
   */
  std::vector<double>
  coefficient_support_points() const
  {
    // At degree zero the parameters are blocks, whose centres follow from the lexicographic
    // indexing rather than from the mesh. Answering at every degree means a caller relating
    // parameters to geometry -- for a plot, say -- never has to branch, and so never has to
    // guess an ordering. Guessing it is the standard way to get a silently permuted field.
    if(coefficient_degree() == 0)
    {
      unsigned int const per_dim = application->get_blocks_per_dim();

      std::vector<double> flat(n_blocks() * dim);
      for(unsigned int block = 0; block < n_blocks(); ++block)
      {
        unsigned int remaining = block;
        for(unsigned int d = 0; d < dim; ++d)
        {
          flat[block * dim + d] = (remaining % per_dim + 0.5) / per_dim;
          remaining /= per_dim;
        }
      }

      return flat;
    }

    std::map<dealii::types::global_dof_index, dealii::Point<dim>> points;
    dealii::DoFTools::map_dofs_to_support_points(*pde_operator->get_mapping(),
                                                 pde_operator->get_coefficient_dof_handler(),
                                                 points);

    std::vector<double> flat(points.size() * dim);
    for(auto const & entry : points)
      for(unsigned int d = 0; d < dim; ++d)
        flat[entry.first * dim + d] = entry.second[d];

    return flat;
  }

  /**
   * Installs the diffusivity from its expansion coefficients, at any degree.
   *
   * The single entry point: ``a = sum_i c_i phi_i`` in the coefficient space, whatever that
   * space is. Degree 0 makes the entries cell values, degree 1 nodal values, and callers need
   * not know which -- the length is n_parameters() either way and the ordering is the one
   * n_parameters() and apply_component_operator() share.
   *
   * Branching on the degree at every call site was the alternative, and it is the kind of branch
   * that is fine until one place forgets it.
   */
  void
  set_diffusivity(std::vector<double> const & values)
  {
    // Installing the coefficient means a pass over every quadrature point, and a vector array
    // is applied column by column at one parameter, so the same field arrives repeatedly.
    if(values == current_parameters)
      return;

    // At degree zero the parameters are the *blocks*, which need not be the cells: Tier 1 puts
    // four of them on a sixteen-by-sixteen mesh. Only BlocksPerDim = 0 makes the two coincide.
    if(coefficient_degree() == 0)
      install_block_diffusivity(values);
    else
      set_coefficient_dofs(values);

    current_parameters = values;
  }

  /**
   * Solves with the coefficient currently installed and an arbitrary right-hand side.
   *
   * The counterpart of apply_current() for the inverse, and what pyMOR's solver hook needs once
   * the operator is driven by parameters rather than by block values. The preconditioner is
   * rebuilt only when the coefficient has actually changed, since rebuilding is the expensive
   * part and a vector array repeats the same parameter for every column.
   */
  std::shared_ptr<VectorType>
  apply_inverse_current(VectorType const & rhs)
  {
    if(current_parameters != preconditioned_parameters)
    {
      pde_operator->update_preconditioner();
      preconditioned_parameters = current_parameters;
    }

    auto solution = zero_vector();
    pde_operator->solve(*solution, rhs, 0.0 /* time */);

    return solution;
  }

  /**
   * Installs the diffusivity from the degrees of freedom of a coefficient space of degree >= 1.
   *
   * Prefer set_diffusivity(), which is the same thing without the degree restriction.
   */
  void
  set_coefficient_dofs(std::vector<double> const & values)
  {
    AssertThrow(coefficient_degree() > 0,
                dealii::ExcMessage("The coefficient is piecewise constant; use "
                                   "set_diffusivity() or set_cell_diffusivity()."));

    AssertThrow(values.size() == pde_operator->n_coefficient_dofs(),
                dealii::ExcMessage(
                  "Expected " + std::to_string(pde_operator->n_coefficient_dofs()) +
                  " coefficient values, got " + std::to_string(values.size()) + "."));

    VectorType vector;
    pde_operator->initialize_coefficient_dof_vector(vector);
    for(unsigned int i = 0; i < values.size(); ++i)
      vector[i] = values[i];

    pde_operator->set_coefficient_from_dof_vector(vector);

    note_coefficient_changed();
  }

  /**
   * Applies the affine component of parameter @p index.
   *
   * The unified form. For a piecewise constant coefficient the component is the operator with
   * the indicator of one block as its coefficient; for a nodal field it is the operator with a
   * single basis function as its coefficient. Both make
   *
   *     A(c) = sum_i c_i A_i
   *
   * exact, which is the identity the whole reduced model rests on -- and which survives the
   * change of degree because quadrature is linear in the coefficient.
   *
   * A basis function is not non-negative in general, so the *component* operator need not be
   * positive definite. That is fine: only the assembled ``A(c)`` has to be, and the positivity
   * check lives where the field is installed.
   *
   * The index is a *parameter* index: a lexicographic block at degree zero, a coefficient degree
   * of freedom above it. Both are the indexing set_diffusivity() takes and n_parameters()
   * counts, so a caller never has to know which. Note that this is *not* the triangulation's
   * cell order, which is a different permutation of the same set at degree zero.
   */
  void
  apply_component_operator(unsigned int const index, VectorType & dst, VectorType const & src)
  {
    AssertThrow(index < n_parameters(), dealii::ExcMessage("Component index out of range."));

    if(coefficient_degree() == 0)
    {
      apply_block_operator(index, dst, src);
      return;
    }

    VectorType indicator;
    pde_operator->initialize_coefficient_dof_vector(indicator);
    indicator = 0.0;
    indicator[index] = 1.0;

    // a single basis function, not a diffusivity: it is zero outside its support
    pde_operator->set_coefficient_from_dof_vector(indicator, false /* check_positivity */);

    note_coefficient_changed();

    pde_operator->vmult(dst, src);
  }

  /**
   * Applies ``sum_i c_i A_i`` for an arbitrary, possibly signed, coefficient vector, in one
   * cell loop.
   *
   * The affine identity ``A(c) = sum_i c_i A_i`` is linear in ``c``, so a whole recombination of
   * the affine components is the operator assembled with the recombined coefficient -- one cell
   * loop rather than ``n_parameters()`` of them. Measured at ``P = 289``: 0.13 ms against 52 ms
   * for the same answer through apply_component_operator(), a factor of four hundred.
   *
   * The intended use is the right-hand side of a **sensitivity solve**. Differentiating
   * ``A(m) u = f`` along a parameter direction ``d`` gives
   *
   *     A(m) (du/dm . d) = -(sum_i d_i exp(m_i) A_i) u,
   *
   * so the whole right-hand side is one call with ``c_i = d_i exp(m_i)``, and ``d`` is a
   * *direction* -- signed, which is why the positivity check that guards set_diffusivity() is
   * bypassed here. Only the assembled ``A(m)`` of an actual solve has to be positive definite.
   *
   * The convective term is deliberately absent: it carries no parameter dependence, so it does
   * not appear in a derivative of ``A(m) = C + sum_p exp(m_p) A_p``.
   *
   * **The coefficient found on entry is put back verbatim**, together with every cache that
   * describes it. A caller can therefore interleave this with apply_inverse_current() without
   * losing the preconditioner -- which is the whole point, since rebuilding it costs three times
   * a solve, and a sensitivity basis wants tens of solves at one parameter.
   */
  void
  apply_coefficient_operator(std::vector<double> const & coefficients,
                             VectorType &                dst,
                             VectorType const &          src)
  {
    AssertThrow(coefficients.size() == n_parameters(),
                dealii::ExcMessage("Expected " + std::to_string(n_parameters()) +
                                   " coefficients, got " +
                                   std::to_string(coefficients.size()) + "."));

    AssertThrow(not current_parameters.empty(),
                dealii::ExcMessage("apply_coefficient_operator() restores the coefficient it "
                                   "finds installed; install one through set_diffusivity() "
                                   "first."));

    std::vector<double> const restore_parameters = current_parameters;
    std::vector<double> const restore_log        = last_log_diffusivity;

    install_signed_coefficient(coefficients);

    pde_operator->vmult(dst, src);

    install_signed_coefficient(restore_parameters);

    // The field installed is the one that was there, so the operator is unchanged and every
    // cache describing it is valid again. Restoring them is what keeps the round trip free.
    current_parameters   = restore_parameters;
    last_log_diffusivity = restore_log;
  }

  /**
   * Blocks per coordinate direction, with a per-cell request already resolved.
   *
   * The Python side needs it to lay the prior out on the same grid the coefficient uses; a
   * mismatch would permute the parameters silently.
   */
  unsigned int
  blocks_per_dim() const
  {
    return application->get_blocks_per_dim();
  }

  unsigned int
  n_sensors() const
  {
    return sensors.n_points();
  }

  std::shared_ptr<VectorType>
  zero_vector() const
  {
    auto vector = std::make_shared<VectorType>();
    pde_operator->initialize_dof_vector(*vector);
    *vector = 0.0;

    return vector;
  }

  std::shared_ptr<VectorType>
  vector_from_numpy(py::array_t<double, py::array::c_style | py::array::forcecast> const & data)
  {
    AssertThrow(static_cast<unsigned int>(data.size()) == n_dofs(),
                dealii::ExcMessage("Expected " + std::to_string(n_dofs()) + " entries, got " +
                                   std::to_string(data.size()) + "."));

    auto vector = zero_vector();

    // Every rank receives the whole array and keeps the slice it owns. That is pyMOR's model
    // under mpi.call: the same call runs on every rank with the same arguments, so writing a
    // non-owned entry would be both wrong and a deal.II assertion.
    auto view = data.template unchecked<1>();
    for(auto const index : vector->locally_owned_elements())
      (*vector)[index] = view(index);

    vector->compress(dealii::VectorOperation::insert);

    return vector;
  }

  /**
   * Zeroes the constrained (Dirichlet) degrees of freedom.
   *
   * Must be applied to any vector that did not come out of a solve before it is handed to the
   * operators, because the affine decomposition only holds on the constrained subspace -- see
   * apply_block_operator().
   */
  void
  zero_constrained(VectorType & vector) const
  {
    pde_operator->get_matrix_free()
      ->get_affine_constraints(pde_operator->get_dof_index())
      .set_zero(vector);
  }

  /**
   * Applies the affine component A_p, i.e. the operator assembled with the indicator function
   * of block p as coefficient. This is what the reduced operators are projected from.
   *
   * IMPORTANT: the affine identity A(mu) = sum_p exp(mu_p) A_p holds only on vectors whose
   * constrained degrees of freedom vanish. On a constrained row each A_p acts as the identity,
   * so summing P components multiplies that entry by sum_p exp(mu_p) instead of leaving it
   * alone. Snapshots and reduced basis vectors satisfy the constraints by construction, so the
   * reduced operators are unaffected; arbitrary vectors must be passed through
   * zero_constrained() first.
   */
  void
  apply_block_operator(unsigned int const p, VectorType & dst, VectorType const & src)
  {
    AssertThrow(p < n_blocks(), dealii::ExcMessage("Block index out of range."));

    std::vector<double> indicator(n_blocks(), 0.0);
    indicator[p] = 1.0;

    install_block_diffusivity(indicator);
    current_parameters.clear();

    pde_operator->vmult(dst, src);
  }

  /**
   * Applies A(mu) = sum_p exp(mu_p) A_p directly, without going through the affine components.
   * Useful as an independent check of the affine decomposition.
   */
  void
  apply_operator(std::vector<double> const & log_diffusivity,
                 VectorType &                dst,
                 VectorType const &          src)
  {
    set_coefficient(log_diffusivity);

    apply_current(dst, src);
  }

  void
  apply_mass(VectorType & dst, VectorType const & src) const
  {
    mass_operator.vmult(dst, src);
  }

  /**
   * Applies the inverse of the mass matrix.
   *
   * Needed wherever a Riesz representative is formed: least-squares (LSPG) projection, which is
   * what a transport dominated problem requires because Galerkin projection is unstable there,
   * and residual based error estimation, which is the basis of certified reduced models.
   */
  std::shared_ptr<VectorType>
  apply_inverse_mass(VectorType const & src) const
  {
    auto dst = zero_vector();
    inverse_mass_operator.apply(*dst, src);

    return dst;
  }

  std::shared_ptr<VectorType>
  rhs_vector() const
  {
    auto vector = zero_vector();
    pde_operator->rhs(*vector);

    return vector;
  }

  /**
   * Full-order solve at the given log-diffusivities.
   *
   * The preconditioner is rebuilt for the new coefficient; without that the iteration count
   * degrades sharply once the contrast grows.
   */
  std::shared_ptr<VectorType>
  solve(std::vector<double> const & log_diffusivity)
  {
    prepare_solver(log_diffusivity);

    auto rhs = rhs_vector();

    auto solution = zero_vector();
    pde_operator->solve(*solution, *rhs, 0.0 /* time */);

    return solution;
  }

  /**
   * Solves J(mu) x = rhs for an arbitrary right-hand side.
   *
   * For this linear problem the Jacobian is the operator itself, so this is a plain linear
   * solve. It carries the nonlinear name on purpose: the operator contract a reduced-order
   * model needs is residual / apply_jacobian / apply_inverse_jacobian, and keeping that
   * vocabulary here means the nonlinear extension replaces the body rather than the interface.
   *
   * Without this, pyMOR has no way to invert the operator: it falls back to converting to a
   * NumPy matrix, which cannot work for a vector type it never sees the entries of, and every
   * algorithm built on solving the full-order problem -- greedy basis generation, residual-based
   * error estimation, least-squares (LSPG) projection -- fails with an InversionError.
   *
   * On constrained degrees of freedom the operator is the identity, so the solution equals the
   * right-hand side there. That is consistent with vectors entering through the Python layer
   * being zeroed on constrained entries.
   */
  std::shared_ptr<VectorType>
  apply_inverse_jacobian(std::vector<double> const & log_diffusivity, VectorType const & rhs)
  {
    prepare_solver(log_diffusivity);

    auto solution = zero_vector();
    pde_operator->solve(*solution, rhs, 0.0 /* time */);

    return solution;
  }

  std::vector<double>
  observe(VectorType const & solution) const
  {
    return sensors.evaluate(pde_operator->get_dof_handler(), solution);
  }

  /**
   * The observation operator's transpose, ``B^T w``.
   *
   * pyMOR needs this as ``Operator.apply_adjoint`` of the output functional: the residual-based
   * output error estimator of CoerciveRBReductor forms the Riesz representative of ``B^T`` and
   * fails without it. It is also the adjoint right-hand side of any functional of the sensor
   * values, so a dual-weighted-residual reductor reaches the same entry point.
   *
   * The true transpose, with the constrained rows left in place; a solver that eliminates
   * Dirichlet rows has to zero them itself.
   */
  std::shared_ptr<VectorType>
  observe_transpose(std::vector<double> const & weights) const
  {
    auto result = zero_vector();
    sensors.evaluate_transpose(pde_operator->get_dof_handler(), weights, *result);

    return result;
  }

  /**
   * Moves the sensors, rebuilding the point-evaluation pattern.
   *
   * The placement is an argument of the study, not a property of the mesh, and both the forward
   * observation operator and its transpose are built from this one list -- so changing it
   * changes the adjoint with it, and nothing downstream has to be told.
   *
   * @param flat Sensor coordinates, dim entries per point, point index running slowest.
   */
  void
  set_sensor_points(std::vector<double> const & flat)
  {
    AssertThrow(flat.size() % dim == 0,
                dealii::ExcMessage("Expected dim coordinates per sensor point, got " +
                                   std::to_string(flat.size()) + " entries in " +
                                   std::to_string(dim) + "D."));

    std::vector<dealii::Point<dim>> points(flat.size() / dim);
    for(unsigned int i = 0; i < points.size(); ++i)
      for(unsigned int d = 0; d < dim; ++d)
        points[i][d] = flat[i * dim + d];

    sensors.setup(pde_operator->get_dof_handler().get_triangulation(),
                  *pde_operator->get_mapping(),
                  points);
  }

  /**
   * Builds the operator restricted to the given output degrees of freedom.
   *
   * The stencil is resolved here, in C++, because it is a question about the mesh: which cells
   * touch these degrees of freedom, and which other degrees of freedom do those cells carry.
   * Answering it in Python would mean exporting the sparsity pattern, which is a full-order
   * sized object and exactly what this architecture avoids.
   */
  std::shared_ptr<RestrictedLaplace<dim>>
  restricted(std::vector<dealii::types::global_dof_index> const & dofs) const
  {
    return std::make_shared<RestrictedLaplace<dim>>(*pde_operator,
                                                    application->get_blocks_per_dim(),
                                                    dofs);
  }

  /**
   * Installs a diffusivity given as one value per active cell.
   *
   * The route a coefficient takes when it is not tied to a Cartesian grid of blocks, which is
   * every unstructured mesh.
   */
  void
  set_cell_diffusivity(std::vector<double> const & values)
  {
    AssertThrow(values.size() == pde_operator->get_dof_handler().get_triangulation().n_active_cells(),
                dealii::ExcMessage(
                  "Expected one value per active cell (" +
                  std::to_string(
                    pde_operator->get_dof_handler().get_triangulation().n_active_cells()) +
                  "), got " + std::to_string(values.size()) + "."));

    pde_operator->set_coefficient_from_cell_values(values);

    // the block caches no longer describe what is installed, and must not be allowed to skip a
    // later coefficient fill or preconditioner rebuild on the strength of a stale comparison
    note_coefficient_changed();
  }

  /**
   * Applies the operator with whatever coefficient is currently installed.
   *
   * apply_operator() takes its coefficient as an argument and so overwrites what is there, which
   * makes it useless after set_cell_diffusivity(). This is the counterpart that leaves the
   * coefficient alone.
   */
  void
  apply_current(VectorType & dst, VectorType const & src) const
  {
    pde_operator->vmult(dst, src);
  }

  /**
   * Solves with whatever coefficient is currently installed, rebuilding the preconditioner.
   */
  std::shared_ptr<VectorType>
  solve_current()
  {
    auto rhs = rhs_vector();

    pde_operator->update_preconditioner();

    auto solution = zero_vector();
    pde_operator->solve(*solution, *rhs, 0.0 /* time */);

    return solution;
  }

  std::vector<double>
  sensor_coordinates() const
  {
    std::vector<double> flat;
    for(auto const & point : sensors.get_points())
      for(unsigned int d = 0; d < dim; ++d)
        flat.push_back(point[d]);

    return flat;
  }

private:

  /**
   * Sets the coefficient and rebuilds the preconditioner when it has changed.
   *
   * Rebuilding keeps the iteration count from degrading as the contrast between blocks grows,
   * but it is also the expensive part of a solve. Applying the inverse to a whole vector array
   * repeats the same parameter for every column, so caching turns k setups into one.
   */
  void
  prepare_solver(std::vector<double> const & log_diffusivity)
  {
    set_coefficient(log_diffusivity);

    if(log_diffusivity != last_log_diffusivity)
    {
      pde_operator->update_preconditioner();
      last_log_diffusivity = log_diffusivity;
    }
  }

  void
  set_coefficient(std::vector<double> const & log_diffusivity)
  {
    AssertThrow(log_diffusivity.size() == n_blocks(),
                dealii::ExcMessage("Expected " + std::to_string(n_blocks()) +
                                   " log-diffusivities, got " +
                                   std::to_string(log_diffusivity.size()) + "."));

    std::vector<double> diffusivity(log_diffusivity.size());
    for(unsigned int p = 0; p < log_diffusivity.size(); ++p)
      diffusivity[p] = std::exp(log_diffusivity[p]);

    install_block_diffusivity(diffusivity);
    current_parameters.clear();
  }

  /**
   * Installs the *block* diffusivities, skipping the work when they have not changed.
   *
   * Setting the coefficient evaluates a function at every quadrature point of every cell, which
   * for the P0 field costs more than the matrix-vector product it precedes. Projecting the
   * affine decomposition applies each of the P components to every basis vector, repeating the
   * same coefficient r times in a row, so the cache turns P*r coefficient fills into P.
   */
  void
  install_block_diffusivity(std::vector<double> const & diffusivity)
  {
    if(diffusivity == current_diffusivity)
      return;

    pde_operator->set_coefficient(
      BlockCoefficient<dim>(application->get_blocks_per_dim(), diffusivity));

    current_diffusivity = diffusivity;
  }

  /**
   * Installs a coefficient vector that need not be positive, at either degree.
   *
   * The unsigned entry points -- set_diffusivity(), set_coefficient_dofs() -- check positivity,
   * because a diffusivity that changes sign is a broken model rather than an unusual one. A
   * *derivative* direction is a different object and is signed by nature, so it needs the same
   * installation without the check. Kept private: the public surface for it is
   * apply_coefficient_operator(), which puts the field back afterwards.
   */
  void
  install_signed_coefficient(std::vector<double> const & values)
  {
    if(coefficient_degree() == 0)
    {
      install_block_diffusivity(values);
      current_parameters.clear();

      return;
    }

    VectorType vector;
    pde_operator->initialize_coefficient_dof_vector(vector);
    for(unsigned int i = 0; i < values.size(); ++i)
      vector[i] = values[i];

    pde_operator->set_coefficient_from_dof_vector(vector, false /* check_positivity */);

    note_coefficient_changed();
  }

  /**
   * Records that the coefficient installed in the operator has changed.
   *
   * Every cache below describes the operator as it was, and each of them is consulted to *skip*
   * work: the coefficient fill, the preconditioner rebuild, the factorisation. Clearing them
   * together, in one place, is what keeps a new coefficient from being solved with a stale
   * factorisation -- which produces a perfectly ordinary-looking wrong answer, and which
   * separate invalidations at four call sites will eventually miss.
   */
  void
  note_coefficient_changed()
  {
    current_diffusivity.clear();
    last_log_diffusivity.clear();
    current_parameters.clear();
  }

  MPI_Comm mpi_comm;

  std::shared_ptr<Poisson::Application<dim, 1, Number>> application;
  std::unique_ptr<Poisson::Driver<dim, Number>>         driver;
  std::shared_ptr<Poisson::Operator<dim, 1, Number>>    pde_operator;

  MassOperator<dim, 1, Number>        mass_operator;
  InverseMassOperator<dim, 1, Number> inverse_mass_operator;

  SensorOperator<dim, Number> sensors;

  // parameter the preconditioner was last built for; empty until the first solve
  std::vector<double> last_log_diffusivity;

  // diffusivities currently installed in the operator; empty until the first assignment
  std::vector<double> current_diffusivity;

  // parameters last installed through set_diffusivity(), and those the preconditioner was built
  // for; empty means "unknown", which forces the work rather than skipping it
  std::vector<double> current_parameters;
  std::vector<double> preconditioned_parameters;
};

/**
 * Registers the model for one space dimension.
 */
template<int dim>
void
register_model(py::module_ & module, std::string const & name)
{
  py::class_<ThermalBlockFOM<dim>, std::shared_ptr<ThermalBlockFOM<dim>>>(module, name.c_str())
    .def(py::init<std::string const &, unsigned int, unsigned int, bool>(),
         py::arg("input_file"),
         py::arg("degree"),
         py::arg("refinements"),
         py::arg("verbose") = false,
         "Set up the full-order model from an ExaDG input file. Setup writes ExaDG's usual "
         "banner and parameter list to standard output only when verbose is set.")
    .def_property_readonly("n_dofs", &ThermalBlockFOM<dim>::n_dofs)
    .def_property_readonly("n_blocks", &ThermalBlockFOM<dim>::n_blocks)
    .def_property_readonly("n_parameters", &ThermalBlockFOM<dim>::n_parameters)
    .def_property_readonly("coefficient_degree", &ThermalBlockFOM<dim>::coefficient_degree)
    .def_property_readonly("blocks_per_dim", &ThermalBlockFOM<dim>::blocks_per_dim)
    .def_property_readonly("n_sensors", &ThermalBlockFOM<dim>::n_sensors)
    .def_property_readonly("dim", [](ThermalBlockFOM<dim> const &) { return dim; })
    .def("zero_vector", &ThermalBlockFOM<dim>::zero_vector)
    .def("vector_from_numpy", &ThermalBlockFOM<dim>::vector_from_numpy, py::arg("data"))
    .def("zero_constrained",
         &ThermalBlockFOM<dim>::zero_constrained,
         py::arg("vector"),
         "Zero the Dirichlet-constrained entries; required before applying the affine "
         "components to a vector that did not come from a solve.")
    .def("apply_block_operator",
         &ThermalBlockFOM<dim>::apply_block_operator,
         py::arg("block"),
         py::arg("dst"),
         py::arg("src"),
         "Apply the affine component A_p of block p.")
    .def("apply_component_operator",
         &ThermalBlockFOM<dim>::apply_component_operator,
         py::arg("index"),
         py::arg("dst"),
         py::arg("src"),
         "Apply the affine component of parameter index, at any coefficient degree.")
    .def("apply_coefficient_operator",
         &ThermalBlockFOM<dim>::apply_coefficient_operator,
         py::arg("coefficients"),
         py::arg("dst"),
         py::arg("src"),
         "Apply sum_i c_i A_i for an arbitrary, possibly signed, coefficient vector, in one "
         "cell loop. The right-hand side of a sensitivity solve is one call with "
         "c = direction * exp(mu). The installed coefficient and the preconditioner survive.")
    .def("set_coefficient_dofs",
         &ThermalBlockFOM<dim>::set_coefficient_dofs,
         py::arg("values"),
         "Install the diffusivity from its expansion coefficients in the coefficient space.")
    .def("coefficient_support_points",
         &ThermalBlockFOM<dim>::coefficient_support_points,
         "Support points of the coefficient degrees of freedom, flattened.")
    .def("apply_operator",
         &ThermalBlockFOM<dim>::apply_operator,
         py::arg("log_diffusivity"),
         py::arg("dst"),
         py::arg("src"))
    .def("apply_mass", &ThermalBlockFOM<dim>::apply_mass, py::arg("dst"), py::arg("src"))
    .def("apply_inverse_mass", &ThermalBlockFOM<dim>::apply_inverse_mass, py::arg("src"))
    .def("rhs_vector", &ThermalBlockFOM<dim>::rhs_vector)
    .def("solve", &ThermalBlockFOM<dim>::solve, py::arg("log_diffusivity"))
    .def("apply_inverse_jacobian",
         &ThermalBlockFOM<dim>::apply_inverse_jacobian,
         py::arg("log_diffusivity"),
         py::arg("rhs"))
    .def("observe", &ThermalBlockFOM<dim>::observe, py::arg("solution"))
    .def("observe_transpose",
         &ThermalBlockFOM<dim>::observe_transpose,
         py::arg("weights"),
         "The observation operator's transpose B^T w. This is Operator.apply_adjoint of the "
         "output functional, which pyMOR's output error estimator requires.")
    .def("sensor_coordinates", &ThermalBlockFOM<dim>::sensor_coordinates)
    // keep_alive<0, 1> ties the returned object's lifetime to the model's. It holds raw
    // references into the triangulation and the matrix-free data, so a model collected while
    // it is still in use leaves dangling pointers.
    .def("restricted",
         &ThermalBlockFOM<dim>::restricted,
         py::arg("dofs"),
         py::keep_alive<0, 1>())
    .def("set_diffusivity",
         &ThermalBlockFOM<dim>::set_diffusivity,
         py::arg("values"),
         "Install the diffusivity from its expansion coefficients, at any coefficient degree. "
         "One value per parameter, in the ordering n_parameters() counts.")
    .def("set_cell_diffusivity",
         &ThermalBlockFOM<dim>::set_cell_diffusivity,
         py::arg("values"),
         "Install a diffusivity given as one value per active cell, in deal.II's cell order.")
    .def("apply_current",
         &ThermalBlockFOM<dim>::apply_current,
         py::arg("dst"),
         py::arg("src"),
         "Apply the operator with the coefficient currently installed, leaving it alone.")
    .def("apply_inverse_current",
         &ThermalBlockFOM<dim>::apply_inverse_current,
         py::arg("rhs"),
         "Solve with the coefficient currently installed, for an arbitrary right-hand side.")
    .def("set_sensor_points",
         &ThermalBlockFOM<dim>::set_sensor_points,
         py::arg("points"),
         "Move the sensors. dim coordinates per point, point index slowest. The forward "
         "observation operator and its transpose both follow.")
    .def("solve_current",
         &ThermalBlockFOM<dim>::solve_current,
         "Solve with the coefficient currently installed, rebuilding the preconditioner.");

  py::class_<RestrictedLaplace<dim>, std::shared_ptr<RestrictedLaplace<dim>>>(
    module, ("RestrictedLaplace" + std::to_string(dim) + "D").c_str())
    .def_property_readonly("source_dofs", &RestrictedLaplace<dim>::get_source_dofs)
    .def_property_readonly("n_cells", &RestrictedLaplace<dim>::n_cells)
    .def_property_readonly("blocks",
                           &RestrictedLaplace<dim>::get_blocks,
                           "Blocks whose coefficient this restriction reads; every other block "
                           "cannot influence its output.")
    .def("apply",
         &RestrictedLaplace<dim>::apply,
         py::arg("log_diffusivity"),
         py::arg("source_values"))
    .def("apply_coefficients",
         &RestrictedLaplace<dim>::apply_coefficients,
         py::arg("diffusivity"),
         py::arg("source_values"));
}

} // namespace ExaDG

PYBIND11_MODULE(thermal_block, module)
{
  using namespace ExaDG;

  module.doc() =
    "Thermal block full-order model, exposed for pyMOR. Degree-of-freedom sized data stays in "
    "C++; only scalars, small arrays and opaque vector handles cross into Python.";

  // Initialise MPI when the module is imported rather than lazily in a constructor. deal.II
  // objects call MPI from their constructors, and once one of them is a class member the
  // ordering becomes hard to reason about; doing it here removes the question entirely.
  ensure_mpi_initialized();

  module.def(
    "mpi_initialized",
    []() {
      int flag = 0;
      MPI_Initialized(&flag);

      return flag != 0;
    },
    "Whether MPI_Init has run. Diagnostic.");

  module.def(
    "mpi_size",
    []() { return dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); },
    "Number of MPI ranks.");

  // The distributed vector is exposed as an opaque handle carrying exactly the arithmetic
  // pyMOR's ListVectorArray requires. Everything here is a thin forward to deal.II.
  py::class_<VectorType, std::shared_ptr<VectorType>>(module, "Vector")
    .def_property_readonly("dim", [](VectorType const & v) { return v.size(); })
    .def("copy",
         [](VectorType const & v) { return std::make_shared<VectorType>(v); },
         "Deep copy.")
    .def("scal", [](VectorType & v, double const a) { v *= a; }, py::arg("alpha"))
    .def(
      "axpy",
      [](VectorType & v, double const a, VectorType const & x) { v.add(a, x); },
      py::arg("alpha"),
      py::arg("x"))
    .def(
      "inner",
      [](VectorType const & v, VectorType const & x) { return v * x; },
      py::arg("other"),
      "Euclidean inner product; reduces over the communicator internally.")
    .def("norm", [](VectorType const & v) { return v.l2_norm(); })
    .def("norm2", [](VectorType const & v) { return v.norm_sqr(); })
    .def("sup_norm", [](VectorType const & v) { return v.linfty_norm(); })
    .def(
      "amax",
      [](VectorType const & v) {
        // Index and magnitude of the largest entry, which is what empirical interpolation uses
        // to pick its next interpolation point. The index is *global*, because that is the
        // index space dofs() is addressed in and the one an interpolation point has to survive
        // in; returning a local index would be silently wrong on more than one rank.
        //
        // Two reductions rather than one MPI_MAXLOC: a maximum over the values, then a minimum
        // over the global indices of the ranks that attain it. That costs one extra allreduce
        // and buys a deterministic tie-break -- the smallest global index always wins, whatever
        // the partitioning. MAXLOC would break ties by rank, so the interpolation point chosen
        // would depend on the number of ranks, and a basis built on two ranks would differ from
        // one built on four.
        auto const & partitioner = *v.get_partitioner();

        double local_value = 0.0;
        for(unsigned int i = 0; i < v.locally_owned_size(); ++i)
          local_value = std::max(local_value, std::abs(v.local_element(i)));

        double const value = dealii::Utilities::MPI::max(local_value, v.get_mpi_communicator());

        auto local_index = dealii::numbers::invalid_dof_index;
        for(unsigned int i = 0; i < v.locally_owned_size(); ++i)
          if(std::abs(v.local_element(i)) == value)
          {
            local_index = partitioner.local_to_global(i);
            break;
          }

        auto const index = dealii::Utilities::MPI::min(local_index, v.get_mpi_communicator());

        return py::make_tuple(index, value);
      })
    .def(
      "dofs",
      [](VectorType const & v, std::vector<dealii::types::global_dof_index> const & indices) {
        // Collective. pyMOR reaches this through mpi.call, which runs the same call on every
        // rank and keeps rank 0's return value, so every rank has to come back with the whole
        // answer rather than with its own slice.
        //
        // Exactly one rank owns each index, so a sum over ranks with zero from the others
        // gathers the values without anyone needing to know who owns what.
        auto const & partitioner = *v.get_partitioner();

        std::vector<double> values(indices.size(), 0.0);
        for(unsigned int i = 0; i < indices.size(); ++i)
          if(partitioner.in_local_range(indices[i]))
            values[i] = v.local_element(partitioner.global_to_local(indices[i]));

        std::vector<double> gathered(indices.size());
        dealii::Utilities::MPI::sum(values, v.get_mpi_communicator(), gathered);

        return gathered;
      },
      py::arg("indices"),
      "Selected entries by global index, as pyMOR's empirical interpolation requires. "
      "Collective: every rank returns the same values.")
    .def(
      "to_numpy",
      [](VectorType const & v) {
        // Refuses rather than returning the local slice. pyMOR never calls this on more than
        // one rank -- MPIVectorArrayImpl.to_numpy raises -- so a caller who gets here in
        // parallel is a test or a debug print that would otherwise silently compare slices.
        AssertThrow(dealii::Utilities::MPI::n_mpi_processes(v.get_mpi_communicator()) == 1,
                    dealii::ExcMessage("to_numpy() would gather the whole vector onto every "
                                       "rank, which is exactly what this interface exists to "
                                       "avoid. Use dofs() for selected entries."));

        py::array_t<double> array(v.size());
        auto                view = array.mutable_unchecked<1>();
        for(unsigned int i = 0; i < v.size(); ++i)
          view(i) = v[i];

        return array;
      },
      "Copy into a NumPy array. Single rank only; for debugging and tests.");

  register_model<2>(module, "ThermalBlockFOM2D");
  register_model<3>(module, "ThermalBlockFOM3D");
}
