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
 * The thermal block as a PyMOR::FullOrderModel.
 *
 * All the pyMOR-facing structure is declared through exadg/pymor/interface.h, so this file
 * contains only what is specific to *this* problem: a Poisson operator whose diffusivity is
 * piecewise constant on a Cartesian grid of blocks, point sensors as the quantity of interest,
 * and the caches that keep a parameter sweep from refilling coefficients it already has.
 *
 * Nothing here knows about pyMOR. The Python layer wraps the interface classes below without
 * knowing what they discretise, and python/exadg/mor/models/stationary.py assembles them into a
 * pyMOR StationaryModel. Adding an application means writing a file like this one and nothing
 * else.
 *
 * The vector type, the operator base classes and MPI initialisation come from exadg._core, which
 * is imported below. They are bound there rather than here because pybind11's type registry is
 * process-global: two application modules binding the same C++ vector type abort on import.
 *
 * MPI: supported through pyMOR's event loop (pymor.tools.mpi), where Python runs on every rank
 * and rank 0 dispatches. Every call below therefore executes simultaneously on all ranks, and
 * pyMOR keeps rank 0's return value -- so anything that returns data has to return the global
 * answer rather than this rank's slice. The restricted operator is the one thing that is serial,
 * and it says so by returning nullptr rather than by being guarded in Python.
 */

// C/C++
#include <iostream>
#include <sstream>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>

// ExaDG
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/poisson/driver.h>
#include <exadg/pymor/block_coefficient.h>
#include <exadg/pymor/interface.h>
#include <exadg/pymor/restricted_laplace.h>
#include <exadg/pymor/sensor_operator.h>
#include <exadg/utilities/create_directories.h>
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
 * Redirects std::cout for the lifetime of the object, so that constructing a model from a
 * notebook does not print ExaDG's banner and setup tables.
 */
class SuppressOutput
{
public:
  explicit SuppressOutput(bool const active) : buffer(nullptr)
  {
    if(active)
      buffer = std::cout.rdbuf(sink.rdbuf());
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
  std::ostringstream sink;
  std::streambuf *   buffer;
};

} // namespace

/**
 * A Poisson problem whose diffusivity is piecewise constant on a Cartesian grid of blocks.
 *
 *     -div(a(x; mu) grad u) = f,   a piecewise constant, one value per block
 *
 * Because the coefficient is constant per block, the operator is exactly affine in the block
 * values: A(c) = sum_p c_p A_p with A_p the operator assembled with the indicator of block p.
 * That identity is the reason this problem is the standard reduced-basis benchmark, and it is
 * what operator_components() below hands to pyMOR.
 *
 * **The affine identity holds only on the constrained subspace.** On a Dirichlet-constrained row
 * every component acts as the identity, so summing P of them scales that entry by the sum of the
 * coefficients rather than leaving it alone. Vectors that come out of a solve satisfy the
 * constraints; ones pyMOR builds itself -- random probes, interpolation candidates -- do not,
 * which is what make_admissible() is for.
 */
template<int dim>
class ThermalBlockFOM : public PyMOR::FullOrderModel<VectorType>
{
public:
  ThermalBlockFOM(std::string const & input_file,
                  unsigned int const  degree,
                  unsigned int const  refinements,
                  bool const          verbose = false)
    : mpi_comm(MPI_COMM_WORLD)
  {
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
    inverse_mass_data.dof_index                      = pde_operator->get_dof_index();
    inverse_mass_data.quad_index                     = pde_operator->get_quad_index();
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

  // ===========================================================================================
  //  The operators, as PyMOR::LinearOperator implementations
  // ===========================================================================================

  /**
   * The restricted Laplace operator, presented through the interface.
   *
   * Fixed coefficients unless it came from the parametric operator, in which case the Python
   * layer re-parameterises it per apply. Refusing in the fixed case rather than ignoring the
   * argument is deliberate: silently keeping the old coefficient would produce a restricted
   * operator that disagrees with the full one, and empirical interpolation would then converge
   * neatly to the wrong operator.
   */
  class Restricted : public PyMOR::RestrictedOperator
  {
  public:
    Restricted(std::shared_ptr<RestrictedLaplace<dim>> impl,
               std::vector<double>                     coefficients,
               bool const                              parametric)
      : impl(impl), coefficients(coefficients), parametric(parametric)
    {
    }

    std::vector<dealii::types::global_dof_index> const &
    get_source_dofs() const override
    {
      return impl->get_source_dofs();
    }

    std::vector<double>
    apply(std::vector<double> const & source_values) const override
    {
      return impl->apply_coefficients(coefficients, source_values);
    }

    std::vector<unsigned int>
    active_components() const override
    {
      return impl->get_blocks();
    }

    void
    set_coefficients(std::vector<double> const & values) override
    {
      AssertThrow(parametric,
                  dealii::ExcMessage("This restriction has fixed coefficients."));

      coefficients = values;
    }

  private:
    std::shared_ptr<RestrictedLaplace<dim>> impl;

    std::vector<double> coefficients;
    bool const          parametric;
  };

  /**
   * One affine component A_p: the operator assembled with the indicator of parameter p.
   *
   * Indexed by *parameter*, not by block. At coefficient degree zero those coincide; above it
   * the parameters are the coefficient's degrees of freedom. Either way the index is the one
   * parameter_shape() counts, so a caller never has to know which -- and never has to remember
   * a permutation.
   */
  class Component : public PyMOR::LinearOperator<VectorType>
  {
  public:
    Component(std::shared_ptr<ThermalBlockFOM<dim>> fom, unsigned int const index)
      : fom(fom), index(index)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->apply_component_operator(index, dst, src);
    }

    /// A symmetric interior penalty Laplacian; the transpose is the operator itself.
    bool
    is_symmetric() const override
    {
      return true;
    }

    std::shared_ptr<PyMOR::RestrictedOperator>
    restricted(std::vector<dealii::types::global_dof_index> const & output_dofs) const override
    {
      std::vector<double> indicator(fom->n_parameters(), 0.0);
      indicator[index] = 1.0;

      return fom->make_restricted(output_dofs, indicator, false /* parametric */);
    }

    std::string
    get_name() const override
    {
      return "A_" + std::to_string(index);
    }

  private:
    std::shared_ptr<ThermalBlockFOM<dim>> fom;

    unsigned int const index;
  };

  /**
   * The operator at fixed coefficients, carrying this application's solver.
   *
   * What makes the model solvable rather than merely multipliable: apply_inverse() hands the
   * system to ExaDG's preconditioned conjugate gradients, so greedy basis generation, residual
   * based error estimation and least-squares projection run against the real solver.
   */
  class Assembled : public PyMOR::LinearOperator<VectorType>
  {
  public:
    Assembled(std::shared_ptr<ThermalBlockFOM<dim>> fom,
              std::vector<double>                   coefficients,
              std::string                           name)
      : fom(fom), coefficients(coefficients), name(name)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->set_diffusivity(coefficients);
      fom->apply_current(dst, src);
    }

    bool
    is_symmetric() const override
    {
      return true;
    }

    bool
    has_inverse() const override
    {
      return true;
    }

    std::shared_ptr<VectorType>
    apply_inverse(VectorType const & rhs) const override
    {
      fom->set_diffusivity(coefficients);

      return fom->apply_inverse_current(rhs);
    }

    std::shared_ptr<PyMOR::RestrictedOperator>
    restricted(std::vector<dealii::types::global_dof_index> const & output_dofs) const override
    {
      return fom->make_restricted(output_dofs, coefficients, false /* parametric */);
    }

    std::string
    get_name() const override
    {
      return name;
    }

  private:
    std::shared_ptr<ThermalBlockFOM<dim>> fom;

    std::vector<double> const coefficients;
    std::string const         name;
  };

  /**
   * The same operator with its coefficients supplied at apply time.
   *
   * Mathematically identical to the affine form; what differs is what pyMOR can do with it.
   * Given the components, pyMOR projects each one exactly, at a cost of one full-order apply per
   * component per basis vector -- linear in a parameter dimension that grows with the mesh once
   * the coefficient is per cell. Given this instead, the parameter dependence is opaque and
   * pyMOR reaches for empirical interpolation, whose cost is set by the interpolation.
   *
   * The exchange is worth measuring rather than assuming: the restricted evaluation reads the
   * coefficient only on the cells touching its stencil, so the interpolated operator is exactly
   * insensitive to every parameter outside it. python/examples/thermal_block_ei.py measures that
   * against the affine reference.
   */
  class Field : public PyMOR::ParametricOperator<VectorType>
  {
  public:
    explicit Field(std::shared_ptr<ThermalBlockFOM<dim>> fom)
      : fom(fom), coefficients(fom->n_parameters(), 1.0)
    {
    }

    void
    set_coefficients(std::vector<double> const & values) override
    {
      AssertThrow(values.size() == fom->n_parameters(),
                  dealii::ExcMessage("Expected " + std::to_string(fom->n_parameters()) +
                                     " coefficients, got " + std::to_string(values.size()) + "."));

      coefficients = values;
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->set_diffusivity(coefficients);
      fom->apply_current(dst, src);
    }

    bool
    is_symmetric() const override
    {
      return true;
    }

    bool
    has_inverse() const override
    {
      return true;
    }

    std::shared_ptr<VectorType>
    apply_inverse(VectorType const & rhs) const override
    {
      fom->set_diffusivity(coefficients);

      return fom->apply_inverse_current(rhs);
    }

    std::shared_ptr<PyMOR::RestrictedOperator>
    restricted(std::vector<dealii::types::global_dof_index> const & output_dofs) const override
    {
      return fom->make_restricted(output_dofs, coefficients, true /* parametric */);
    }

    std::string
    get_name() const override
    {
      return "A(mu)";
    }

  private:
    std::shared_ptr<ThermalBlockFOM<dim>> fom;

    std::vector<double> coefficients;
  };

  /**
   * The mass matrix, i.e. the L2 inner product of the finite element space.
   *
   * The product a proper orthogonal decomposition should be taken in: the Euclidean inner
   * product of coefficient vectors weights degrees of freedom by the local mesh size and is not
   * an optimal basis in any norm of interest. The inverse is what forms a Riesz representative,
   * which residual-based error estimation and least-squares projection both need.
   */
  class Mass : public PyMOR::LinearOperator<VectorType>
  {
  public:
    explicit Mass(std::shared_ptr<ThermalBlockFOM<dim>> fom) : fom(fom)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->apply_mass(dst, src);
    }

    bool
    is_symmetric() const override
    {
      return true;
    }

    bool
    has_inverse() const override
    {
      return true;
    }

    std::shared_ptr<VectorType>
    apply_inverse(VectorType const & rhs) const override
    {
      return fom->apply_inverse_mass(rhs);
    }

    std::string
    get_name() const override
    {
      return "mass";
    }

  private:
    std::shared_ptr<ThermalBlockFOM<dim>> fom;
  };

  /**
   * Point sensors as the quantity of interest.
   *
   * Evaluated with RemotePointEvaluation, so the sensors may lie anywhere in the domain
   * irrespective of the mesh partitioning.
   */
  class Sensors : public PyMOR::Functional<VectorType>
  {
  public:
    explicit Sensors(std::shared_ptr<ThermalBlockFOM<dim>> fom) : fom(fom)
    {
    }

    unsigned int
    n_outputs() const override
    {
      return fom->n_sensors();
    }

    std::vector<double>
    apply(VectorType const & src) const override
    {
      return fom->observe(src);
    }

    std::shared_ptr<VectorType>
    apply_transpose(std::vector<double> const & weights) const override
    {
      return fom->observe_transpose(weights);
    }

  private:
    std::shared_ptr<ThermalBlockFOM<dim>> fom;
  };

  // ===========================================================================================
  //  PyMOR::FullOrderModel
  // ===========================================================================================

  dealii::types::global_dof_index
  n_dofs() const override
  {
    return pde_operator->get_number_of_dofs();
  }

  std::shared_ptr<VectorType>
  zero_vector() const override
  {
    auto vector = std::make_shared<VectorType>();
    pde_operator->initialize_dof_vector(*vector);
    *vector = 0.0;

    return vector;
  }

  /**
   * Zeroes the Dirichlet-constrained degrees of freedom, which is the subspace the affine
   * identity holds on.
   */
  void
  make_admissible(VectorType & vector) const override
  {
    pde_operator->get_matrix_free()
      ->get_affine_constraints(pde_operator->get_dof_index())
      .set_zero(vector);
  }

  /**
   * One parameter group holding the whole coefficient field.
   */
  std::vector<unsigned int>
  parameter_shape() const override
  {
    return {n_parameters()};
  }

  std::vector<PyMOR::AffineComponent<VectorType>>
  operator_components() override
  {
    auto const self = shared_self();

    std::vector<PyMOR::AffineComponent<VectorType>> components;
    for(unsigned int p = 0; p < n_parameters(); ++p)
      components.push_back({std::make_shared<Component>(self, p), 0 /* slot */, p});

    return components;
  }

  std::shared_ptr<PyMOR::ParametricOperator<VectorType>>
  parametric_operator() override
  {
    return std::make_shared<Field>(shared_self());
  }

  /**
   * The operator at these coefficients, or nullptr if it is not one this application can solve.
   *
   * pyMOR reaches here whenever it assembles a linear combination of the affine components, and
   * the combination may legitimately be signed -- a reductor forming a difference of operators
   * does exactly that. A diffusivity that changes sign is a broken model rather than an unusual
   * one, so those combinations are declined and pyMOR falls back to its generic path, which is
   * correct if slower. Deciding it here is the point: the coefficient means something to this
   * application and nothing to the Python layer.
   */
  std::shared_ptr<PyMOR::LinearOperator<VectorType>>
  assemble(std::vector<double> const & coefficients) override
  {
    if(coefficients.size() != n_parameters())
      return nullptr;

    for(auto const value : coefficients)
      if(not(value > 0.0))
        return nullptr;

    return std::make_shared<Assembled>(shared_self(), coefficients, "A(mu)");
  }

  std::shared_ptr<VectorType>
  rhs() override
  {
    auto vector = zero_vector();
    pde_operator->rhs(*vector);

    return vector;
  }

  /**
   * The mass matrix, and the operator at unit diffusivities.
   *
   * The second is the H1 seminorm, and it is the product a coercive residual error estimator has
   * to be taken in: A(mu) >= min_p mu_p * A(1) in the A(1) inner product, so the smallest
   * coefficient is then an exact lower bound on the coercivity constant. Against the mass
   * product the same functional is not a bound at all and the estimator comes out orders of
   * magnitude too large -- measured here at an effectivity of 2.4e-4 before the energy product
   * was added, against 0.12 to 0.16 after.
   */
  std::map<std::string, std::shared_ptr<PyMOR::LinearOperator<VectorType>>>
  products() override
  {
    auto const self = shared_self();

    return {{"mass", std::make_shared<Mass>(self)},
            {"energy",
             std::make_shared<Assembled>(self,
                                         std::vector<double>(n_parameters(), 1.0),
                                         "energy")}};
  }

  std::shared_ptr<PyMOR::Functional<VectorType>>
  output_functional() override
  {
    return std::make_shared<Sensors>(shared_self());
  }

  /**
   * Writes one or more fields as a VTU/PVTU record and returns the path of the record.
   *
   * deal.II's write_vtu_with_pvtu_record() produces one piece per rank plus a .pvtu that ParaView
   * opens as a single field, so the parallel case is the normal case rather than a special one.
   */
  std::string
  write_vtu(std::string const &                              directory,
            std::string const &                              basename,
            std::vector<std::shared_ptr<VectorType>> const & vectors,
            std::vector<std::string> const &                 names) const override
  {
    AssertThrow(vectors.size() == names.size(),
                dealii::ExcMessage("Expected one name per vector, got " +
                                   std::to_string(names.size()) + " for " +
                                   std::to_string(vectors.size()) + " vectors."));
    AssertThrow(not vectors.empty(), dealii::ExcMessage("Nothing to write."));

    // deal.II concatenates directory and file name verbatim, so a missing separator silently
    // writes "outputsolution_0.pvtu" next to the directory instead of inside it.
    std::string const path =
      (directory.empty() or directory.back() == '/') ? directory : directory + "/";

    create_directories(path, mpi_comm);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(pde_operator->get_dof_handler());

    // DataOut reads ghost entries, and it keeps a reference rather than a copy -- so the
    // ghosted vectors have to be sized up front and outlive build_patches().
    std::vector<VectorType> ghosted(vectors.size());
    for(unsigned int i = 0; i < vectors.size(); ++i)
    {
      pde_operator->initialize_dof_vector(ghosted[i]);
      ghosted[i] = *vectors[i];
      ghosted[i].update_ghost_values();

      data_out.add_data_vector(ghosted[i], names[i]);
    }

    data_out.build_patches(*pde_operator->get_mapping(),
                           pde_operator->get_dof_handler().get_fe().degree);

    // deal.II returns the record's name relative to the directory; prepend it so the caller
    // gets a path it can actually open.
    return path + data_out.write_vtu_with_pvtu_record(path, basename, 0, mpi_comm);
  }

  // ===========================================================================================
  //  Thermal-block specifics, exposed to Python for scripting and plots
  // ===========================================================================================

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
   * otherwise. This -- not n_blocks -- is what indexes the affine components.
   */
  unsigned int
  n_parameters() const
  {
    return coefficient_degree() == 0 ?
             n_blocks() :
             static_cast<unsigned int>(pde_operator->n_coefficient_dofs());
  }

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

  std::vector<double>
  sensor_coordinates() const
  {
    std::vector<double> flat;
    for(auto const & point : sensors.get_points())
      for(unsigned int d = 0; d < dim; ++d)
        flat.push_back(point[d]);

    return flat;
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
   * Installs a diffusivity given as one value per active cell.
   *
   * The route a coefficient takes when it is not tied to a Cartesian grid of blocks, which is
   * every unstructured mesh.
   */
  void
  set_cell_diffusivity(std::vector<double> const & values)
  {
    AssertThrow(
      values.size() == pde_operator->get_dof_handler().get_triangulation().n_active_cells(),
      dealii::ExcMessage(
        "Expected one value per active cell (" +
        std::to_string(pde_operator->get_dof_handler().get_triangulation().n_active_cells()) +
        "), got " + std::to_string(values.size()) + "."));

    pde_operator->set_coefficient_from_cell_values(values);

    // the block caches no longer describe what is installed, and must not be allowed to skip a
    // later coefficient fill or preconditioner rebuild on the strength of a stale comparison
    note_coefficient_changed();
  }

  // ===========================================================================================
  //  Used by the operators above
  // ===========================================================================================

  /**
   * Installs the diffusivity from its expansion coefficients, at any degree.
   *
   * The single entry point: a = sum_i c_i phi_i in the coefficient space, whatever that space
   * is. Degree 0 makes the entries cell values, degree 1 nodal values, and callers need not know
   * which -- the length is n_parameters() either way.
   *
   * Installing a coefficient means a pass over every quadrature point, and a vector array is
   * applied column by column at one parameter, so the same field arrives repeatedly. The cache
   * turns P*r fills into P.
   */
  void
  set_diffusivity(std::vector<double> const & values)
  {
    if(values == current_parameters)
      return;

    // At degree zero the parameters are the *blocks*, which need not be the cells: four blocks
    // on a sixteen-by-sixteen mesh is a normal configuration.
    if(coefficient_degree() == 0)
      install_block_diffusivity(values);
    else
      set_coefficient_dofs(values);

    current_parameters = values;
  }

  /**
   * Applies the operator with whatever coefficient is currently installed.
   */
  void
  apply_current(VectorType & dst, VectorType const & src) const
  {
    pde_operator->vmult(dst, src);
  }

  /**
   * Solves with the coefficient currently installed, for an arbitrary right-hand side.
   *
   * The preconditioner is rebuilt only when the coefficient has actually changed, since
   * rebuilding is the expensive part -- it costs about three solves -- and applying an inverse
   * to a vector array repeats the same parameter for every column.
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
   * Applies the affine component of the given parameter index.
   */
  void
  apply_component_operator(unsigned int const index, VectorType & dst, VectorType const & src)
  {
    AssertThrow(index < n_parameters(), dealii::ExcMessage("Component index out of range."));

    if(coefficient_degree() == 0)
    {
      std::vector<double> indicator(n_blocks(), 0.0);
      indicator[index] = 1.0;

      install_block_diffusivity(indicator);
      current_parameters.clear();

      pde_operator->vmult(dst, src);

      return;
    }

    VectorType indicator;
    pde_operator->initialize_coefficient_dof_vector(indicator);
    indicator        = 0.0;
    indicator[index] = 1.0;

    // a single basis function, not a diffusivity: it is zero outside its support
    pde_operator->set_coefficient_from_dof_vector(indicator, false /* check_positivity */);

    note_coefficient_changed();

    pde_operator->vmult(dst, src);
  }

  void
  apply_mass(VectorType & dst, VectorType const & src) const
  {
    mass_operator.vmult(dst, src);
  }

  std::shared_ptr<VectorType>
  apply_inverse_mass(VectorType const & src) const
  {
    auto dst = zero_vector();
    inverse_mass_operator.apply(*dst, src);

    return dst;
  }

  std::vector<double>
  observe(VectorType const & solution) const
  {
    return sensors.evaluate(pde_operator->get_dof_handler(), solution);
  }

  /**
   * The observation operator's transpose, with constrained rows left in place.
   */
  std::shared_ptr<VectorType>
  observe_transpose(std::vector<double> const & weights) const
  {
    auto result = zero_vector();
    sensors.evaluate_transpose(pde_operator->get_dof_handler(), weights, *result);

    return result;
  }

  /**
   * The operator restricted to the given output degrees of freedom, or nullptr.
   *
   * nullptr rather than an exception in the two cases this application cannot serve, because
   * that is how the interface declares a missing capability and pyMOR handles it: an operator
   * without a restriction makes empirical interpolation fall back to evaluating the full
   * operator, which is slower and correct, instead of producing a wrong one.
   *
   * **More than one rank.** The stencil is collected from locally owned cells only, so each rank
   * would build a different piece of it and none of them the whole operator. Making it parallel
   * means agreeing the cell set across ranks and gathering the evaluation, which is separate
   * work rather than a missing line.
   *
   * **A coefficient of degree above zero.** RestrictedLaplace precomputes each cell's
   * contribution per block, which presumes the coefficient is constant on a cell. With a nodal
   * field a cell carries several basis functions and the precomputation would have to be per
   * local degree of freedom instead.
   */
  std::shared_ptr<PyMOR::RestrictedOperator>
  make_restricted(std::vector<dealii::types::global_dof_index> const & output_dofs,
                  std::vector<double> const &                          coefficients,
                  bool const                                           parametric)
  {
    if(dealii::Utilities::MPI::n_mpi_processes(mpi_comm) > 1)
      return nullptr;

    if(coefficient_degree() > 0)
      return nullptr;

    auto impl = std::make_shared<RestrictedLaplace<dim>>(
      pde_operator->get_dof_handler(),
      *pde_operator->get_mapping(),
      pde_operator->get_matrix_free()->get_affine_constraints(pde_operator->get_dof_index()),
      application->get_blocks_per_dim(),
      output_dofs);

    return std::make_shared<Restricted>(impl, coefficients, parametric);
  }

private:
  /**
   * This model as a shared pointer of its own type.
   *
   * The operators above hold the model and outlive the call that produced them, so they need
   * ownership rather than a raw pointer. FullOrderModel derives from enable_shared_from_this and
   * pybind11 holds the instance in a shared_ptr, so the control block already exists; this only
   * recovers the derived type. There is no cycle -- the model does not store the operators.
   */
  std::shared_ptr<ThermalBlockFOM<dim>>
  shared_self()
  {
    return std::static_pointer_cast<ThermalBlockFOM<dim>>(this->shared_from_this());
  }

  /**
   * Installs the diffusivity from the degrees of freedom of a coefficient space of degree >= 1.
   */
  void
  set_coefficient_dofs(std::vector<double> const & values)
  {
    AssertThrow(values.size() == n_parameters(),
                dealii::ExcMessage("Expected " + std::to_string(n_parameters()) +
                                   " coefficient values, got " + std::to_string(values.size()) +
                                   "."));

    VectorType vector;
    pde_operator->initialize_coefficient_dof_vector(vector);
    for(unsigned int i = 0; i < values.size(); ++i)
      vector[i] = values[i];

    pde_operator->set_coefficient_from_dof_vector(vector, true /* check_positivity */);

    note_coefficient_changed();
  }

  /**
   * Installs one diffusivity per block, skipping the fill when it is already there.
   */
  void
  install_block_diffusivity(std::vector<double> const & diffusivity)
  {
    AssertThrow(diffusivity.size() == n_blocks(),
                dealii::ExcMessage("Expected " + std::to_string(n_blocks()) +
                                   " block diffusivities, got " +
                                   std::to_string(diffusivity.size()) + "."));

    if(diffusivity == current_diffusivity)
      return;

    pde_operator->set_coefficient(
      BlockCoefficient<dim>(application->get_blocks_per_dim(), diffusivity));

    current_diffusivity = diffusivity;
  }

  /**
   * Records that the coefficient installed in the operator has changed.
   *
   * Every cache below is consulted to *skip* work: the coefficient fill, the preconditioner
   * rebuild. Clearing them together, in one place, is what keeps a new coefficient from being
   * solved with a stale preconditioner -- which produces a perfectly ordinary-looking wrong
   * answer, and which separate invalidations at four call sites will eventually miss.
   */
  void
  note_coefficient_changed()
  {
    current_diffusivity.clear();
    current_parameters.clear();
    preconditioned_parameters.clear();
  }

  MPI_Comm mpi_comm;

  std::shared_ptr<Poisson::Application<dim, 1, Number>> application;
  std::unique_ptr<Poisson::Driver<dim, Number>>         driver;
  std::shared_ptr<Poisson::Operator<dim, 1, Number>>    pde_operator;

  MassOperator<dim, 1, Number>        mass_operator;
  InverseMassOperator<dim, 1, Number> inverse_mass_operator;

  SensorOperator<dim, Number> sensors;

  // diffusivities currently installed in the operator; empty until the first assignment
  std::vector<double> current_diffusivity;

  // parameters last installed through set_diffusivity(), and those the preconditioner was built
  // for; empty means "unknown", which forces the work rather than skipping it
  std::vector<double> current_parameters;
  std::vector<double> preconditioned_parameters;
};

/**
 * Registers the model for one space dimension.
 *
 * The pyMOR-facing surface is inherited from PyMOR::FullOrderModel, which exadg._core binds; only
 * the constructor and the thermal-block specifics are added here.
 */
template<int dim>
void
register_model(py::module_ & module, std::string const & name)
{
  py::class_<ThermalBlockFOM<dim>,
             PyMOR::FullOrderModel<VectorType>,
             std::shared_ptr<ThermalBlockFOM<dim>>>(module, name.c_str())
    .def(py::init<std::string const &, unsigned int, unsigned int, bool>(),
         py::arg("input_file"),
         py::arg("degree")      = 3,
         py::arg("refinements") = 4,
         py::arg("verbose")     = false)
    .def_property_readonly("n_blocks", &ThermalBlockFOM<dim>::n_blocks)
    .def_property_readonly("n_parameters", &ThermalBlockFOM<dim>::n_parameters)
    .def_property_readonly("coefficient_degree", &ThermalBlockFOM<dim>::coefficient_degree)
    .def_property_readonly("blocks_per_dim", &ThermalBlockFOM<dim>::blocks_per_dim)
    .def_property_readonly("n_sensors", &ThermalBlockFOM<dim>::n_sensors)
    .def_property_readonly("dim", [](ThermalBlockFOM<dim> const &) { return dim; })
    .def("coefficient_support_points", &ThermalBlockFOM<dim>::coefficient_support_points)
    .def("sensor_coordinates", &ThermalBlockFOM<dim>::sensor_coordinates)
    .def("set_sensor_points", &ThermalBlockFOM<dim>::set_sensor_points, py::arg("points"))
    .def("set_cell_diffusivity", &ThermalBlockFOM<dim>::set_cell_diffusivity, py::arg("values"));
}

} // namespace ExaDG

PYBIND11_MODULE(thermal_block, module)
{
  using namespace ExaDG;

  // Brings in the vector and operator base classes, and initialises MPI and deal.II's external
  // libraries. Without it the types this module returns would be unregistered and pybind11 would
  // report them as opaque.
  py::module_::import("exadg._core");

  module.doc() =
    "The ExaDG thermal block as a full-order model for pyMOR.\n\n"
    "A Poisson problem with a piecewise constant diffusivity on a Cartesian grid of blocks, "
    "exposed through exadg/pymor/interface.h. Wrap it with "
    "exadg.mor.models.stationary.stationary_model().";

  register_model<2>(module, "ThermalBlockFOM2D");
  register_model<3>(module, "ThermalBlockFOM3D");
}
