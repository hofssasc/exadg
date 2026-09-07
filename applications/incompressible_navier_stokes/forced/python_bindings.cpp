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
 * The forced box as a PyMOR::SaddlePointModel.
 *
 * The velocity/pressure counterpart of the thermal block's python_bindings.cpp: implement the
 * interface of exadg/pymor/interface.h, bind the model, import exadg._core, and the Python layer
 * builds the pyMOR model.
 *
 * ExaDG's coupled operator already *is* the block system pyMOR wants,
 *
 *     [ A   B* ] [u]   [f]
 *     [ B   0  ] [p] = [g]
 *
 * with A = momentum_operator, B = -s div and B* = +s grad, where s is
 * scaling_factor_continuity and the sign keeps the matrix symmetric. All this file does is hand
 * out those three blocks separately, because a projection-based reduced model projects each onto
 * its own basis rather than applying the assembled system.
 *
 * Parameters are the forcing amplitudes. A and B do not depend on them, which makes this the
 * smallest problem that still poses the saddle-point question -- and one whose answer is known:
 * the solution is linear in the amplitudes, so a basis of the P mode solutions has to reproduce
 * the full-order model exactly.
 */

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/driver.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/operators/mass_operator.h>
#include <exadg/pymor/interface.h>
#include <exadg/utilities/create_directories.h>

// application
#include "application.h"

namespace py = pybind11;

namespace ExaDG
{
namespace IncNS
{
using Number          = double;
using VectorType      = dealii::LinearAlgebra::distributed::Vector<Number>;
using BlockVectorType = dealii::LinearAlgebra::distributed::BlockVector<Number>;

namespace
{
/// Redirects std::cout while a model is being built, so a notebook does not get ExaDG's banner.
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
 * Steady incompressible flow in a box, driven by a body force expanded in modes.
 */
template<int dim>
class ForcedFOM : public PyMOR::SaddlePointModel<VectorType>
{
public:
  ForcedFOM(std::string const & input_file,
            unsigned int const  degree,
            unsigned int const  refinements,
            bool const          verbose = false)
    : mpi_comm(MPI_COMM_WORLD)
  {
    SuppressOutput const suppress(not verbose);

    application = std::make_shared<Application<dim, Number>>(input_file, mpi_comm);
    application->set_parameters_convergence_study(degree, refinements, 0);

    driver = std::make_unique<Driver<dim, Number>>(mpi_comm, application, true, false);
    driver->setup();

    pde_operator = std::dynamic_pointer_cast<OperatorCoupled<dim, Number>>(
      driver->get_pde_operator());

    AssertThrow(pde_operator.get() != nullptr,
                dealii::ExcMessage("This model needs the coupled solver; set "
                                   "TemporalDiscretization::BDFCoupledSolution."));

    // Steady: no mass term in the (1,1) block. ExaDG sets this inside solve_linear_problem(),
    // which a reduced-order model never calls, so it has to be set once here -- otherwise the
    // operator that gets projected is not the operator that gets solved.
    pde_operator->get_momentum_operator().set_scaling_factor_mass_operator(0.0);

    // The pressure inner product. ExaDG carries a velocity mass operator but no pressure one,
    // since nothing in a monolithic solve needs it; a POD of pressure snapshots does.
    MassOperatorData<dim, Number> pressure_mass_data;
    pressure_mass_data.dof_index  = pde_operator->get_dof_index_pressure();
    pressure_mass_data.quad_index = pde_operator->get_quad_index_pressure();

    pressure_mass.initialize(pde_operator->get_matrix_free(),
                             pde_operator->get_matrix_free().get_affine_constraints(
                               pde_operator->get_dof_index_pressure()),
                             pressure_mass_data);
  }

  // ===========================================================================================
  //  Spaces
  // ===========================================================================================

  /// The velocity space. make_admissible() is a no-op: a discontinuous Galerkin velocity space
  /// imposes its boundary conditions weakly and constrains no degree of freedom.
  class VelocitySpace : public PyMOR::Space<VectorType>
  {
  public:
    explicit VelocitySpace(std::shared_ptr<ForcedFOM<dim>> fom) : fom(fom)
    {
    }

    dealii::types::global_dof_index
    n_dofs() const override
    {
      return fom->pde_operator->get_dof_handler_u().n_dofs();
    }

    std::shared_ptr<VectorType>
    zero_vector() const override
    {
      auto vector = std::make_shared<VectorType>();
      fom->pde_operator->initialize_vector_velocity(*vector);
      *vector = 0.0;

      return vector;
    }

    std::string
    write_vtu(std::string const &                              directory,
              std::string const &                              basename,
              std::vector<std::shared_ptr<VectorType>> const & fields,
              std::vector<std::string> const &                 names) const override
    {
      return fom->write_fields(fom->pde_operator->get_dof_handler_u(),
                               directory,
                               basename,
                               fields,
                               names,
                               true /* vector valued */);
    }

  private:
    std::shared_ptr<ForcedFOM<dim>> fom;
  };

  class PressureSpace : public PyMOR::Space<VectorType>
  {
  public:
    explicit PressureSpace(std::shared_ptr<ForcedFOM<dim>> fom) : fom(fom)
    {
    }

    dealii::types::global_dof_index
    n_dofs() const override
    {
      return fom->pde_operator->get_dof_handler_p().n_dofs();
    }

    std::shared_ptr<VectorType>
    zero_vector() const override
    {
      auto vector = std::make_shared<VectorType>();
      fom->pde_operator->initialize_vector_pressure(*vector);
      *vector = 0.0;

      return vector;
    }

    std::string
    write_vtu(std::string const &                              directory,
              std::string const &                              basename,
              std::vector<std::shared_ptr<VectorType>> const & fields,
              std::vector<std::string> const &                 names) const override
    {
      return fom->write_fields(fom->pde_operator->get_dof_handler_p(),
                               directory,
                               basename,
                               fields,
                               names,
                               false /* scalar */);
    }

  private:
    std::shared_ptr<ForcedFOM<dim>> fom;
  };

  // ===========================================================================================
  //  The three blocks
  // ===========================================================================================

  /// A, the (1,1) block. Viscous only while the equation is Stokes and the problem is steady.
  class Momentum : public PyMOR::LinearOperator<VectorType>
  {
  public:
    explicit Momentum(std::shared_ptr<ForcedFOM<dim>> fom) : fom(fom)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->pde_operator->get_momentum_operator().vmult(dst, src);
    }

    /// Only while there is no convective term: the viscous operator is symmetric, u . grad u is
    /// not. The Navier-Stokes step has to override apply_transpose() rather than flip this.
    bool
    is_symmetric() const override
    {
      return not fom->application->get_parameters().convective_problem();
    }

    std::string
    get_name() const override
    {
      return "A";
    }

  private:
    std::shared_ptr<ForcedFOM<dim>> fom;
  };

  /**
   * B, the (2,1) block: velocity in, pressure out.
   *
   * ExaDG applies -s div here and +s grad in the (1,2) block, s being
   * scaling_factor_continuity, so that the assembled matrix is symmetric. pyMOR builds its (1,2)
   * block as AdjointOperator(B), which therefore has to reproduce +s grad exactly -- that is an
   * identity between two separately implemented operators, so the example checks it rather than
   * assuming it.
   */
  class Divergence : public PyMOR::LinearOperator<VectorType>
  {
  public:
    explicit Divergence(std::shared_ptr<ForcedFOM<dim>> fom) : fom(fom)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->pde_operator->get_divergence_operator().apply(dst, src);
      dst *= -fom->scaling_factor_continuity();
    }

    /// Different operator, not the same one: never claim symmetry here.
    bool
    is_symmetric() const override
    {
      return false;
    }

    void
    apply_transpose(VectorType & dst, VectorType const & src) const override
    {
      fom->pde_operator->get_gradient_operator().apply(dst, src);
      dst *= fom->scaling_factor_continuity();
    }

    std::string
    get_name() const override
    {
      return "B";
    }

  private:
    std::shared_ptr<ForcedFOM<dim>> fom;
  };

  /// The velocity mass matrix. Block diagonal for a discontinuous space, so the inverse is
  /// elementwise -- which is what makes supremizer enrichment affordable.
  class VelocityMass : public PyMOR::LinearOperator<VectorType>
  {
  public:
    explicit VelocityMass(std::shared_ptr<ForcedFOM<dim>> fom) : fom(fom)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->pde_operator->apply_mass_operator(dst, src);
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
      auto dst = std::make_shared<VectorType>();
      fom->pde_operator->initialize_vector_velocity(*dst);
      fom->pde_operator->apply_inverse_mass_operator(*dst, rhs);

      return dst;
    }

    std::string
    get_name() const override
    {
      return "velocity mass";
    }

  private:
    std::shared_ptr<ForcedFOM<dim>> fom;
  };

  class PressureMass : public PyMOR::LinearOperator<VectorType>
  {
  public:
    explicit PressureMass(std::shared_ptr<ForcedFOM<dim>> fom) : fom(fom)
    {
    }

    void
    apply(VectorType & dst, VectorType const & src) const override
    {
      fom->pressure_mass.vmult(dst, src);
    }

    bool
    is_symmetric() const override
    {
      return true;
    }

    std::string
    get_name() const override
    {
      return "pressure mass";
    }

  private:
    std::shared_ptr<ForcedFOM<dim>> fom;
  };

  // ===========================================================================================
  //  PyMOR::SaddlePointModel
  // ===========================================================================================

  std::shared_ptr<PyMOR::Space<VectorType>>
  velocity_space() override
  {
    return std::make_shared<VelocitySpace>(shared_self());
  }

  std::shared_ptr<PyMOR::Space<VectorType>>
  pressure_space() override
  {
    return std::make_shared<PressureSpace>(shared_self());
  }

  /// One group, holding the forcing amplitudes.
  std::vector<unsigned int>
  parameter_shape() const override
  {
    return {application->get_forcing()->n_modes()};
  }

  std::shared_ptr<PyMOR::LinearOperator<VectorType>>
  momentum() override
  {
    return std::make_shared<Momentum>(shared_self());
  }

  std::shared_ptr<PyMOR::LinearOperator<VectorType>>
  divergence() override
  {
    return std::make_shared<Divergence>(shared_self());
  }

  std::shared_ptr<PyMOR::LinearOperator<VectorType>>
  velocity_product() override
  {
    return std::make_shared<VelocityMass>(shared_self());
  }

  std::shared_ptr<PyMOR::LinearOperator<VectorType>>
  pressure_product() override
  {
    return std::make_shared<PressureMass>(shared_self());
  }

  /**
   * The velocity right-hand side at zero amplitudes.
   *
   * Not assumed to vanish. The boundary conditions here are homogeneous, so it should, but
   * rhs_linear_problem() also collects the boundary terms of the gradient and viscous operators
   * and this is the honest constant term of the affine decomposition either way. The example
   * checks that it is zero rather than this file asserting it.
   */
  std::shared_ptr<VectorType>
  velocity_rhs() override
  {
    return assemble_rhs(std::vector<double>(n_modes(), 0.0)).first;
  }

  /// f_i = rhs(e_i) - rhs(0), one per forcing mode.
  std::vector<PyMOR::AffineVector<VectorType>>
  velocity_rhs_components() override
  {
    auto const constant = velocity_rhs();

    std::vector<PyMOR::AffineVector<VectorType>> components;
    for(unsigned int i = 0; i < n_modes(); ++i)
    {
      std::vector<double> amplitudes(n_modes(), 0.0);
      amplitudes[i] = 1.0;

      auto mode = assemble_rhs(amplitudes).first;
      mode->add(-1.0, *constant);

      components.push_back({mode, 0 /* slot */, i});
    }

    return components;
  }

  std::shared_ptr<VectorType>
  pressure_rhs() override
  {
    return assemble_rhs(std::vector<double>(n_modes(), 0.0)).second;
  }

  /**
   * The full-order coupled solve for the given right-hand side.
   *
   * Mirrors DriverSteadyProblems::do_solve() rather than calling it, for one reason: that driver
   * keeps its solution between calls and would warm-start the next parameter from the previous
   * one. A snapshot has to be a function of its parameter alone, so the guess is zeroed here.
   */
  bool
  solve(VectorType const & f, VectorType const & g, VectorType & u, VectorType & p) override
  {
    BlockVectorType solution;
    pde_operator->initialize_block_vector_velocity_pressure(solution);
    solution = 0.0;

    if(application->get_parameters().nonlinear_problem_has_to_be_solved())
    {
      // ExaDG's nonlinear solve takes the body force alone; the pressure equation of a steady
      // incompressible problem has no right-hand side to give it.
      if(g.l2_norm() != 0.0)
        return false;

      pde_operator->solve_nonlinear_problem(
        solution, f, application->get_parameters().update_preconditioner_coupled, 0.0 /* time */);
    }
    else
    {
      BlockVectorType rhs;
      pde_operator->initialize_block_vector_velocity_pressure(rhs);
      rhs.block(0) = f;
      rhs.block(1) = g;

      VectorType transport_velocity;

      pde_operator->solve_linear_problem(solution,
                                         rhs,
                                         transport_velocity,
                                         application->get_parameters()
                                           .update_preconditioner_coupled,
                                         0.0 /* scaling_factor_mass: steady */);
    }

    pde_operator->adjust_pressure_level_if_undefined(solution.block(1), 0.0);

    u = solution.block(0);
    p = solution.block(1);

    return true;
  }

  /// Number of forcing modes, i.e. of parameters.
  unsigned int
  n_modes() const
  {
    return application->get_forcing()->n_modes();
  }

private:
  std::shared_ptr<ForcedFOM<dim>>
  shared_self()
  {
    return std::static_pointer_cast<ForcedFOM<dim>>(this->shared_from_this());
  }

  double
  scaling_factor_continuity() const
  {
    // ExaDG defaults it to one and only changes it for a pressure-scaled formulation, which this
    // application does not use. Asserting beats reading a member that may drift.
    return 1.0;
  }

  /// The coupled right-hand side at the given amplitudes, as (velocity, pressure).
  std::pair<std::shared_ptr<VectorType>, std::shared_ptr<VectorType>>
  assemble_rhs(std::vector<double> const & amplitudes)
  {
    application->get_forcing()->set_amplitudes(amplitudes);

    BlockVectorType rhs;
    pde_operator->initialize_block_vector_velocity_pressure(rhs);

    VectorType transport_velocity;
    pde_operator->rhs_linear_problem(rhs, transport_velocity, 0.0);

    auto velocity = std::make_shared<VectorType>(rhs.block(0));
    auto pressure = std::make_shared<VectorType>(rhs.block(1));

    return {velocity, pressure};
  }

  std::string
  write_fields(dealii::DoFHandler<dim> const &                  dof_handler,
               std::string const &                              directory,
               std::string const &                              basename,
               std::vector<std::shared_ptr<VectorType>> const & fields,
               std::vector<std::string> const &                 names,
               bool const                                       vector_valued) const
  {
    AssertThrow(fields.size() == names.size() and not fields.empty(),
                dealii::ExcMessage("Expected one name per field, and at least one field."));

    // deal.II concatenates directory and file name verbatim, so a missing separator writes the
    // record next to the directory instead of inside it.
    std::string const path =
      (directory.empty() or directory.back() == '/') ? directory : directory + "/";

    create_directories(path, mpi_comm);

    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<VectorType> ghosted(fields.size());
    for(unsigned int i = 0; i < fields.size(); ++i)
    {
      if(vector_valued)
        pde_operator->initialize_vector_velocity(ghosted[i]);
      else
        pde_operator->initialize_vector_pressure(ghosted[i]);

      ghosted[i] = *fields[i];
      ghosted[i].update_ghost_values();

      if(vector_valued)
        data_out.add_data_vector(
          ghosted[i],
          std::vector<std::string>(dim, names[i]),
          dealii::DataOut<dim>::type_dof_data,
          std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>(
            dim, dealii::DataComponentInterpretation::component_is_part_of_vector));
      else
        data_out.add_data_vector(ghosted[i], names[i]);
    }

    data_out.build_patches(*pde_operator->get_mapping(), dof_handler.get_fe().degree);

    return path + data_out.write_vtu_with_pvtu_record(path, basename, 0, mpi_comm);
  }

  MPI_Comm mpi_comm;

  std::shared_ptr<Application<dim, Number>>         application;
  std::unique_ptr<Driver<dim, Number>>              driver;
  std::shared_ptr<OperatorCoupled<dim, Number>>     pde_operator;

  MassOperator<dim, 1, Number> pressure_mass;
};

template<int dim>
void
register_model(py::module_ & module, std::string const & name)
{
  py::class_<ForcedFOM<dim>,
             PyMOR::SaddlePointModel<VectorType>,
             std::shared_ptr<ForcedFOM<dim>>>(module, name.c_str())
    .def(py::init<std::string const &, unsigned int, unsigned int, bool>(),
         py::arg("input_file"),
         py::arg("degree")      = 2,
         py::arg("refinements") = 4,
         py::arg("verbose")     = false)
    .def_property_readonly("n_modes", &ForcedFOM<dim>::n_modes);
}

} // namespace IncNS
} // namespace ExaDG

PYBIND11_MODULE(forced, module)
{
  using namespace ExaDG::IncNS;

  py::module_::import("exadg._core");

  module.doc() =
    "Steady incompressible flow in a box, driven by a parameterised body force.\n\n"
    "A velocity/pressure saddle point exposed through exadg/pymor/interface.h. Wrap it with "
    "exadg.mor.models.saddle_point.saddle_point_model().";

  register_model<2>(module, "ForcedFOM2D");
  register_model<3>(module, "ForcedFOM3D");
}
