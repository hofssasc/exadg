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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_FORCED_APPLICATION_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_FORCED_APPLICATION_H_

namespace ExaDG
{
namespace IncNS
{
/**
 * A body force built from a few smooth modes, affine in its amplitudes.
 *
 *     f(x; a) = sum_i a_i exp(-|x - c_i|^2 / 2 w^2) e_x
 *
 * Smooth blobs rather than a piecewise constant field on purpose. A discontinuous coefficient
 * would be the natural transplant of the thermal block, but it is the wrong object here twice
 * over: a discontinuous *forcing* is not a physical load, and a discontinuous *viscosity* has no
 * exact affine decomposition in this discretisation at all, because the interior-face viscosity
 * is a harmonic mean of the two sides and the harmonic mean is not linear.
 *
 * The amplitudes are mutable so that a parameter sweep changes them without rebuilding the
 * operator; the field functions hold this object by shared pointer and re-evaluate it whenever
 * the right-hand side is assembled.
 */
template<int dim>
class ForcingModes : public dealii::Function<dim>
{
public:
  ForcingModes(unsigned int const modes_per_dim, double const width)
    : dealii::Function<dim>(dim, 0.0),
      centres(lattice(modes_per_dim)),
      width(width),
      amplitudes(centres.size(), 1.0)
  {
    AssertThrow(width > 0.0, dealii::ExcMessage("The forcing width must be positive."));
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    // The flow is driven towards the outflow, so only the streamwise component is forced.
    if(component != 0)
      return 0.0;

    double sum = 0.0;
    for(unsigned int i = 0; i < centres.size(); ++i)
      sum += amplitudes[i] * std::exp(-0.5 * centres[i].distance_square(p) / (width * width));

    return sum;
  }

  void
  set_amplitudes(std::vector<double> const & values)
  {
    AssertThrow(values.size() == amplitudes.size(),
                dealii::ExcMessage("Expected " + std::to_string(amplitudes.size()) +
                                   " amplitudes, got " + std::to_string(values.size()) + "."));
    amplitudes = values;
  }

  unsigned int
  n_modes() const
  {
    return static_cast<unsigned int>(centres.size());
  }

  std::vector<dealii::Point<dim>> const &
  get_centres() const
  {
    return centres;
  }

private:
  /** Mode centres on an interior Cartesian lattice of the unit hypercube. */
  static std::vector<dealii::Point<dim>>
  lattice(unsigned int const per_dim)
  {
    std::vector<dealii::Point<dim>> points(dealii::Utilities::pow(per_dim, dim));

    for(unsigned int i = 0; i < points.size(); ++i)
    {
      unsigned int remaining = i;
      for(unsigned int d = 0; d < dim; ++d)
      {
        points[i][d] = (remaining % per_dim + 0.5) / per_dim;
        remaining /= per_dim;
      }
    }

    return points;
  }

  std::vector<dealii::Point<dim>> const centres;
  double const                          width;
  std::vector<double>                   amplitudes;
};

/**
 * Steady incompressible flow in a box, driven by a parameterised body force.
 *
 * One application for both halves of the saddle-point reduced-order model. "Equation" selects
 * Stokes or Navier-Stokes and the convective term is the *only* difference between them, so a
 * reduced model for one is a controlled comparison for the other: everything that could go wrong
 * in the block plumbing is shared, and only the nonlinearity is not.
 *
 * Steady, so there is no time dimension; coupled rather than split, because only the coupled
 * formulation is a single operator that can be projected.
 *
 * The P parameters are the forcing amplitudes,
 *
 *     A = nu K (fixed),    rhs(a) = sum_i a_i f_i,
 *
 * so the right-hand side is exactly affine and the operator does not depend on the parameter at
 * all. The viscosity is *constant in space* -- it is the inverse Reynolds number and nothing
 * else -- and it is a setting rather than a parameter, because ExaDG bakes it into the viscous
 * kernel at setup together with the interior penalty parameter derived from it. A blockwise
 * viscosity would not even be affine here, since the interior-face viscosity is a harmonic mean
 * of the two sides.
 *
 * **Stokes is a verification problem, not a benchmark.** The solution is linear in the
 * amplitudes,
 *
 *     u(a) = sum_i a_i u_i,    p(a) = sum_i a_i p_i,
 *
 * so the solution manifold is exactly P-dimensional in each block and a reduced basis of P
 * velocity and P pressure modes must reproduce the full-order model to machine precision.
 * Anything else is a bug in the saddle-point projection rather than an approximation error.
 * That exact answer is what makes it worth running first; the benchmark is Navier-Stokes.
 *
 * With the convective term on that property is gone -- the solution is no longer linear in the
 * amplitudes -- and the Reynolds number has to be chosen so that the nonlinearity actually
 * matters while a steady solution still exists.
 *
 * Boundary conditions are homogeneous throughout -- no-slip on three sides and a do-nothing
 * outflow at x = 1 -- for two reasons. There is no Dirichlet lifting, so the right-hand side
 * stays affine in the amplitudes alone; and the outflow pins the pressure level, so the reduced
 * saddle-point system inherits no nullspace. A lid-driven cavity would fail on both counts.
 *
 * The velocity space is discontinuous (SpatialDiscretization::L2, ExaDG's default). ExaDG offers
 * no continuous-Galerkin option for incompressible flow -- the choice is L2 or HDIV -- so this is
 * a property of the solver rather than of this application.
 */
template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("Equation",
                        equation,
                        "Stokes or NavierStokes. The only difference is the convective term, so "
                        "the two are the same problem with and without the nonlinearity -- which "
                        "is what makes a reduced model for one a controlled comparison for the "
                        "other.",
                        dealii::Patterns::Selection("Stokes|NavierStokes"));
      prm.add_parameter("Viscosity",
                        viscosity,
                        "Kinematic viscosity, constant in space. With a unit box and unit "
                        "forcing this is the inverse Reynolds number.");
      prm.add_parameter("ForcingModesPerDim",
                        modes_per_dim,
                        "Body-force modes per coordinate direction; the number of amplitude "
                        "parameters is this to the power dim.",
                        dealii::Patterns::Integer(1, 16));
      prm.add_parameter("SolverTolerance",
                        solver_tolerance,
                        "Relative tolerance of the coupled linear solver. GMRES converges the "
                        "residual, so the error in the solution is this times the condition "
                        "number of the saddle point -- which is what sets the floor for any "
                        "check that a reduced model reproduces the full one exactly.");
      prm.add_parameter("ForcingWidth",
                        forcing_width,
                        "Standard deviation of each forcing mode.");
      prm.add_parameter("UpwindFactor",
                        upwind_factor,
                        "Weight of the Lax-Friedrichs stabilisation in the convective numerical "
                        "flux; 1.0 is ExaDG's default. Exposed because it is the one term of the "
                        "convective operator that is not a polynomial in the velocity -- lambda "
                        "is a maximum of absolute normal velocities -- and therefore the one term "
                        "a reduced model cannot represent as a third-order tensor.",
                        dealii::Patterns::Double(0.0, 1.0));
    }
    prm.leave_subsection();
  }

  /** The forcing object, so that a parameter sweep can change its amplitudes. */
  std::shared_ptr<ForcingModes<dim>>
  get_forcing() const
  {
    return forcing;
  }

  double
  get_viscosity() const
  {
    return viscosity;
  }

  double
  get_upwind_factor() const
  {
    return upwind_factor;
  }

  /** "Stokes" or "NavierStokes"; the convective term is the only difference. */
  std::string
  get_equation() const
  {
    return equation;
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type             = ProblemType::Steady;
    this->param.equation_type            = equation == "Stokes" ?
                                             EquationType::Stokes :
                                             EquationType::NavierStokes;

    // A steady solver has no time level to lag the convective term behind, so ExaDG requires it
    // implicitly. That is also what the reduced model needs: an implicit convective term makes
    // the (1,1) block depend on the current velocity, which is precisely the Jacobian a Newton
    // iteration -- full order or reduced -- has to be handed.
    this->param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
    this->param.formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
    this->param.right_hand_side          = true;
    this->param.upwind_factor            = upwind_factor;

    // PHYSICAL QUANTITIES
    this->param.start_time = 0.0;
    this->param.end_time   = 1.0;
    this->param.viscosity  = viscosity;

    // TEMPORAL DISCRETIZATION
    // Steady + coupled: the only combination that is a single saddle-point operator. The
    // splitting schemes are time-integration algorithms whose substeps do not compose into one
    // residual, so they cannot be projected.
    this->param.solver_type             = SolverType::Steady;
    this->param.temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1.0;
    this->param.order_time_integrator   = 1;

    this->param.convergence_criterion_steady_problem =
      ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
    this->param.abs_tol_steady = 1.e-12;
    this->param.rel_tol_steady = 1.e-10;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = this->param.degree_u;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;
    this->param.degree_p                    = DegreePressure::MixedOrder;
    this->param.IP_formulation_viscous      = InteriorPenaltyFormulation::SIPG;

    // The penalty terms would make the (1,1) block depend on the current velocity, i.e. on the
    // solution, which is exactly the parametric structure this first step exists to avoid. They
    // are a stabilisation for convection-dominated flow and Stokes has none.
    this->param.use_divergence_penalty = false;
    this->param.use_continuity_penalty = false;

    // COUPLED SOLVER
    this->param.solver_data_coupled =
      SolverData(1e4, 1.e-14, solver_tolerance, LinearSolver::GMRES, 100);
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // Not InverseMassMatrix for the velocity block: that preconditioner scales by the inverse
    // of scaling_factor_mass, which is zero for a steady problem.
    this->param.preconditioner_velocity_block = MomentumPreconditioner::Multigrid;
    // Stokes: the (1,1) block is the viscous operator alone, with no convective part.
    // The velocity-block preconditioner deliberately ignores convection, for both equations. It
    // is only a preconditioner, so leaving the convective term out costs iterations rather than
    // correctness, and it keeps the block symmetric -- which the Chebyshev smoother below
    // requires. Measured cost of that choice: the average GMRES count per Newton step grows from
    // 25 at nu = 1 to 520 at nu = 0.005, and no longer converges at nu = 0.002.
    //
    // Switching to ReactionConvectionDiffusion is *not* a drop-in improvement and was measured
    // to be worse: it makes the block non-symmetric, so Chebyshev's CG eigenvalue estimate
    // divides by zero, and replacing Chebyshev with a GMRES smoother then failed to converge at
    // nu = 1 where the original settings need 25 iterations. A convection-aware preconditioner
    // here is a piece of solver tuning in its own right, not a line to change in passing.
    this->param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;

    // Pointless while the preconditioner ignores convection: it does not depend on the
    // linearisation velocity that Newton updates.
    this->param.update_preconditioner_coupled = false;
    this->param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Chebyshev;
    this->param.multigrid_data_velocity_block.coarse_problem.solver =
      MultigridCoarseGridSolver::Chebyshev;

    // For Stokes the Schur complement is spectrally equivalent to the pressure mass matrix
    // scaled by the inverse viscosity, which is what this preconditioner applies.
    this->param.preconditioner_pressure_block =
      SchurComplementPreconditioner::InverseMassMatrix;

    this->param.solver_info_data.interval_time_steps = 1;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        // colorize() numbers the faces x- = 0, x+ = 1, y- = 2, y+ = 3, ...
        dealii::GridGenerator::hyper_cube(tria, 0.0, 1.0, true /* colorize */);
        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {});

    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // No-slip everywhere except the outflow face x = 1, which carries the do-nothing condition:
    // Neumann for the velocity, Dirichlet for the pressure. That single face is what makes the
    // pressure unique, and therefore what keeps the reduced saddle-point system nonsingular.
    for(dealii::types::boundary_id id = 0; id < 2 * dim; ++id)
    {
      if(id == 1)
      {
        this->boundary_descriptor->velocity->neumann_bc.insert(
          pair(id, new dealii::Functions::ZeroFunction<dim>(dim)));
        this->boundary_descriptor->pressure->dirichlet_bc.insert(
          pair(id, new dealii::Functions::ZeroFunction<dim>(1)));
      }
      else
      {
        this->boundary_descriptor->velocity->dirichlet_bc.insert(
          pair(id, new dealii::Functions::ZeroFunction<dim>(dim)));
        this->boundary_descriptor->pressure->neumann_bc.insert(id);
      }
    }
  }

  void
  set_field_functions() final
  {
    forcing = std::make_shared<ForcingModes<dim>>(modes_per_dim, forcing_width);

    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));

    this->field_functions->right_hand_side = forcing;
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory;
    pp_data.output_data.filename                    = this->output_parameters.filename;
    pp_data.output_data.write_divergence            = true;
    pp_data.output_data.degree                      = this->param.degree_u;

    // The correctness check for an incompressible solver: the discrete divergence of the
    // velocity, and the mass flux across element faces. Both must be at solver tolerance.
    pp_data.mass_data.time_control_data.is_active        = true;
    pp_data.mass_data.time_control_data.trigger_interval = 1.0;
    pp_data.mass_data.directory                          = this->output_parameters.directory;
    pp_data.mass_data.filename                           = this->output_parameters.filename;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // "Stokes" or "NavierStokes": the convective term is the only difference between them
  std::string  equation      = "Stokes";

  double       viscosity        = 1.0;
  double       solver_tolerance = 1.e-10;
  double       upwind_factor    = 1.0;
  unsigned int modes_per_dim = 2;
  double       forcing_width = 0.15;

  std::shared_ptr<ForcingModes<dim>> forcing;
};

} // namespace IncNS
} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_FORCED_APPLICATION_H_ */
