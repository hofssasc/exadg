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

#ifndef APPLICATIONS_POISSON_THERMAL_BLOCK_APPLICATION_H_
#define APPLICATIONS_POISSON_THERMAL_BLOCK_APPLICATION_H_

// ExaDG
#include <exadg/pymor/block_coefficient.h>

// application
#include "postprocessor.h"

namespace ExaDG
{
namespace Poisson
{
/**
 * Thermal block benchmark: -div(a(x) grad(u)) = f on the unit hypercube with homogeneous
 * Dirichlet conditions, where the diffusivity a is constant on each of a Cartesian grid of
 * blocks.
 *
 * This is Tier 1 of the Bayesian calibration benchmark suite. The parameters inferred are the
 * per-block log-diffusivities mu_p, so that a = exp(mu_p) on block p. Two properties make it
 * the right first case for a reduced-order model:
 *
 *  - The operator is *exactly* affine in exp(mu), A(mu) = sum_p exp(mu_p) A_p, so the
 *    offline/online split is exact and the derivatives dA/dmu_p = exp(mu_p) A_p and
 *    d2A/dmu_p2 = exp(mu_p) A_p are analytic. The reduced model's error is then purely due to
 *    basis truncation, which is the quantity the benchmark studies.
 *  - The map from mu to the observed solution is nonlinear, so the Laplace approximation is a
 *    genuine approximation rather than an identity, unlike in the Tier 0 heat problem.
 *
 * Homogeneous Dirichlet conditions and a parameter-independent source keep the right-hand side
 * independent of mu, so f^r = V^T f is computed once. Inhomogeneous Dirichlet data would make
 * the lifting term coefficient-dependent; still affine, but an avoidable complication.
 */
template<int dim, int n_components, typename Number>
class Application : public ApplicationBase<dim, n_components, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, n_components, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, n_components, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("BlocksPerDim",
                        blocks_per_dim,
                        "Number of coefficient blocks per coordinate direction. The total "
                        "number of blocks, and hence of parameters, is BlocksPerDim^dim. Zero "
                        "requests one block per mesh cell, which is the piecewise constant "
                        "(P0) coefficient field of the high-dimensional benchmark; the "
                        "parameter count is then the number of cells.",
                        dealii::Patterns::Integer(0, 100000));
      prm.add_parameter("CoefficientDegree",
                        coefficient_degree,
                        "Polynomial degree of the diffusivity field. Zero gives one value per "
                        "block (or per cell, if BlocksPerDim is zero); one gives a continuous "
                        "multilinear field whose parameters are its nodal values. Higher "
                        "degrees are implemented but rejected -- see Parameters::check().",
                        dealii::Patterns::Integer(0, 10));
      prm.add_parameter("LogDiffusivity",
                        log_diffusivity,
                        "Log-diffusivity mu_p of each block in lexicographic order, x fastest. "
                        "The diffusivity is a = exp(mu_p). If a single value is given it is "
                        "used for every block.");
      prm.add_parameter("SourceAmplitude",
                        source_amplitude,
                        "Amplitude of the constant right-hand side f. It carries no calibration "
                        "parameter, so the reduced right-hand side is assembled once.");
      prm.add_parameter("Preconditioner",
                        preconditioner,
                        "Preconditioner for the linear system. Geometric multigrid is not "
                        "coefficient-aware yet, so AMG is the default here.");
      prm.add_parameter("WriteSensors",
                        output_data.write_sensors,
                        "Write the values at the sensor points, i.e. the data of the inverse "
                        "problem.");
      prm.add_parameter("SensorsPerDim",
                        output_data.sensors_per_dim,
                        "Number of sensors per coordinate direction; the total number of "
                        "observations is SensorsPerDim^dim.",
                        dealii::Patterns::Integer(1, 1000));
    }
    prm.leave_subsection();
  }

  /**
   * Total number of blocks, i.e. of calibration parameters.
   */
  unsigned int
  n_blocks() const
  {
    return BlockCoefficient<dim>::n_blocks(get_blocks_per_dim());
  }

  /**
   * Number of cells per coordinate direction of the uniformly refined Cartesian mesh.
   *
   * The grid is a hypercube subdivided n_subdivisions_1d_hypercube times and then refined
   * globally, so this is exact rather than an estimate, and it is available before the
   * triangulation exists.
   */
  unsigned int
  get_cells_per_dim() const
  {
    return this->n_subdivisions_1d_hypercube * (1u << this->param.grid.n_refine_global);
  }

  /**
   * Degree of the diffusivity field; zero is the piecewise constant case.
   */
  unsigned int
  get_coefficient_degree() const
  {
    return coefficient_degree;
  }

  /**
   * Blocks per coordinate direction, with the per-cell request resolved.
   *
   * Only meaningful for a piecewise constant coefficient. Blocks must align with cell
   * boundaries: if they did not, a single cell would carry more than one coefficient value, the
   * coefficient would no longer be the piecewise constant function on the mesh that the
   * parameterization claims, and the quadrature would sample the discontinuity at whichever
   * points happen to fall on either side.
   */
  unsigned int
  get_blocks_per_dim() const
  {
    unsigned int const cells_per_dim = get_cells_per_dim();

    // one block per cell: the P0 coefficient field, with as many parameters as cells
    if(blocks_per_dim == 0)
      return cells_per_dim;

    AssertThrow(cells_per_dim % blocks_per_dim == 0,
                dealii::ExcMessage(
                  "BlocksPerDim = " + std::to_string(blocks_per_dim) +
                  " does not divide the " + std::to_string(cells_per_dim) +
                  " cells per direction, so block boundaries would cut through cells."));

    return blocks_per_dim;
  }

  /**
   * Number of sensors per coordinate direction, needed by the offline stage to build the same
   * observation operator the solver used.
   */
  unsigned int
  get_sensors_per_dim() const
  {
    return output_data.sensors_per_dim;
  }

  /**
   * Diffusivity exp(mu_p) of every block, in the order expected by BlockCoefficient.
   */
  std::vector<double>
  get_diffusivities() const
  {
    std::vector<double> const log_values = expanded_log_diffusivity();

    std::vector<double> diffusivities(log_values.size());
    for(unsigned int p = 0; p < log_values.size(); ++p)
      diffusivities[p] = std::exp(log_values[p]);

    return diffusivities;
  }

private:
  /**
   * Expands a single supplied value to all blocks, and checks the length otherwise.
   */
  std::vector<double>
  expanded_log_diffusivity() const
  {
    if(log_diffusivity.size() == 1)
      return std::vector<double>(n_blocks(), log_diffusivity[0]);

    AssertThrow(log_diffusivity.size() == n_blocks(),
                dealii::ExcMessage(
                  "LogDiffusivity must hold either a single value or BlocksPerDim^dim = " +
                  std::to_string(n_blocks()) + " values, but " +
                  std::to_string(log_diffusivity.size()) + " were given."));

    return log_diffusivity;
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;

    // the variable coefficient is currently only implemented for continuous elements
    this->param.spatial_discretization = SpatialDiscretization::CG;

    this->param.coefficient_is_variable = true;
    this->param.coefficient_degree      = coefficient_degree;

    // SOLVER
    this->param.solver_data                 = SolverData(1e4, 1e-20, 1e-12, LinearSolver::CG);
    this->param.compute_performance_metrics = false;
    this->param.preconditioner              = preconditioner;
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

        // the unit hypercube, matching the domain BlockCoefficient subdivides
        dealii::GridGenerator::subdivided_hyper_cube(
          tria, this->n_subdivisions_1d_hypercube, 0.0, 1.0, true /* colorize */);

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

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
    // homogeneous Dirichlet on the whole boundary keeps the right-hand side independent of
    // the parameters, so the reduced right-hand side is assembled once
    for(dealii::types::boundary_id id = 0; id < 2 * dim; ++id)
      this->boundary_descriptor->dirichlet_bc.insert(
        {id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(1)});
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);

    this->field_functions->right_hand_side =
      std::make_shared<dealii::Functions::ConstantFunction<dim>>(source_amplitude, 1);

    // Only meaningful for a piecewise constant field. At higher degree the coefficient is
    // prescribed through its own degrees of freedom instead -- there is no function to
    // evaluate, because the field is defined by an expansion rather than by a rule.
    if(coefficient_degree == 0)
      this->field_functions->coefficient =
        std::make_shared<BlockCoefficient<dim>>(get_blocks_per_dim(), get_diffusivities());
    else
      this->field_functions->coefficient =
        std::make_shared<dealii::Functions::ConstantFunction<dim>>(1.0, 1);
  }

  std::shared_ptr<PostProcessorBase<dim, n_components, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                    = this->output_parameters.filename;
    pp_data.output_data.write_higher_order          = false;
    pp_data.output_data.degree                      = this->param.degree;

    ThermalBlockOutputData data = output_data;
    data.directory              = this->output_parameters.directory;
    data.filename               = this->output_parameters.filename;

    std::shared_ptr<PostProcessorBase<dim, n_components, Number>> pp;
    pp.reset(new ThermalBlockPostProcessor<dim, n_components, Number>(pp_data,
                                                                     this->mpi_comm,
                                                                     data));

    return pp;
  }

  // zero requests one block per cell; see get_blocks_per_dim()
  unsigned int blocks_per_dim = 4;

  // zero keeps the piecewise constant field; see Parameters::coefficient_degree
  unsigned int coefficient_degree = 0;

  // a single entry is expanded to all blocks, which keeps the default input file independent
  // of BlocksPerDim and of the space dimension
  std::vector<double> log_diffusivity = {0.0};

  double source_amplitude = 1.0;

  /**
   * Upstream of centre in x, centred in the other directions.
   */
  Preconditioner preconditioner = Preconditioner::AMG;

  ThermalBlockOutputData output_data;
};

} // namespace Poisson
} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_POISSON_THERMAL_BLOCK_APPLICATION_H_ */
