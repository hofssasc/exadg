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

// deal.II
#include <deal.II/base/exceptions.h>

// ExaDG
#include <exadg/poisson/user_interface/parameters.h>

namespace ExaDG
{
namespace Poisson
{
Parameters::Parameters()
  : // MATHEMATICAL MODEL
    right_hand_side(false),

    // SPATIAL DISCRETIZATION
    grid(GridData()),
    mapping_degree(1),
    mapping_degree_coarse_grids(1),
    spatial_discretization(SpatialDiscretization::Undefined),
    degree(1),
    IP_factor(1.0),
    coefficient_is_variable(false),
    coefficient_degree(0),
    coefficient(1.0),
    use_matrix_based_operator(false),
    sparse_matrix_type(SparseMatrixType::Undefined),

    // SOLVER
    solver_data(SolverData(1e4, 1.e-20, 1.e-12, LinearSolver::CG)),
    compute_performance_metrics(false),
    preconditioner(Preconditioner::Undefined),
    multigrid_data(MultigridData()),
    enable_cell_based_face_loops(false)
{
}

void
Parameters::check() const
{
  // MATHEMATICAL MODEL

  // SPATIAL DISCRETIZATION
  grid.check();

  AssertThrow(spatial_discretization != SpatialDiscretization::Undefined,
              dealii::ExcMessage("parameter must be defined."));

  AssertThrow(degree > 0, dealii::ExcMessage("Polynomial degree must be larger than zero."));

  if(use_matrix_based_operator)
  {
    AssertThrow(sparse_matrix_type != SparseMatrixType::Undefined,
                dealii::ExcMessage("Parameter must be defined."));
  }

  if(coefficient_is_variable)
  {
    AssertThrow(spatial_discretization == SpatialDiscretization::CG,
                dealii::ExcMessage(
                  "A variable coefficient is currently only supported for continuous elements "
                  "(SpatialDiscretization::CG)."));

    // The multigrid level operators are built from the same LaplaceOperatorData, so each level
    // allocates its own coefficient table but nothing transfers the fine-level coefficient to
    // the coarser levels. The solution would still be correct -- a preconditioner built for the
    // plain Laplacian is a valid preconditioner -- but the iteration count degrades badly for
    // high coefficient contrast, and silently so. Reject it until coarse-level coefficient
    // transfer is implemented.
    AssertThrow(preconditioner != Preconditioner::Multigrid,
                dealii::ExcMessage(
                  "Geometric multigrid is not yet coefficient-aware: the coarse-level operators "
                  "would be built for the plain Laplacian. Use Preconditioner::AMG or "
                  "Preconditioner::PointJacobi with a variable coefficient."));

    // The coefficient machinery is written for an arbitrary degree; only this check stands in
    // the way, and it stands there for two independent reasons.
    //
    // POSITIVITY. The coefficient is expanded as a = sum_i c_i phi_i with c_i > 0, so a is
    // positive wherever the basis is non-negative. Lagrange bases are non-negative only up to
    // degree 1: on the reference cell the minimum basis value is -0.125 at degree 2 and -0.316
    // at degree 3. A coefficient that dips negative between nodes destroys coercivity, and it
    // does so invisibly, because every *coefficient* c_i is still positive -- it is the field
    // that is not. A basis that is non-negative at every degree (Bernstein, for instance) would
    // lift this, at the price of the coefficient degrees of freedom no longer being point
    // values of the field.
    //
    // QUADRATURE. The integrand a grad(u).grad(v) has per-variable degree 2*degree +
    // coefficient_degree, while the rule built in Operator::fill_matrix_free_data has degree +
    // 1 points and is exact to 2*degree + 1. So it is exact exactly while coefficient_degree
    // <= 1, and under-integrates by a few percent from degree 2 -- measured at 2-4% for
    // coefficient_degree 2 and 4-7% for 3, independently of the solution degree. Lifting this
    // needs n = degree + 1 + ceil(coefficient_degree / 2) points, i.e. a second quadrature
    // index for the operator.
    AssertThrow(coefficient_degree <= 1,
                dealii::ExcMessage(
                  "A variable coefficient of degree " + std::to_string(coefficient_degree) +
                  " is not admissible. Lagrange bases take negative values from degree 2, so "
                  "the coefficient could become negative between nodes and the operator would "
                  "lose coercivity; and the quadrature rule in use is exact only up to "
                  "coefficient degree 1. See the comment at this assertion for what lifting "
                  "either restriction would take."));
  }
  else
  {
    AssertThrow(coefficient > 0.0,
                dealii::ExcMessage("The coefficient must be positive for the operator to be "
                                   "coercive."));
  }

  // SOLVER
  AssertThrow(solver_data.linear_solver != LinearSolver::Undefined,
              dealii::ExcMessage("Parameter must be defined."));
  AssertThrow(preconditioner != Preconditioner::Undefined,
              dealii::ExcMessage("parameter must be defined."));
}

bool
Parameters::involves_h_multigrid() const
{
  if(preconditioner == Preconditioner::Multigrid and multigrid_data.involves_h_transfer())
    return true;
  else
    return false;
}

void
Parameters::print(dealii::ConditionalOStream const & pcout, std::string const & name) const
{
  pcout << std::endl << name << std::endl;

  // MATHEMATICAL MODEL
  print_parameters_mathematical_model(pcout);

  // SPATIAL DISCRETIZATION
  print_parameters_spatial_discretization(pcout);

  // SOLVER
  print_parameters_solver(pcout);

  // NUMERICAL PARAMETERS
  print_parameters_numerical_parameters(pcout);
}

void
Parameters::print_parameters_mathematical_model(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Mathematical model:" << std::endl;

  print_parameter(pcout, "Right-hand side", right_hand_side);
}

void
Parameters::print_parameters_spatial_discretization(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Spatial Discretization:" << std::endl;

  grid.print(pcout);

  print_parameter(pcout, "Mapping degree", mapping_degree);

  if(involves_h_multigrid())
    print_parameter(pcout, "Mapping degree coarse grids", mapping_degree_coarse_grids);

  print_parameter(pcout, "FE space", spatial_discretization);

  print_parameter(pcout, "Polynomial degree", degree);

  if(spatial_discretization == SpatialDiscretization::DG)
    print_parameter(pcout, "IP factor", IP_factor);

  // Only report the coefficient if it deviates from the plain Laplacian, so that the output of
  // existing applications is unchanged.
  if(coefficient_is_variable)
  {
    print_parameter(pcout, "Variable coefficient", coefficient_is_variable);
    print_parameter(pcout, "Coefficient degree", coefficient_degree);
  }
  else if(coefficient != 1.0)
    print_parameter(pcout, "Coefficient", coefficient);

  print_parameter(pcout, "Use matrix-based operator", use_matrix_based_operator);

  if(use_matrix_based_operator)
  {
    print_parameter(pcout, "Sparse matrix type", sparse_matrix_type);
  }
}

void
Parameters::print_parameters_solver(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Solver:" << std::endl;

  solver_data.print(pcout);

  print_parameter(pcout, "Preconditioner", preconditioner);

  if(preconditioner == Preconditioner::Multigrid)
    multigrid_data.print(pcout);
}


void
Parameters::print_parameters_numerical_parameters(dealii::ConditionalOStream const & pcout) const
{
  pcout << std::endl << "Numerical parameters:" << std::endl;

  print_parameter(pcout, "Enable cell-based face loops", enable_cell_based_face_loops);
}


} // namespace Poisson
} // namespace ExaDG
