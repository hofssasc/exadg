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

#ifndef EXADG_REDUCED_ORDER_SENSOR_OPERATOR_H_
#define EXADG_REDUCED_ORDER_SENSOR_OPERATOR_H_

// deal.II
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

// C/C++
#include <memory>
#include <vector>

namespace ExaDG
{
/**
 * Evaluates a finite element field at a fixed set of sensor points.
 *
 * This is the observation operator B of the inverse problem: it maps a discrete solution to the
 * vector of measurements the likelihood compares against data. The same operator is applied
 * column by column to the reduced basis to obtain B*V, the only object the reduced-order model
 * needs in order to predict observations without ever reconstructing a full field.
 *
 * Evaluation uses dealii::Utilities::MPI::RemotePointEvaluation, so the sensors may lie
 * anywhere in the domain irrespective of the mesh partitioning, and the resulting values are
 * available on every rank. Unlike PointwiseOutputGenerator this does not require deal.II to be
 * configured with HDF5, and it is not tied to a time loop.
 *
 * The transpose is available as well -- evaluate_transpose(), and evaluate_gradient_transpose()
 * for functionals of the derivative. That is what an adjoint needs, and it is the difference
 * between a parameter gradient costing one solve per *parameter* and one solve per *functional*.
 * Because both directions are built from the same set of points, changing the sensor placement
 * is a call to setup() and nothing else: the adjoint follows.
 */
template<int dim, typename Number>
class SensorOperator
{
public:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  SensorOperator() = default;

  /**
   * Prepares the evaluation for the given sensor points.
   *
   * @param triangulation Triangulation the field is defined on.
   * @param mapping Mapping used for the search of the surrounding cells.
   * @param points_in Sensor locations. All of them must lie inside the domain.
   * @param tolerance Tolerance of the point search in unit cell coordinates.
   */
  void
  setup(dealii::Triangulation<dim> const &     triangulation,
        dealii::Mapping<dim> const &           mapping,
        std::vector<dealii::Point<dim>> const & points_in,
        double const                            tolerance = 1.0e-6)
  {
    points = points_in;

    remote_evaluator = std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(
      typename dealii::Utilities::MPI::RemotePointEvaluation<dim>::AdditionalData(tolerance,
                                                                                  false,
                                                                                  0));

    remote_evaluator->reinit(points, triangulation, mapping);

    AssertThrow(remote_evaluator->all_points_found(),
                dealii::ExcMessage(
                  "Not all sensor points were found inside the domain. Check that the sensor "
                  "locations lie strictly inside the computational domain."));
  }

  /**
   * Evaluates @p solution in every sensor point.
   *
   * @return Vector of length n_points(), identical on all ranks.
   */
  std::vector<Number>
  evaluate(dealii::DoFHandler<dim> const & dof_handler, VectorType const & solution) const
  {
    AssertThrow(remote_evaluator.get() != nullptr,
                dealii::ExcMessage("SensorOperator::setup() has to be called first."));

    // the ghost values of the solution are needed for points that fall into cells owned by
    // another rank
    solution.update_ghost_values();

    std::vector<Number> const values =
      dealii::VectorTools::point_values<1>(*remote_evaluator, dof_handler, solution);

    solution.zero_out_ghost_values();

    return values;
  }

  /**
   * Applies the operator to a set of vectors, which is how the reduced observation operator
   * B*V is assembled from the reduced basis V.
   *
   * @return Row-major matrix of shape (n_points(), vectors.size()).
   */
  std::vector<Number>
  evaluate(dealii::DoFHandler<dim> const & dof_handler,
           std::vector<VectorType> const & vectors) const
  {
    std::vector<Number> matrix(n_points() * vectors.size());

    for(unsigned int j = 0; j < vectors.size(); ++j)
    {
      std::vector<Number> const column = evaluate(dof_handler, vectors[j]);

      for(unsigned int i = 0; i < n_points(); ++i)
        matrix[i * vectors.size() + j] = column[i];
    }

    return matrix;
  }

  /**
   * The transpose B^T: spreads one weight per sensor back onto the finite element space.
   *
   * This is the object an adjoint needs and the reason it can be built at all. The gradient of
   * a functional of the solution is
   *
   *     dJ/dm = -p^T (dA/dm) u,      A^T p = dJ/du,
   *
   * and for J = weights . B u the adjoint right-hand side is exactly B^T weights. Without the
   * transpose the only route to a derivative is the forward sensitivity equation, which gives
   * whole *columns* -- one solve per parameter rather than one per functional.
   *
   * Assembled as the exact transpose of evaluate() rather than as an analytically equivalent
   * operator, and the distinction is not pedantic: a point sitting on a cell boundary is found
   * in every cell that touches it and evaluate() *averages* the results, so the transpose has to
   * hand each of those cells its share. On a Cartesian mesh with sensors at cell vertices --
   * which is the common case, not a corner one -- every sensor is such a point.
   *
   * No constraints are applied, matching evaluate(), which reads the raw degrees of freedom.
   * So the result can be nonzero on constrained degrees of freedom whenever a sensor's element
   * touches the boundary, and a caller solving the adjoint system has to zero those rows -- the
   * adjoint of a system whose Dirichlet rows have been eliminated is the same system restricted
   * to the free rows. See ThermalBlockFOM::adjoint_rhs() for that step.
   *
   * @param dof_handler The space the field lives in, the same one evaluate() is called with.
   * @param weights One weight per sensor point.
   * @param dst Overwritten with B^T weights.
   */
  void
  evaluate_transpose(dealii::DoFHandler<dim> const & dof_handler,
                     std::vector<Number> const &     weights,
                     VectorType &                    dst) const
  {
    AssertThrow(remote_evaluator.get() != nullptr,
                dealii::ExcMessage("SensorOperator::setup() has to be called first."));

    AssertThrow(weights.size() == n_points(),
                dealii::ExcMessage("Expected one weight per sensor point, got " +
                                   std::to_string(weights.size()) + " for " +
                                   std::to_string(n_points()) + " points."));

    dst = 0.0;

    auto const integration_function = [&](auto const & values, auto const & cell_data) {
      dealii::FEPointEvaluation<1, dim, dim, Number> evaluator(remote_evaluator->get_mapping(),
                                                               dof_handler.get_fe(),
                                                               dealii::update_values);

      std::vector<Number>                          local_values;
      std::vector<dealii::types::global_dof_index> local_dof_indices;

      for(auto const cell : cell_data.cell_indices())
      {
        auto const cell_dofs =
          cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(dof_handler);

        evaluator.reinit(cell_dofs, cell_data.get_unit_points(cell));

        auto const shares = cell_data.get_data_view(cell, values);
        for(auto const q : evaluator.quadrature_point_indices())
          evaluator.submit_value(shares[q], q);

        // test_and_sum() and not integrate(): a point functional carries no JxW factor. Using
        // integrate() here would weight every sensor by the volume of the cell it landed in,
        // which is a smooth, plausible and completely different operator.
        local_values.resize(cell_dofs->get_fe().n_dofs_per_cell());
        evaluator.test_and_sum(local_values, dealii::EvaluationFlags::values);

        local_dof_indices.resize(cell_dofs->get_fe().n_dofs_per_cell());
        cell_dofs->get_dof_indices(local_dof_indices);

        dealii::AffineConstraints<Number>().distribute_local_to_global(local_values,
                                                                       local_dof_indices,
                                                                       dst);
      }
    };

    remote_evaluator->template process_and_evaluate<Number>(shared_weights(weights),
                                                            integration_function);

    dst.compress(dealii::VectorOperation::add);
  }

  /**
   * The transpose of point *gradient* evaluation, for functionals of the derivative.
   *
   * The counterpart of evaluate_transpose() for J = sum_i w_i . grad u(x_i), which is deal.II
   * step-14's second dual functional -- the x-derivative in a point -- and everything of that
   * family. Given that the whole adjoint machinery above it takes a right-hand side vector and
   * nothing else, this is what makes "any point functional", rather than "sensors", the actual
   * scope.
   *
   * @param weights dim entries per sensor point, point index running slowest.
   * @param dst Overwritten with sum_i w_i . grad phi_j(x_i).
   */
  void
  evaluate_gradient_transpose(dealii::DoFHandler<dim> const & dof_handler,
                              std::vector<Number> const &     weights,
                              VectorType &                    dst) const
  {
    AssertThrow(remote_evaluator.get() != nullptr,
                dealii::ExcMessage("SensorOperator::setup() has to be called first."));

    AssertThrow(weights.size() == n_points() * dim,
                dealii::ExcMessage("Expected dim weights per sensor point, got " +
                                   std::to_string(weights.size()) + " for " +
                                   std::to_string(n_points()) + " points in " +
                                   std::to_string(dim) + "D."));

    using GradientType = dealii::Tensor<1, dim, Number>;

    dst = 0.0;

    std::vector<Number> const flat = shared_weights(weights, dim);

    std::vector<GradientType> input(n_points());
    for(unsigned int i = 0; i < n_points(); ++i)
      for(unsigned int d = 0; d < dim; ++d)
        input[i][d] = flat[i * dim + d];

    auto const integration_function = [&](auto const & values, auto const & cell_data) {
      dealii::FEPointEvaluation<1, dim, dim, Number> evaluator(remote_evaluator->get_mapping(),
                                                               dof_handler.get_fe(),
                                                               dealii::update_gradients);

      std::vector<Number>                          local_values;
      std::vector<dealii::types::global_dof_index> local_dof_indices;

      for(auto const cell : cell_data.cell_indices())
      {
        auto const cell_dofs =
          cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(dof_handler);

        evaluator.reinit(cell_dofs, cell_data.get_unit_points(cell));

        auto const shares = cell_data.get_data_view(cell, values);
        for(auto const q : evaluator.quadrature_point_indices())
          evaluator.submit_gradient(shares[q], q);

        local_values.resize(cell_dofs->get_fe().n_dofs_per_cell());
        evaluator.test_and_sum(local_values, dealii::EvaluationFlags::gradients);

        local_dof_indices.resize(cell_dofs->get_fe().n_dofs_per_cell());
        cell_dofs->get_dof_indices(local_dof_indices);

        dealii::AffineConstraints<Number>().distribute_local_to_global(local_values,
                                                                       local_dof_indices,
                                                                       dst);
      }
    };

    remote_evaluator->template process_and_evaluate<GradientType>(input, integration_function);

    dst.compress(dealii::VectorOperation::add);
  }

  unsigned int
  n_points() const
  {
    return points.size();
  }

  std::vector<dealii::Point<dim>> const &
  get_points() const
  {
    return points;
  }

  /**
   * Equispaced interior points of the unit hypercube, x_j = j/(n+1) per direction with
   * j = 1 ... n, in lexicographic order with the x-direction running fastest.
   *
   * The boundary is excluded deliberately: with homogeneous Dirichlet conditions the solution
   * vanishes there and a sensor would carry no information. This is the same layout used by
   * the Tier 0 benchmark, where it makes the aliasing structure exactly analysable.
   */
  static std::vector<dealii::Point<dim>>
  interior_cartesian_grid(unsigned int const points_per_dim)
  {
    AssertThrow(points_per_dim > 0, dealii::ExcMessage("points_per_dim must be positive."));

    unsigned int total = 1;
    for(unsigned int d = 0; d < dim; ++d)
      total *= points_per_dim;

    std::vector<dealii::Point<dim>> grid(total);

    for(unsigned int index = 0; index < total; ++index)
    {
      unsigned int remainder = index;

      for(unsigned int d = 0; d < dim; ++d)
      {
        unsigned int const i = remainder % points_per_dim;
        remainder /= points_per_dim;

        grid[index][d] = static_cast<double>(i + 1) / static_cast<double>(points_per_dim + 1);
      }
    }

    return grid;
  }

private:
  /**
   * Splits each weight over the cells its point was found in, which is what makes the transpose
   * exact.
   *
   * dealii::VectorTools::point_values() reduces duplicates with EvaluationFlags::avg, so the
   * forward operator's row for a point found in m cells is the *mean* of m cell-local rows.
   * Its transpose therefore sends w/m to each of them. RemotePointEvaluation::process_and_
   * evaluate() copies one input entry to every duplicate, so the division has to happen here.
   *
   * @param stride Number of entries per point; dim for a gradient functional.
   */
  std::vector<Number>
  shared_weights(std::vector<Number> const & weights, unsigned int const stride = 1) const
  {
    auto const & point_ptrs = remote_evaluator->get_point_ptrs();

    std::vector<Number> shares(weights.size(), 0.0);

    for(unsigned int i = 0; i < n_points(); ++i)
    {
      unsigned int const multiplicity = point_ptrs[i + 1] - point_ptrs[i];

      // A point found nowhere contributes nothing. setup() rejects that case, so reaching this
      // branch means the triangulation moved underneath a set-up operator.
      if(multiplicity == 0)
        continue;

      for(unsigned int c = 0; c < stride; ++c)
        shares[i * stride + c] = weights[i * stride + c] / static_cast<Number>(multiplicity);
    }

    return shares;
  }

  std::vector<dealii::Point<dim>> points;

  std::shared_ptr<dealii::Utilities::MPI::RemotePointEvaluation<dim>> remote_evaluator;
};

} // namespace ExaDG

#endif /* EXADG_REDUCED_ORDER_SENSOR_OPERATOR_H_ */
