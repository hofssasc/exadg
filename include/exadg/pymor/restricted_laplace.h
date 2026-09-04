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

#ifndef EXADG_PYMOR_RESTRICTED_LAPLACE_H_
#define EXADG_PYMOR_RESTRICTED_LAPLACE_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>

// C/C++
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

// ExaDG
#include <exadg/pymor/block_coefficient.h>

namespace ExaDG
{
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
template<int dim, typename Number = double>
class RestrictedLaplace
{
public:
  /**
   * @param dof_handler The solution space.
   * @param mapping Used for the cell quadrature.
   * @param constraints The same object the operator applies, so that constrained rows and
   *        columns are reproduced rather than approximated.
   * @param blocks_per_dim Blocks per coordinate direction of the coefficient.
   * @param output_dofs The rows to reproduce; must be distinct.
   */
  RestrictedLaplace(dealii::DoFHandler<dim> const &                      dof_handler,
                    dealii::Mapping<dim> const &                         mapping,
                    dealii::AffineConstraints<Number> const &            affine_constraints,
                    unsigned int const                                   blocks_per_dim,
                    std::vector<dealii::types::global_dof_index> const & output_dofs)
    : blocks_per_dim(blocks_per_dim), output_dofs(output_dofs), constraints(&affine_constraints)
  {
    auto const & fe = dof_handler.get_fe();

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

    dealii::FEValues<dim> fe_values(mapping,
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

} // namespace ExaDG

#endif /* EXADG_PYMOR_RESTRICTED_LAPLACE_H_ */
