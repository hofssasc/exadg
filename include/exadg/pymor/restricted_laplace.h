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
#include <deal.II/base/mpi.h>
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
#include <exadg/operators/block_coefficient.h>

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
 *
 * **In parallel the stencil is replicated, not distributed.** Each rank assembles the cells it
 * owns and then every rank receives all of them, so afterwards each holds the complete operator
 * and apply() needs no communication at all. That is not a shortcut: pyMOR builds the restricted
 * operator collectively but keeps only rank 0's object and calls it there alone, so an operator
 * that still needed its peers would deadlock on the first evaluation. It is also affordable
 * precisely because hyper-reduction is the point -- the stencil is a handful of cells whatever
 * the mesh, which is the same reason a hyper-reduced online phase need not be distributed.
 */
template<int dim, typename Number = double>
class RestrictedLaplace
{
public:
  /**
   * @param dof_handler The solution space.
   * @param mapping Used for the cell quadrature.
   * @param affine_constraints The same object the operator applies, so that constrained rows and
   *        columns are reproduced rather than approximated.
   * @param blocks_per_dim Blocks per coordinate direction of the coefficient.
   * @param output_dofs The rows to reproduce; must be distinct.
   * @param mpi_comm The communicator the stencil is gathered over.
   */
  RestrictedLaplace(dealii::DoFHandler<dim> const &                      dof_handler,
                    dealii::Mapping<dim> const &                         mapping,
                    dealii::AffineConstraints<Number> const &            affine_constraints,
                    unsigned int const                                   blocks_per_dim,
                    std::vector<dealii::types::global_dof_index> const & output_dofs,
                    MPI_Comm const &                                     mpi_comm)
    : blocks_per_dim(blocks_per_dim), output_dofs(output_dofs)
  {
    auto const & fe = dof_handler.get_fe();

    std::set<dealii::types::global_dof_index> requested(output_dofs.begin(), output_dofs.end());
    AssertThrow(requested.size() == output_dofs.size(),
                dealii::ExcMessage("The requested degrees of freedom must be distinct."));

    dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> local(dofs_per_cell);

    dealii::QGauss<dim> const quadrature(fe.degree + 1);

    dealii::FEValues<dim> fe_values(mapping,
                                    fe,
                                    quadrature,
                                    dealii::update_gradients | dealii::update_JxW_values |
                                      dealii::update_quadrature_points);

    unsigned int const n_blocks = dealii::Utilities::pow(blocks_per_dim, dim);

    // Assemble the cells this rank owns. A row of the operator receives contributions only from
    // cells containing that degree of freedom, so this set is exactly what is needed. The whole
    // geometric part -- quadrature, shape gradients, Jacobians -- is evaluated once here and
    // never again: the operator is affine in the block diffusivities cell by cell, so applying
    // it afterwards is dense arithmetic on matrices of size (dofs per cell) squared. That is also
    // precisely the structure ECSW assumes, a per-element contribution scaled by a weight.
    std::vector<Cell> mine;

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

      if(not touches)
        continue;

      Cell entry;
      entry.dofs = local;
      entry.constrained.resize(dofs_per_cell);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        entry.constrained[i] = affine_constraints.is_constrained(local[i]) ? 1 : 0;

      fe_values.reinit(cell);

      std::map<unsigned int, std::vector<double>> per_block;
      for(unsigned int q = 0; q < quadrature.size(); ++q)
      {
        unsigned int const block =
          BlockCoefficient<dim>::block_index_of(blocks_per_dim, fe_values.quadrature_point(q));

        AssertThrow(block < n_blocks, dealii::ExcMessage("Block index out of range."));

        auto & matrix =
          per_block.try_emplace(block, dofs_per_cell * dofs_per_cell, 0.0).first->second;

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            matrix[i * dofs_per_cell + j] +=
              (fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)) * fe_values.JxW(q);
      }

      for(auto const & item : per_block)
        entry.contributions.push_back({item.first, item.second});

      mine.push_back(std::move(entry));
    }

    // Replicate: after this every rank holds the whole stencil and can evaluate it alone.
    for(auto const & from_rank : dealii::Utilities::MPI::all_gather(mpi_comm, mine))
      cells.insert(cells.end(), from_rank.begin(), from_rank.end());

    std::set<dealii::types::global_dof_index> source_set;
    for(auto const & cell : cells)
      source_set.insert(cell.dofs.begin(), cell.dofs.end());

    source_dofs.assign(source_set.begin(), source_set.end());

    std::map<dealii::types::global_dof_index, unsigned int> source_position;
    for(unsigned int i = 0; i < source_dofs.size(); ++i)
      source_position[source_dofs[i]] = i;

    std::map<dealii::types::global_dof_index, unsigned int> output_position;
    for(unsigned int i = 0; i < output_dofs.size(); ++i)
      output_position[output_dofs[i]] = i;

    // Whether a row is constrained is read back from the gathered cells rather than queried on
    // the constraints object: every output degree of freedom belongs to a stencil cell on some
    // rank, but not necessarily on this one, and AffineConstraints only knows its own lines.
    output_constrained.assign(output_dofs.size(), false);

    for(auto & cell : cells)
    {
      cell.gather.resize(dofs_per_cell);
      cell.scatter.resize(dofs_per_cell);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        cell.gather[i] = source_position.at(cell.dofs[i]);

        auto const found = output_position.find(cell.dofs[i]);
        cell.scatter[i]  = (found == output_position.end()) ?
                             dealii::numbers::invalid_unsigned_int :
                             found->second;

        if(cell.scatter[i] != dealii::numbers::invalid_unsigned_int and cell.constrained[i])
          output_constrained[cell.scatter[i]] = true;
      }
    }
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
    for(auto const & cell : cells)
      for(auto const & contribution : cell.contributions)
        blocks.insert(contribution.block);

    return std::vector<unsigned int>(blocks.begin(), blocks.end());
  }

  /**
   * Applies the restricted operator at the given block coefficients.
   *
   * Coefficients, not parameters: a single affine component is the operator with the indicator
   * of one block, and how a parameter maps to a coefficient is decided in Python. Purely local --
   * the stencil was replicated at construction.
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

    for(auto const & cell : cells)
    {
      // The operator reads zero from constrained columns and acts as the identity on constrained
      // rows. Replicating both is what makes the restriction agree with the full apply exactly
      // rather than merely closely.
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        local_values[i] = cell.constrained[i] ? 0.0 : source_values[cell.gather[i]];

      for(auto const & contribution : cell.contributions)
      {
        double const coefficient = diffusivity[contribution.block];

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          if(cell.scatter[i] == dealii::numbers::invalid_unsigned_int or cell.constrained[i])
            continue;

          double value = 0.0;
          for(unsigned int j = 0; j < dofs_per_cell; ++j)
            value += contribution.matrix[i * dofs_per_cell + j] * local_values[j];

          result[cell.scatter[i]] += coefficient * value;
        }
      }
    }

    for(unsigned int i = 0; i < output_dofs.size(); ++i)
      if(output_constrained[i])
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

    template<class Archive>
    void
    serialize(Archive & archive, unsigned int const /*version*/)
    {
      archive & block & matrix;
    }
  };

  /**
   * One stencil cell, in a form that survives being sent to another rank.
   *
   * Degrees of freedom are global indices, which is what makes that possible: the gather and
   * scatter positions below are rebuilt locally once the union is known. `constrained` is char
   * rather than bool because std::vector<bool> does not serialise as a container of values.
   */
  struct Cell
  {
    std::vector<dealii::types::global_dof_index> dofs;
    std::vector<char>                            constrained;
    std::vector<Contribution>                    contributions;

    std::vector<unsigned int> gather;
    std::vector<unsigned int> scatter;

    template<class Archive>
    void
    serialize(Archive & archive, unsigned int const /*version*/)
    {
      archive & dofs & constrained & contributions;
    }
  };

  unsigned int                                 blocks_per_dim;
  std::vector<dealii::types::global_dof_index> output_dofs;
  std::vector<dealii::types::global_dof_index> source_dofs;

  std::vector<Cell> cells;
  std::vector<bool> output_constrained;

  unsigned int dofs_per_cell;
};

} // namespace ExaDG

#endif /* EXADG_PYMOR_RESTRICTED_LAPLACE_H_ */
