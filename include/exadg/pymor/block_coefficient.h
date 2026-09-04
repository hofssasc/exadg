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

#ifndef EXADG_REDUCED_ORDER_BLOCK_COEFFICIENT_H_
#define EXADG_REDUCED_ORDER_BLOCK_COEFFICIENT_H_

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

// C/C++
#include <cmath>
#include <vector>

namespace ExaDG
{
/**
 * Piecewise constant coefficient on a Cartesian grid of blocks covering the unit hypercube.
 *
 * The domain (0,1)^dim is subdivided into blocks_per_dim^dim equally sized blocks, numbered
 * lexicographically with the x-direction running fastest, so that block index
 *
 *   p = i_0 + blocks_per_dim * i_1 + blocks_per_dim^2 * i_2
 *
 * corresponds to the block with indices (i_0, ..., i_{dim-1}).
 *
 * This is the parameterization of the thermal block benchmark. Because the coefficient is
 * constant per block, the operator is *exactly* affine in the block values,
 *
 *   A(mu) = sum_p exp(mu_p) A_p,
 *
 * where A_p is the operator assembled with the indicator function of block p as coefficient.
 * That exactness is the reason for choosing a piecewise constant parameterization: the error
 * of the reduced-order model is then purely due to basis truncation, with no additional
 * interpolation error of the coefficient confounding it.
 */
template<int dim>
class BlockCoefficient : public dealii::Function<dim>
{
public:
  /**
   * Constructs a coefficient with one value per block.
   *
   * @param blocks_per_dim Number of blocks per coordinate direction.
   * @param block_values Coefficient value of each block, length blocks_per_dim^dim in
   *        lexicographic order.
   */
  BlockCoefficient(unsigned int const blocks_per_dim, std::vector<double> const & block_values)
    : dealii::Function<dim>(1),
      blocks_per_dim(blocks_per_dim),
      block_values(block_values)
  {
    AssertThrow(blocks_per_dim > 0, dealii::ExcMessage("blocks_per_dim must be positive."));
    AssertThrow(block_values.size() == n_blocks(blocks_per_dim),
                dealii::ExcMessage("Expected blocks_per_dim^dim coefficient values, got " +
                                   std::to_string(block_values.size()) + " instead of " +
                                   std::to_string(n_blocks(blocks_per_dim)) + "."));
  }

  /**
   * Convenience constructor for the indicator function of a single block, which is what the
   * affine components A_p are assembled with.
   */
  static BlockCoefficient<dim>
  indicator(unsigned int const blocks_per_dim, unsigned int const block_index)
  {
    std::vector<double> values(n_blocks(blocks_per_dim), 0.0);
    AssertThrow(block_index < values.size(),
                dealii::ExcMessage("Block index out of range."));
    values[block_index] = 1.0;

    return BlockCoefficient<dim>(blocks_per_dim, values);
  }

  /**
   * Total number of blocks, blocks_per_dim^dim.
   */
  static unsigned int
  n_blocks(unsigned int const blocks_per_dim)
  {
    unsigned int count = 1;
    for(unsigned int d = 0; d < dim; ++d)
      count *= blocks_per_dim;

    return count;
  }

  /**
   * Lexicographic index of the block containing @p p, with the x-direction running fastest.
   * Points on the upper boundary are assigned to the last block.
   */
  static unsigned int
  block_index_of(unsigned int const blocks_per_dim, dealii::Point<dim> const & p)
  {
    unsigned int index  = 0;
    unsigned int stride = 1;

    for(unsigned int d = 0; d < dim; ++d)
    {
      int const scaled = static_cast<int>(std::floor(p[d] * blocks_per_dim));

      // clamp so that points on the domain boundary, and points slightly outside due to
      // round-off, still land in a valid block
      unsigned int const i =
        static_cast<unsigned int>(std::min(std::max(scaled, 0), static_cast<int>(blocks_per_dim) - 1));

      index += i * stride;
      stride *= blocks_per_dim;
    }

    return index;
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const override
  {
    (void)component;

    return block_values[block_index_of(blocks_per_dim, p)];
  }

private:
  unsigned int const  blocks_per_dim;
  std::vector<double> block_values;
};

} // namespace ExaDG

#endif /* EXADG_REDUCED_ORDER_BLOCK_COEFFICIENT_H_ */
