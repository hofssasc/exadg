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
 * Tests that the global degree of freedom numbering does not depend on the number of MPI ranks.
 *
 * The pyMOR binding addresses individual entries by global index: VectorArray.dofs() gathers
 * them, and Operator.restricted() is defined on a set of them. Both are only meaningful if the
 * index of a given *physical* degree of freedom is independent of how the mesh was partitioned.
 * deal.II numbers degrees of freedom along the p4est space filling curve, and changing the rank
 * count changes only where that curve is cut, not its order -- but nothing enforces this, so it
 * is checked here. It is the precondition for running empirical interpolation under MPI.
 *
 * The check works by filling a vector from a smooth function of each degree of freedom's support
 * point, so the entry at a global index is determined by physical position. If the numbering were
 * partition dependent, the resulting global array would be a permutation of itself between rank
 * counts. A plain sum would not notice; the index weighted sum below would.
 *
 * The test therefore prints no rank dependent quantity at all, and the invariant is expressed as
 * the reference outputs for one, two and four ranks being byte identical. A partitioning
 * dependence shows up as a diff rather than as a silently corrupted basis much later on.
 */

// C/C++
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

unsigned int const dim = 2;

using Number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

/**
 * Value assigned to a degree of freedom sitting at point p. Any smooth function of position
 * works; this one is chosen to have no symmetry that could make a permutation undetectable.
 */
double
value_at(dealii::Point<dim> const & p)
{
  return std::sin(3.0 * p[0]) * std::cos(5.0 * p[1]) + 0.25 * p[0] * p[1];
}

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    MPI_Comm const     mpi_comm  = MPI_COMM_WORLD;
    unsigned int const this_rank = dealii::Utilities::MPI::this_mpi_process(mpi_comm);

    dealii::ConditionalOStream pcout(std::cout, this_rank == 0);
    pcout << std::scientific << std::setprecision(8);

    dealii::parallel::distributed::Triangulation<dim> triangulation(mpi_comm);
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(3);

    dealii::FE_Q<dim>       fe(2);
    dealii::DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    dealii::types::global_dof_index const n_dofs = dof_handler.n_dofs();

    dealii::MappingQ1<dim> const mapping;
    std::map<dealii::types::global_dof_index, dealii::Point<dim>> const support_points =
      dealii::DoFTools::map_dofs_to_support_points(mapping, dof_handler);

    dealii::IndexSet const owned = dof_handler.locally_owned_dofs();

    VectorType written(owned, mpi_comm);
    for(auto const i : owned)
      written[i] = value_at(support_points.at(i));
    written.compress(dealii::VectorOperation::insert);

    // A plain sum is invariant under any permutation of the numbering, so it can only confirm
    // that the same set of values is present. The index weighted sum is what actually pins the
    // ordering down, and it is the quantity a partitioning dependence would change.
    double local_sum = 0.0, local_weighted = 0.0;
    for(auto const i : owned)
    {
      local_sum += written[i];
      local_weighted += static_cast<double>(i + 1) * written[i];
    }

    double const sum      = dealii::Utilities::MPI::sum(local_sum, mpi_comm);
    double const weighted = dealii::Utilities::MPI::sum(local_weighted, mpi_comm);

    // Sampled entries, so a diff points at where the ordering changed rather than only saying
    // that some checksum moved. Only the owning rank contributes; the sum acts as a gather.
    std::vector<dealii::types::global_dof_index> const samples = {0,
                                                                  n_dofs / 4,
                                                                  n_dofs / 2,
                                                                  3 * n_dofs / 4,
                                                                  n_dofs - 1};

    pcout << "  number of degrees of freedom      = " << n_dofs << "\n"
          << "  sum of entries                    = " << sum << "\n"
          << "  index weighted sum                = " << weighted << "\n";

    for(auto const g : samples)
    {
      double const local = owned.is_element(g) ? written[g] : 0.0;
      double const x     = owned.is_element(g) ? support_points.at(g)[0] : 0.0;
      double const y     = owned.is_element(g) ? support_points.at(g)[1] : 0.0;

      pcout << "  entry " << std::setw(5) << g << " at ("
            << std::fixed << std::setprecision(4) << dealii::Utilities::MPI::sum(x, mpi_comm)
            << ", " << dealii::Utilities::MPI::sum(y, mpi_comm) << ")"
            << std::scientific << std::setprecision(8)
            << " = " << dealii::Utilities::MPI::sum(local, mpi_comm) << "\n";
    }
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
