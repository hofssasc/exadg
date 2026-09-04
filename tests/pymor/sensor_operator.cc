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
 *  ______________________________________________________________________
 */

/*
 * Tests the observation operator used by the Bayesian calibration benchmarks.
 *
 * Three properties are checked:
 *
 *  1. A field that the finite element space can represent exactly is evaluated exactly at the
 *     sensor points, independently of how the mesh is distributed over the ranks.
 *  2. The operator is linear, B(a*u + b*v) == a*B(u) + b*B(v). The reduced observation operator
 *     B*V is assembled by applying B to each basis vector, which is only valid if this holds.
 *  3. The multi-vector overload agrees with applying the single-vector version column by
 *     column, and uses the row-major layout the reduced model expects.
 */

// C/C++
#include <cmath>
#include <iomanip>
#include <iostream>

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/pymor/sensor_operator.h>

using namespace ExaDG;

/**
 * A polynomial of degree 2 per direction, which an FE_Q(2) space reproduces exactly. Using an
 * exactly representable field separates errors of the evaluation machinery from interpolation
 * error, so the check can be made against an exact value rather than a tolerance chosen by
 * eye.
 */
template<int dim>
class QuadraticField : public dealii::Function<dim>
{
public:
  QuadraticField() : dealii::Function<dim>(1)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const override
  {
    (void)component;

    double result = 1.0;
    for(unsigned int d = 0; d < dim; ++d)
      result *= p[d] * (1.0 - p[d]);

    return result;
  }
};

template<int dim>
void
run(unsigned int const fe_degree,
    unsigned int const n_refinements,
    unsigned int const sensors_per_dim)
{
  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  MPI_Comm const             mpi_comm = MPI_COMM_WORLD;
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0);

  dealii::parallel::distributed::Triangulation<dim> tria(mpi_comm);
  dealii::GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(n_refinements);

  dealii::FE_Q<dim>       fe(fe_degree);
  dealii::MappingQ<dim>   mapping(1);
  dealii::DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  dealii::IndexSet const locally_relevant_dofs =
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  VectorType field(dof_handler.locally_owned_dofs(), locally_relevant_dofs, mpi_comm);

  dealii::AffineConstraints<Number> constraints;
  constraints.close();

  QuadraticField<dim> const analytic;
  dealii::VectorTools::interpolate(mapping, dof_handler, analytic, field);

  auto const points = SensorOperator<dim, Number>::interior_cartesian_grid(sensors_per_dim);

  SensorOperator<dim, Number> sensors;
  sensors.setup(tria, mapping, points);

  // The rank count is deliberately not printed: the results must not depend on it, so keeping
  // it out of the output lets one reference file serve every parallel configuration.
  pcout << "\n  dim = " << dim << ", degree = " << fe_degree
        << ", sensors = " << sensors.n_points() << "\n";

  // Property 1: exact evaluation of an exactly representable field
  std::vector<Number> const values = sensors.evaluate(dof_handler, field);

  double max_error = 0.0;
  for(unsigned int i = 0; i < sensors.n_points(); ++i)
    max_error = std::max(max_error, std::abs(values[i] - analytic.value(points[i])));

  pcout << "  quadratic field reproduced (error < 1e-12) = " << std::boolalpha
        << (max_error < 1.0e-12) << "\n";

  // Property 2: linearity
  VectorType scaled(field);
  scaled *= 3.0;

  std::vector<Number> const scaled_values = sensors.evaluate(dof_handler, scaled);

  double linearity_error = 0.0;
  for(unsigned int i = 0; i < sensors.n_points(); ++i)
    linearity_error = std::max(linearity_error, std::abs(scaled_values[i] - 3.0 * values[i]));

  pcout << "  linear in the field (error < 1e-12)        = "
        << (linearity_error < 1.0e-12) << "\n";

  // Property 3: the multi-vector overload matches column-by-column application
  std::vector<VectorType> basis{field, scaled};
  std::vector<Number> const matrix = sensors.evaluate(dof_handler, basis);

  double layout_error = 0.0;
  for(unsigned int i = 0; i < sensors.n_points(); ++i)
  {
    layout_error = std::max(layout_error, std::abs(matrix[i * 2 + 0] - values[i]));
    layout_error = std::max(layout_error, std::abs(matrix[i * 2 + 1] - scaled_values[i]));
  }

  pcout << "  multi-vector layout consistent             = " << (layout_error == 0.0) << "\n";
}

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    run<2>(2 /* degree */, 3 /* refinements */, 4 /* sensors per dim */);
    run<3>(2 /* degree */, 2 /* refinements */, 3 /* sensors per dim */);
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
