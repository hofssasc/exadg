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
 * Tests the variable coefficient of the Laplace operator, i.e. -div(a(x) grad(u)).
 *
 * Three properties are checked, in increasing order of importance:
 *
 *  1. a(x) == 1 reproduces the plain Laplacian bit for bit, so enabling the feature cannot
 *     silently change existing results.
 *  2. a(x) == c reproduces c times the plain Laplacian.
 *  3. The operator is exactly affine in the block values of a piecewise constant coefficient,
 *     A(sum_p c_p chi_p) == sum_p c_p A(chi_p).
 *
 * Property 3 is the one the reduced-order model depends on: it is what allows the affine
 * components A_p to be projected once offline and recombined online without ever touching the
 * full-order operator again.
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

// ExaDG
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/quadrature.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/pymor/block_coefficient.h>

using namespace ExaDG;

template<int dim>
class LaplaceTester
{
  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Laplace    = Poisson::LaplaceOperator<dim, Number, 1>;

public:
  LaplaceTester(unsigned int const fe_degree,
                unsigned int const n_refinements,
                unsigned int const blocks_per_dim)
    : mpi_comm(MPI_COMM_WORLD),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      fe_degree(fe_degree),
      n_refinements(n_refinements),
      blocks_per_dim(blocks_per_dim),
      tria(mpi_comm),
      fe(fe_degree),
      mapping(1)
  {
  }

  void
  run()
  {
    setup();

    pcout << "\n  dim = " << dim << ", degree = " << fe_degree
          << ", blocks per direction = " << blocks_per_dim
          << ", n_dofs = " << dof_handler.n_dofs() << "\n";

    check_constant_coefficient(1.0);
    check_constant_coefficient(2.5);
    check_affine_decomposition();
  }

private:
  void
  setup()
  {
    dealii::GridGenerator::hyper_cube(tria, 0.0, 1.0, true /* colorize */);
    tria.refine_global(n_refinements);

    dof_handler.reinit(tria);
    dof_handler.distribute_dofs(fe);

    // homogeneous Dirichlet conditions on the whole boundary
    dealii::IndexSet const locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
    dealii::DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();

    MatrixFreeData<dim, Number> matrix_free_data;
    matrix_free_data.append_mapping_flags(
      Poisson::Operators::LaplaceKernel<dim, Number, 1>::get_mapping_flags(
        false /* interior faces, CG */, true, true /* coefficient_is_variable */));

    matrix_free_data.insert_dof_handler(&dof_handler, "dof");
    matrix_free_data.insert_constraint(&constraints, "dof");

    std::shared_ptr<dealii::Quadrature<dim>> quadrature =
      create_quadrature<dim>(ElementType::Hypercube, fe_degree + 1);
    matrix_free_data.insert_quadrature(*quadrature, "quad");

    dof_index  = matrix_free_data.get_dof_index("dof");
    quad_index = matrix_free_data.get_quad_index("quad");

    matrix_free.reinit(mapping,
                       matrix_free_data.get_dof_handler_vector(),
                       matrix_free_data.get_constraint_vector(),
                       matrix_free_data.get_quadrature_vector(),
                       matrix_free_data.data);

    // a fixed, reproducible source vector to apply the operators to
    matrix_free.initialize_dof_vector(source);
    for(auto const i : source.locally_owned_elements())
      source[i] = std::sin(0.7 * static_cast<double>(i) + 0.3);
    source.compress(dealii::VectorOperation::insert);
    constraints.set_zero(source);
  }

  /**
   * Boundary descriptor with homogeneous Dirichlet data on every boundary id produced by
   * hyper_cube(colorize = true), i.e. 0 .. 2*dim-1.
   */
  std::shared_ptr<Poisson::BoundaryDescriptor<0, dim>>
  make_boundary_descriptor() const
  {
    auto descriptor = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();
    for(dealii::types::boundary_id id = 0; id < 2 * dim; ++id)
      descriptor->dirichlet_bc.insert({id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(1)});

    return descriptor;
  }

  /**
   * Builds a Laplace operator, either with a constant coefficient or with a variable one that
   * still has to be prescribed via set_coefficient().
   */
  std::shared_ptr<Laplace>
  make_operator(bool const coefficient_is_variable, double const constant_coefficient)
  {
    Poisson::LaplaceOperatorData<0, dim> data;
    data.dof_index                            = dof_index;
    data.quad_index                           = quad_index;
    data.bc                                   = make_boundary_descriptor();
    data.kernel_data.coefficient_is_variable  = coefficient_is_variable;
    data.kernel_data.coefficient              = constant_coefficient;

    auto laplace = std::make_shared<Laplace>();
    laplace->initialize(matrix_free, constraints, data, false /* assemble_matrix */);

    return laplace;
  }

  static double
  relative_difference(VectorType const & a, VectorType const & b)
  {
    VectorType difference(a);
    difference -= b;

    double const norm = a.l2_norm();

    return norm > 0.0 ? difference.l2_norm() / norm : difference.l2_norm();
  }

  /**
   * Property 1 and 2: a variable coefficient that happens to be constant must reproduce the
   * constant-coefficient operator.
   */
  void
  check_constant_coefficient(double const coefficient)
  {
    auto reference = make_operator(false, coefficient);

    auto variable = make_operator(true, 1.0);
    variable->set_coefficient(dealii::Functions::ConstantFunction<dim>(coefficient, 1));

    VectorType expected, actual;
    matrix_free.initialize_dof_vector(expected);
    matrix_free.initialize_dof_vector(actual);

    reference->vmult(expected, source);
    variable->vmult(actual, source);

    // a variable coefficient that is constant must reproduce the constant-coefficient path
    // exactly, not just to some tolerance, since the arithmetic is identical
    pcout << "  constant coefficient a = " << coefficient
          << ": reproduces constant-coefficient operator exactly = " << std::boolalpha
          << (relative_difference(expected, actual) == 0.0) << "\n";
  }

  /**
   * Property 3: exact affinity in the block values. This is what the reduced-order model
   * relies on, so it is checked against a genuinely varying coefficient rather than a
   * constant one.
   */
  void
  check_affine_decomposition()
  {
    unsigned int const n_blocks = BlockCoefficient<dim>::n_blocks(blocks_per_dim);

    // an arbitrary, strongly varying set of block values with high contrast
    std::vector<double> block_values(n_blocks);
    for(unsigned int p = 0; p < n_blocks; ++p)
      block_values[p] = std::exp(1.5 * std::sin(2.3 * static_cast<double>(p) + 0.4));

    auto combined = make_operator(true, 1.0);
    combined->set_coefficient(BlockCoefficient<dim>(blocks_per_dim, block_values));

    VectorType expected, contribution, accumulated;
    matrix_free.initialize_dof_vector(expected);
    matrix_free.initialize_dof_vector(contribution);
    matrix_free.initialize_dof_vector(accumulated);

    combined->vmult(expected, source);

    // sum_p c_p A(chi_p) x
    accumulated = 0.0;
    auto single = make_operator(true, 1.0);
    for(unsigned int p = 0; p < n_blocks; ++p)
    {
      single->set_coefficient(BlockCoefficient<dim>::indicator(blocks_per_dim, p));
      single->vmult(contribution, source);
      accumulated.add(block_values[p], contribution);
    }

    // The two paths accumulate the same contributions in a different order, so agreement is
    // only up to round-off. A tolerance a few orders above machine epsilon keeps the test
    // stable across platforms while still failing loudly if affinity is genuinely broken --
    // a real violation would show up at the size of the coefficient contrast, not at 1e-14.
    double constexpr tolerance = 1.0e-12;

    pcout << "  affine decomposition over " << n_blocks
          << " blocks: exact to round-off = " << std::boolalpha
          << (relative_difference(expected, accumulated) < tolerance) << "\n";
  }

  MPI_Comm                   mpi_comm;
  dealii::ConditionalOStream pcout;

  unsigned int const fe_degree;
  unsigned int const n_refinements;
  unsigned int const blocks_per_dim;

  dealii::parallel::distributed::Triangulation<dim> tria;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::FE_Q<dim>                                 fe;
  dealii::MappingQ<dim>                             mapping;
  dealii::AffineConstraints<Number>                 constraints;

  dealii::MatrixFree<dim, Number> matrix_free;

  unsigned int dof_index  = 0;
  unsigned int quad_index = 0;

  VectorType source;
};

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::scientific << std::setprecision(4);

    {
      LaplaceTester<2> tester(2 /* degree */, 3 /* refinements */, 4 /* blocks per dim */);
      tester.run();
    }
    {
      LaplaceTester<3> tester(2 /* degree */, 2 /* refinements */, 2 /* blocks per dim */);
      tester.run();
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
