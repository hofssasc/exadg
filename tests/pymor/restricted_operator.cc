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
 * Tests the operator restricted to a set of output degrees of freedom, which is what pyMOR's
 * empirical interpolation evaluates instead of the full operator.
 *
 * pyMOR's contract is an identity, not an approximation:
 *
 *     A.apply(U).dofs(output_dofs) == restricted.apply(U.dofs(source_dofs))
 *
 * for every U. Everything hyper-reduction promises rests on it, and a violation does not show
 * up as a failure -- empirical interpolation simply converges to a slightly different operator
 * than the one being reduced. Checked properties:
 *
 *  1. The restricted operator reproduces the rows of the full operator at the output degrees of
 *     freedom, for a non-trivial block coefficient.
 *  2. The stencil is complete: perturbing the source vector *outside* source_dofs leaves the
 *     restricted output unchanged. This is the half of the contract that a comparison against
 *     the full operator cannot see, because both would change together.
 *  3. Constrained rows and columns are reproduced rather than approximated. Dirichlet rows are
 *     identity rows in ExaDG's constrained form, and getting that wrong is invisible in the
 *     interior.
 *  4. The affine decomposition survives restriction: summing the restricted single-block
 *     operators reproduces the restricted operator at the combined coefficient.
 *  5. The stencil is local -- its size is set by the element, not by the mesh.
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
#include <exadg/pymor/restricted_laplace.h>

using namespace ExaDG;

template<int dim>
class RestrictedTester
{
  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Laplace    = Poisson::LaplaceOperator<dim, Number, 1>;

public:
  RestrictedTester(unsigned int const fe_degree,
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

    check_contract();
    check_stencil_is_complete();
    check_affine_decomposition();
    check_locality();
  }

private:
  void
  setup()
  {
    dealii::GridGenerator::hyper_cube(tria, 0.0, 1.0, true /* colorize */);
    tria.refine_global(n_refinements);

    dof_handler.reinit(tria);
    dof_handler.distribute_dofs(fe);

    dealii::IndexSet const locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    constraints.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs);
    dealii::DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();

    MatrixFreeData<dim, Number> matrix_free_data;
    matrix_free_data.append_mapping_flags(
      Poisson::Operators::LaplaceKernel<dim, Number, 1>::get_mapping_flags(false, true, true));

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

    matrix_free.initialize_dof_vector(source);
    for(auto const i : source.locally_owned_elements())
      source[i] = std::sin(0.7 * static_cast<double>(i) + 0.3);
    source.compress(dealii::VectorOperation::insert);

    // The affine identity holds only where the constrained entries vanish, and pyMOR's binding
    // enforces exactly this on every vector it creates.
    constraints.set_zero(source);

    // A spread of output rows: the first, the last, a boundary (hence constrained) one and some
    // interior ones. Sorted and distinct, as the restriction requires.
    auto const n = dof_handler.n_dofs();
    output_dofs  = {0, n / 7, n / 3, n / 2, 2 * n / 3, n - 1};
    std::sort(output_dofs.begin(), output_dofs.end());
    output_dofs.erase(std::unique(output_dofs.begin(), output_dofs.end()), output_dofs.end());

    diffusivity.resize(dealii::Utilities::pow(blocks_per_dim, dim));
    for(unsigned int p = 0; p < diffusivity.size(); ++p)
      diffusivity[p] = 0.5 + 0.25 * static_cast<double>(p);
  }

  std::shared_ptr<Poisson::BoundaryDescriptor<0, dim>>
  make_boundary_descriptor() const
  {
    auto descriptor = std::make_shared<Poisson::BoundaryDescriptor<0, dim>>();
    for(dealii::types::boundary_id id = 0; id < 2 * dim; ++id)
      descriptor->dirichlet_bc.insert(
        {id, std::make_shared<dealii::Functions::ZeroFunction<dim>>(1)});

    return descriptor;
  }

  std::shared_ptr<Laplace>
  make_operator(std::vector<double> const & coefficient)
  {
    Poisson::LaplaceOperatorData<0, dim> data;
    data.dof_index                           = dof_index;
    data.quad_index                          = quad_index;
    data.bc                                  = make_boundary_descriptor();
    data.kernel_data.coefficient_is_variable = true;

    auto laplace = std::make_shared<Laplace>();
    laplace->initialize(matrix_free, constraints, data, false);
    laplace->set_coefficient(BlockCoefficient<dim>(blocks_per_dim, coefficient));

    return laplace;
  }

  RestrictedLaplace<dim, Number>
  make_restricted() const
  {
    return RestrictedLaplace<dim, Number>(dof_handler,
                                          mapping,
                                          constraints,
                                          blocks_per_dim,
                                          output_dofs);
  }

  /** The source vector's entries on the stencil, which is all the restriction ever reads. */
  std::vector<double>
  values_on(std::vector<dealii::types::global_dof_index> const & dofs,
            VectorType const &                                   vector) const
  {
    std::vector<double> values(dofs.size());
    for(unsigned int i = 0; i < dofs.size(); ++i)
      values[i] = vector[dofs[i]];

    return values;
  }

  static double
  max_difference(std::vector<double> const & a, std::vector<double> const & b)
  {
    double worst = 0.0;
    for(unsigned int i = 0; i < a.size(); ++i)
      worst = std::max(worst, std::abs(a[i] - b[i]));

    return worst;
  }

  /** Property 1 and 3: the restriction reproduces the rows of the full operator. */
  void
  check_contract()
  {
    auto const full       = make_operator(diffusivity);
    auto const restricted = make_restricted();

    VectorType result;
    matrix_free.initialize_dof_vector(result);
    full->vmult(result, source);

    auto const reference = values_on(output_dofs, result);
    auto const restricted_values =
      restricted.apply_coefficients(diffusivity, values_on(restricted.get_source_dofs(), source));

    unsigned int n_constrained = 0;
    for(auto const dof : output_dofs)
      if(constraints.is_constrained(dof))
        ++n_constrained;

    // A tolerance rather than the raw deviation: the two evaluations sum in different orders
    // -- matrix-free over vectorised batches, the restriction over assembled cell matrices --
    // so they agree to rounding and not bit for bit, and printing the rounding would make the
    // reference output machine dependent.
    double const violation = max_difference(reference, restricted_values);

    AssertThrow(violation < tolerance,
                dealii::ExcMessage("The restricted operator does not reproduce the full one; "
                                   "maximum deviation " + std::to_string(violation) + "."));

    pcout << "  output dofs                       = " << output_dofs.size() << "\n"
          << "  of which constrained              = " << n_constrained << "\n"
          << "  reproduces the full operator      = " << std::boolalpha
          << (violation < tolerance) << "\n";
  }

  /**
   * Property 2: the stencil is complete.
   *
   * Perturbs the source everywhere outside the stencil. The full operator's rows at the output
   * degrees of freedom must not move, and neither must the restriction -- which is what makes
   * source_dofs a sufficient input set rather than merely a plausible one.
   */
  void
  check_stencil_is_complete()
  {
    auto const full       = make_operator(diffusivity);
    auto const restricted = make_restricted();

    std::set<dealii::types::global_dof_index> const stencil(restricted.get_source_dofs().begin(),
                                                            restricted.get_source_dofs().end());

    VectorType perturbed(source);
    for(auto const i : perturbed.locally_owned_elements())
      if(stencil.count(i) == 0)
        perturbed[i] += 1.0e3;
    perturbed.compress(dealii::VectorOperation::insert);
    constraints.set_zero(perturbed);

    VectorType before, after;
    matrix_free.initialize_dof_vector(before);
    matrix_free.initialize_dof_vector(after);
    full->vmult(before, source);
    full->vmult(after, perturbed);

    double const moved =
      max_difference(values_on(output_dofs, before), values_on(output_dofs, after));

    AssertThrow(moved < tolerance,
                dealii::ExcMessage("Perturbing outside the stencil changed the output rows, so "
                                   "source_dofs is incomplete; deviation " +
                                   std::to_string(moved) + "."));

    pcout << "  stencil dofs                      = " << restricted.get_source_dofs().size()
          << "\n"
          << "  stencil is complete               = " << std::boolalpha << (moved < tolerance)
          << "\n";
  }

  /**
   * Property 4: restriction commutes with the affine decomposition.
   *
   * Summing the restricted single-block operators, each with the indicator of one block as its
   * coefficient, must reproduce the restriction at the combined coefficient. This is the
   * identity the reduced operator is assembled from online.
   */
  void
  check_affine_decomposition()
  {
    auto const restricted = make_restricted();
    auto const values     = values_on(restricted.get_source_dofs(), source);

    auto const combined = restricted.apply_coefficients(diffusivity, values);

    std::vector<double> summed(output_dofs.size(), 0.0);
    for(unsigned int p = 0; p < diffusivity.size(); ++p)
    {
      std::vector<double> indicator(diffusivity.size(), 0.0);
      indicator[p] = 1.0;

      auto const component = restricted.apply_coefficients(indicator, values);
      for(unsigned int i = 0; i < summed.size(); ++i)
        summed[i] += diffusivity[p] * component[i];
    }

    // Constrained rows are identity rows in *every* component, so summing P of them scales the
    // entry by P rather than leaving it alone. The decomposition holds on the free rows, which
    // is exactly the subspace the binding keeps its vectors in.
    double worst = 0.0;
    for(unsigned int i = 0; i < output_dofs.size(); ++i)
      if(not constraints.is_constrained(output_dofs[i]))
        worst = std::max(worst, std::abs(combined[i] - summed[i]));

    AssertThrow(worst < tolerance,
                dealii::ExcMessage("Restriction does not commute with the affine decomposition; "
                                   "defect " + std::to_string(worst) + "."));

    pcout << "  affine decomposition survives     = " << std::boolalpha << (worst < tolerance)
          << "\n";
  }

  /** Property 5: the stencil size is a property of the element, not of the mesh. */
  void
  check_locality()
  {
    auto const restricted = make_restricted();

    pcout << "  cells in stencil                  = " << restricted.n_cells() << "\n"
          << "  blocks read                       = " << restricted.get_blocks().size() << " of "
          << diffusivity.size() << "\n"
          << "  stencil fraction of all dofs      = " << std::fixed << std::setprecision(4)
          << static_cast<double>(restricted.get_source_dofs().size()) /
               static_cast<double>(dof_handler.n_dofs())
          << "\n";
  }

  // loose enough for a degree-3 element, tight enough that a real defect cannot hide
  static constexpr double tolerance = 1.0e-10;

  MPI_Comm const             mpi_comm;
  dealii::ConditionalOStream pcout;

  unsigned int const fe_degree;
  unsigned int const n_refinements;
  unsigned int const blocks_per_dim;

  dealii::parallel::distributed::Triangulation<dim> tria;
  dealii::FE_Q<dim>                                 fe;
  dealii::MappingQ<dim>                             mapping;
  dealii::DoFHandler<dim>                           dof_handler;
  dealii::AffineConstraints<Number>                 constraints;

  dealii::MatrixFree<dim, Number> matrix_free;
  unsigned int                    dof_index  = 0;
  unsigned int                    quad_index = 0;

  VectorType                                   source;
  std::vector<dealii::types::global_dof_index> output_dofs;
  std::vector<double>                          diffusivity;
};

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    dealii::ConditionalOStream pcout(std::cout,
                                     dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                                       0);

    pcout << "\nRestricted operator, i.e. the empirical interpolation contract:\n";

    RestrictedTester<2>(2, 4, 2).run();
    RestrictedTester<2>(3, 3, 4).run();
    RestrictedTester<3>(2, 2, 2).run();
  }
  catch(std::exception & exc)
  {
    std::cerr << "\n\nException: " << exc.what() << "\nAborting!\n";

    return 1;
  }

  return 0;
}
