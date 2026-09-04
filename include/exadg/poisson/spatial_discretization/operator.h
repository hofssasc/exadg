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

#ifndef EXADG_POISSON_SPATIAL_DISCRETIZATION_OPERATOR_H_
#define EXADG_POISSON_SPATIAL_DISCRETIZATION_OPERATOR_H_

// ExaDG
#include <exadg/grid/grid.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/rhs_operator.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, int n_components, typename Number>
class Operator : public dealii::EnableObserverPointer
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef LaplaceOperator<dim, Number, n_components> Laplace;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

public:
  Operator(std::shared_ptr<Grid<dim> const>                      grid,
           std::shared_ptr<dealii::Mapping<dim> const>           mapping,
           std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
           std::shared_ptr<BoundaryDescriptor<rank, dim> const>  boundary_descriptor,
           std::shared_ptr<FieldFunctions<dim> const>            field_functions,
           Parameters const &                                    param,
           std::string const &                                   field,
           MPI_Comm const &                                      mpi_comm);

  void
  fill_matrix_free_data(MatrixFreeData<dim, Number> & matrix_free_data) const;

  /**
   * Call this setup() function if the dealii::MatrixFree object can be set up by the present class.
   */
  void
  setup();

  /**
   * Call this setup() function if the dealii::MatrixFree object needs to be created outside this
   * class. The typical use case would be multiphysics-coupling with one MatrixFree object handed
   * over to several single-field solvers.
   */
  void
  setup(std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free,
        std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data);

  void
  initialize_dof_vector(VectorType & src) const;

  /*
   * Prescribe initial conditions using a specified analytical function.
   */
  void
  prescribe_initial_conditions(VectorType & src) const;

  void
  rhs(VectorType & dst, double const time = 0.0) const;

  void
  vmult(VectorType & dst, VectorType const & src) const;

  void
  evaluate(VectorType & dst, VectorType const & src, double const time = 0.0) const;

  unsigned int
  solve(VectorType & sol, VectorType const & rhs, double const time) const;

#ifdef DEAL_II_WITH_TRILINOS
  /**
   * Allocates the assembled counterpart of the matrix-free operator.
   *
   * Exposed because a *non-symmetric* extension of this operator -- a convection-diffusion
   * problem with a fixed transport field -- cannot be solved by the Krylov method configured
   * here, and a sparse direct solve is both robust at every Peclet number and, once factorised,
   * the cheapest route to the many adjoint solves an exact Jacobian needs. Taking the matrix
   * from the operator rather than reassembling it elsewhere is what guarantees that the system
   * being solved is the operator being projected.
   *
   * The sparsity pattern is fixed, so this is called once; calculate_system_matrix() then
   * refills it at each coefficient.
   */
  void
  init_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix) const;

  /**
   * Fills the assembled matrix for the coefficient currently installed.
   */
  void
  calculate_system_matrix(dealii::TrilinosWrappers::SparseMatrix & system_matrix) const;
#endif

  /**
   * Prescribes the spatially varying coefficient a(x) of -div(a(x) grad(u)).
   *
   * Requires Parameters::coefficient_is_variable. Applying the operator afterwards
   * (vmult(), solve()) uses the new coefficient, so a parameter sweep amounts to repeated
   * calls of this function followed by solve().
   *
   * Evaluating the operator once per subdomain indicator function yields the affine
   * components A_p of A(mu) = sum_p exp(mu_p) A_p, which is what the reduced-order model
   * projects.
   */
  void
  set_coefficient(dealii::Function<dim> const & function);

  /**
   * Prescribes the coefficient as one value per active cell.
   *
   * The form a piecewise constant field takes on an unstructured mesh, where there is no
   * geometric rule to evaluate. See
   * LaplaceOperator::set_coefficient_from_cell_values().
   */
  void
  set_coefficient_from_cell_values(std::vector<Number> const & cell_values);

  /**
   * Rebuilds the preconditioner for the current coefficient.
   *
   * The preconditioner is constructed once during setup. Changing the coefficient afterwards
   * leaves it valid but increasingly ineffective, since it then approximates a different
   * operator; for the high contrasts arising in a parameter sweep the iteration count degrades
   * badly. Call this after set_coefficient() when the operator is to be solved rather than only
   * applied.
   */
  void
  update_preconditioner();

  /*
   * Setters and getters.
   */

  std::shared_ptr<dealii::MatrixFree<dim, Number> const>
  get_matrix_free() const;

  dealii::DoFHandler<dim> const &
  get_dof_handler() const;

  dealii::types::global_dof_index
  get_number_of_dofs() const;

  double
  get_n_10() const;

  double
  get_average_convergence_rate() const;

  // Multiphysics coupling via "Cached" boundary conditions
  std::shared_ptr<ContainerInterfaceData<rank, dim, double>>
  get_container_interface_data() const;

  std::shared_ptr<TimerTree>
  get_timings() const;

  std::shared_ptr<dealii::Mapping<dim> const>
  get_mapping() const;

  // TODO: we currently need this function public for precice-based FSI
  unsigned int
  get_dof_index() const;

  unsigned int
  get_quad_index() const;

  /**
   * The finite element space the variable coefficient lives in.
   *
   * Its own DoFHandler, so the coefficient's degree is independent of the solution's: degree 0
   * gives one value per cell, degree 1 a continuous multilinear field. Only allocated when
   * Parameters::coefficient_is_variable.
   */
  dealii::DoFHandler<dim> const &
  get_coefficient_dof_handler() const;

  unsigned int
  get_coefficient_dof_index() const;

  /**
   * Number of coefficient degrees of freedom, i.e. of calibration parameters.
   */
  dealii::types::global_dof_index
  n_coefficient_dofs() const;

  void
  initialize_coefficient_dof_vector(VectorType & vector) const;

  /**
   * Prescribes the coefficient from a vector of its own degrees of freedom.
   *
   * The general form: the coefficient is expanded as ``a = sum_i c_i phi_i`` in the coefficient
   * space, so the entries are the expansion coefficients ``c_i``. Degree 0 makes them cell
   * values, which is the piecewise constant field; degree 1 makes them nodal values.
   *
   * The expansion is what keeps the operator affine, ``A(c) = sum_i c_i A[phi_i]``, at any
   * degree -- so the reduced model's structure, and every derivative built on it, is unchanged
   * by raising the degree.
   */
  void
  set_coefficient_from_dof_vector(VectorType const & coefficient_values,
                                  bool const         check_positivity = true);

  /**
   * Derivative of ``p^T A(c) u`` with respect to every coefficient degree of freedom.
   *
   * The parameter half of an adjoint gradient: with @p p solving the adjoint problem, the
   * derivative of a functional ``J`` of the solution is ``dJ/dc_i = -[this]_i``, and it comes
   * out for all coefficient degrees of freedom in a single cell loop rather than in one operator
   * apply per parameter. See LaplaceOperator::compute_coefficient_sensitivity().
   *
   * @param dst Sized by initialize_coefficient_dof_vector().
   */
  void
  compute_coefficient_sensitivity(VectorType const & u,
                                  VectorType const & p,
                                  VectorType &       dst) const;

  /**
   * The transpose of the constrained system matrix, for reference and for testing.
   *
   * The Laplacian with a scalar coefficient is self-adjoint and ExaDG's constrained form keeps
   * it symmetric, so this delegates to vmult(). It exists so that an adjoint solver reads as an
   * adjoint solver: a caller that writes ``vmult()`` where it means ``A^T`` is correct here and
   * silently wrong for the first non-symmetric operator the same code is pointed at.
   */
  void
  vmult_transpose(VectorType & dst, VectorType const & src) const;

private:
  std::string
  get_dof_name() const;

  unsigned int
  get_dof_index_periodicity_and_hanging_node_constraints() const;

  std::string
  get_dof_name_periodicity_and_hanging_node_constraints() const;

  std::string
  get_coefficient_dof_name() const;

  std::string
  get_quad_name() const;

  std::string
  get_quad_gauss_lobatto_name() const;

  unsigned int
  get_quad_index_gauss_lobatto() const;

  void
  initialize_dof_handler_and_constraints();

  void
  setup_coupling_boundary_conditions();

  void
  setup_operators();

  void
  setup_preconditioner_and_solver();

  /*
   * Grid
   */
  std::shared_ptr<Grid<dim> const> grid;

  /*
   * Mapping
   */
  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings;

  /*
   * User interface: Boundary conditions and field functions.
   */
  std::shared_ptr<BoundaryDescriptor<rank, dim> const> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim> const>           field_functions;

  /*
   * List of parameters.
   */
  Parameters const & param;

  std::string const field;

  /*
   * Basic finite element ingredients.
   */
  std::shared_ptr<dealii::FiniteElement<dim>> fe;

  dealii::DoFHandler<dim> dof_handler;

  // the coefficient's own space; empty unless the coefficient is variable
  std::shared_ptr<dealii::FiniteElement<dim>> coefficient_fe;

  dealii::DoFHandler<dim> coefficient_dof_handler;

  // no constraints on the coefficient: it carries no boundary conditions of its own, and
  // hanging nodes are excluded because the variable coefficient requires a uniform mesh anyway
  dealii::AffineConstraints<Number> coefficient_constraints;

  // This AffineConstraints object applies homogeneous boundary conditions as needed by vmult()/
  // apply() functions in iterative solvers for linear systems of equations and preconditioners
  // such as multigrid, implemented via dealii::MatrixFree and FEEvaluation::read_dof_values()
  // (or gather_evaluate()).
  // To deal with inhomogeneous boundary data, a separate object of type AffineConstraints is
  // needed (see below).
  mutable dealii::AffineConstraints<Number> affine_constraints;

  // To treat inhomogeneous Dirichlet BCs correctly in the context of matrix-free operator
  // evaluation using dealii::MatrixFree/FEEvaluation, we need a separate AffineConstraints
  // object containing only periodicity and hanging node constraints. This is only relevant
  // for continuous Galerkin discretizations.
  dealii::AffineConstraints<Number> affine_constraints_periodicity_and_hanging_nodes;

  std::string const dof_index = "dof";
  std::string const dof_index_periodicity_and_handing_node_constraints =
    "dof_periodicity_hanging_nodes";

  std::string const quad_index               = "quad";
  std::string const quad_index_gauss_lobatto = "quad_gauss_lobatto";

  std::shared_ptr<dealii::MatrixFree<dim, Number> const> matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number> const>     matrix_free_data;

  /*
   * Interface coupling
   */
  // TODO: The PDE operator should only have read access to interface data
  mutable std::shared_ptr<ContainerInterfaceData<rank, dim, double>>
    interface_data_dirichlet_cached;

  RHSOperator<dim, Number, n_components> rhs_operator;

  Laplace laplace_operator;

  std::shared_ptr<PreconditionerBase<Number>>     preconditioner;
  std::shared_ptr<Krylov::SolverBase<VectorType>> iterative_solver;

  /*
   * MPI
   */
  MPI_Comm const mpi_comm;

  /*
   * Output to screen.
   */
  dealii::ConditionalOStream pcout;
};
} // namespace Poisson
} // namespace ExaDG

#endif /* EXADG_POISSON_SPATIAL_DISCRETIZATION_OPERATOR_H_ */
