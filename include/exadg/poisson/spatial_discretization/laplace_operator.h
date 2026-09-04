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

#ifndef EXADG_POISSON_SPATIAL_DISCRETIZATION_LAPLACE_OPERATOR_H_
#define EXADG_POISSON_SPATIAL_DISCRETIZATION_LAPLACE_OPERATOR_H_

// ExaDG
#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>
#include <exadg/grid/grid_data.h>
#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/operator_type.h>
#include <exadg/operators/variable_coefficients.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace Poisson
{
namespace Operators
{
struct LaplaceKernelData
{
  LaplaceKernelData() : IP_factor(1.0), coefficient_is_variable(false), coefficient(1.0)
  {
  }

  double IP_factor;

  /**
   * If true, the operator realizes -div(a(x) grad(u)) with a spatially varying coefficient
   * a(x) stored at the quadrature points. If false, the constant @p coefficient is used, and
   * the default value of 1.0 recovers the plain Laplacian.
   *
   * Currently only supported for continuous elements (CG). For DG the coefficient would in
   * addition have to enter the gradient and value fluxes and scale the interior penalty
   * parameter, which is not implemented yet.
   */
  bool coefficient_is_variable;

  /**
   * Constant coefficient, used when @p coefficient_is_variable is false.
   */
  double coefficient;
};

template<int dim, typename Number, int n_components = 1>
class LaplaceKernel
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

  typedef FaceIntegrator<dim, n_components, Number> IntegratorFace;

public:
  LaplaceKernel() : degree(1), tau(dealii::make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &               data_in,
         unsigned int const                      dof_index,
         unsigned int const                      quad_index)
  {
    data = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                                = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);

    if(data.coefficient_is_variable)
    {
      // face data is not needed as long as only continuous elements are supported
      coefficients.initialize(matrix_free,
                              quad_index,
                              false /* store_face_data */,
                              false /* store_cell_based_face_data */);
      coefficients.set_coefficients(dealii::make_vectorized_array<Number>(data.coefficient));
    }
  }

  /**
   * Returns the coefficient in quadrature point @p q of cell batch @p cell. Falls back to the
   * constant coefficient if no variable coefficient is in use, so callers do not have to
   * branch.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    get_coefficient_cell(unsigned int const cell, unsigned int const q) const
  {
    if(data.coefficient_is_variable)
      return coefficients.get_coefficient_cell(cell, q);

    return dealii::make_vectorized_array<Number>(data.coefficient);
  }

  void
  set_coefficient_cell(unsigned int const cell, unsigned int const q, scalar const & coefficient)
  {
    coefficients.set_coefficient_cell(cell, q, coefficient);
  }

  bool
  coefficient_is_variable() const
  {
    return data.coefficient_is_variable;
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter, matrix_free, dof_index);
  }

  IntegratorFlags
  get_integrator_flags(bool const is_dg) const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    if(is_dg)
    {
      flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
      flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    }
    else
    {
      // evaluation of Neumann BCs for continuous elements
      flags.face_evaluate  = dealii::EvaluationFlags::nothing;
      flags.face_integrate = dealii::EvaluationFlags::values;
    }

    return flags;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals,
                    bool const coefficient_is_variable = false)
  {
    MappingFlags flags;

    flags.cells = dealii::update_gradients | dealii::update_JxW_values;

    // set_coefficient() evaluates the coefficient function in the quadrature points, so their
    // real-space locations have to be available
    if(coefficient_is_variable)
      flags.cells |= dealii::update_quadrature_points;

    if(compute_interior_face_integrals)
    {
      flags.inner_faces =
        dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
    }

    if(compute_boundary_face_integrals)
    {
      flags.boundary_faces = dealii::update_gradients | dealii::update_JxW_values |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;
    }

    return flags;
  }

  void
  reinit_face(IntegratorFace &   integrator_m,
              IntegratorFace &   integrator_p,
              unsigned int const dof_index) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m, unsigned int const dof_index) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(
            degree,
            get_element_type(
              integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
            data.IP_factor);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p,
                         unsigned int const               dof_index) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                     integrator_p.read_cell_data(array_penalty_parameter)) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
    else // boundary face
    {
      tau = integrator_m.read_cell_data(array_penalty_parameter) *
            IP::get_penalty_factor<dim, Number>(
              degree,
              get_element_type(
                integrator_m.get_matrix_free().get_dof_handler(dof_index).get_triangulation()),
              data.IP_factor);
    }
  }

  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    calculate_gradient_flux(T const & value_m, T const & value_p) const
  {
    return -0.5 * (value_m - value_p);
  }

  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    calculate_value_flux(T const & normal_gradient_m,
                         T const & normal_gradient_p,
                         T const & value_m,
                         T const & value_p) const
  {
    return 0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p);
  }

private:
  LaplaceKernelData data;

  unsigned int degree;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;

  // only allocated if data.coefficient_is_variable
  VariableCoefficients<scalar> coefficients;
};

} // namespace Operators

template<int rank, int dim>
struct LaplaceOperatorData : public OperatorBaseData
{
  LaplaceOperatorData() : OperatorBaseData(), quad_index_gauss_lobatto(0)
  {
  }

  Operators::LaplaceKernelData kernel_data;

  // continuous FE:
  // for DirichletCached boundary conditions, another quadrature rule
  // is needed to set the constrained DoFs.
  unsigned int quad_index_gauss_lobatto;

  std::shared_ptr<BoundaryDescriptor<rank, dim> const> bc;
};

template<int dim, typename Number, int n_components>
class LaplaceOperator : public OperatorBase<dim, Number, n_components>
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef OperatorBase<dim, Number, n_components>    Base;
  typedef LaplaceOperator<dim, Number, n_components> This;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::Range Range;

  typedef dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> value;

  typedef typename Base::VectorType VectorType;

public:
  typedef Number value_type;

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             LaplaceOperatorData<rank, dim> const &    data,
             bool const                                assemble_matrix);

  LaplaceOperatorData<rank, dim> const &
  get_data() const
  {
    return operator_data;
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index);

  void
  update_penalty_parameter();

  /**
   * Fills the variable coefficient a(x) by evaluating @p function in every quadrature point.
   *
   * Requires LaplaceKernelData::coefficient_is_variable to be set. The matrix representation
   * is re-assembled afterwards if the operator is matrix-based, so callers may set a new
   * coefficient and immediately apply the operator.
   *
   * This is the entry point used to realize an affine decomposition: evaluating the operator
   * once per indicator function of a subdomain yields the affine components A_p of
   * A(mu) = sum_p exp(mu_p) A_p.
   */
  void
  set_coefficient(dealii::Function<dim> const & function);

  /**
   * Fills the variable coefficient from one value per active cell.
   *
   * The values are indexed by dealii::CellAccessor::active_cell_index(), which is the ordering
   * a cell-wise field naturally has and which is independent of any geometric structure. That
   * matters once the mesh is unstructured: a coefficient described by a Function has to be
   * evaluated at quadrature points and so needs a rule mapping position to value, whereas a
   * genuinely piecewise constant field on the cells has no such rule and should not be forced
   * to invent one.
   *
   * The coefficient is constant within each cell, so every quadrature point of a cell receives
   * the same value; no quadrature error is introduced by the coefficient itself.
   */
  void
  set_coefficient_from_cell_values(std::vector<Number> const & cell_values);

  /**
   * Fills the variable coefficient by evaluating a finite element field at the quadrature points.
   *
   * The general form. The coefficient is expanded as ``a = sum_i c_i phi_i`` in a space of its
   * own, and @p coefficient_values holds the ``c_i``; @p coefficient_dof_index says which of the
   * matrix-free object's DoFHandlers that space is.
   *
   * Written for any degree. The evaluation reads the coefficient at *this operator's* quadrature
   * points, not the coefficient space's, which is the thing to be careful about: using the wrong
   * quadrature index gives a smooth, plausible and wrong operator. The two spaces need not share
   * a degree, and the coefficient's is free to be lower -- which is the usual case, since the
   * parameter dimension should not be dictated by the solution's resolution.
   *
   * The affine decomposition survives at any degree: quadrature is linear in the coefficient, so
   * ``A(c) = sum_i c_i A[phi_i]`` holds identically at the discrete level whatever rule is used.
   * Under-integrating changes which continuous operator is being discretised; it does not break
   * the expansion the reduced model is built on.
   */
  void
  set_coefficient_from_dof_vector(VectorType const & coefficient_values,
                                  unsigned int const coefficient_dof_index,
                                  bool const         check_positivity = true);

  /**
   * Assembles ``g_i = p^T A_i u`` for *every* coefficient degree of freedom in one cell loop.
   *
   * The other half of an adjoint. With the coefficient expanded as ``a = sum_i c_i phi_i`` the
   * operator is affine, ``A(c) = sum_i c_i A_i``, and differentiating ``A(c) u = f`` gives
   *
   *     d(p . B u)/dc_i = -p^T A_i u,        A^T p = B^T (dJ/dy).
   *
   * Because ``A_i`` is the operator with a *single basis function* as its coefficient,
   *
   *     p^T A_i u = integral( phi_i grad(u) . grad(p) ),
   *
   * which is a load vector in the coefficient space with the integrand grad(u).grad(p). So the
   * whole gradient -- all P entries -- is one matrix-free cell loop, not P operator applies.
   * That is what makes the adjoint route cost ``d`` solves rather than ``d + P`` anything, and
   * it is the same structure for any affinely parameterised coefficient.
   *
   * @p u and @p p are read with the operator's own constraints, exactly as vmult() reads its
   * argument, so the result is ``p^T A_i u`` for the operator as implemented rather than for an
   * idealisation of it. With inhomogeneous Dirichlet data the lifting term is *not* included
   * here -- the constrained read zeroes it -- and has to be added by the caller.
   *
   * @param u Solution of the forward problem.
   * @param p Solution of the adjoint problem, zero on constrained degrees of freedom.
   * @param coefficient_dof_index The coefficient space, as in set_coefficient_from_dof_vector().
   * @param dst Overwritten with the sensitivities, one per coefficient degree of freedom.
   */
  void
  compute_coefficient_sensitivity(VectorType const & u,
                                  VectorType const & p,
                                  unsigned int const coefficient_dof_index,
                                  VectorType &       dst) const;

  // continuous FE: This function sets the inhomogeneous Dirichlet boundary values for Dirichlet
  // degrees of freedom and optionally enforces hanging node and periodicity constraints.
  void
  set_inhomogeneous_constrained_values(VectorType & solution) const final;

  // only relevant for discontinuous Galerkin discretization (DG):
  // Some more functionality on top of what is provided by the base class.
  // This function evaluates the inhomogeneous boundary face integrals in DG where the
  // Dirichlet boundary condition is extracted from a dof vector instead of a dealii::Function<dim>.
  void
  rhs_add_dirichlet_bc_from_dof_vector(VectorType & dst, VectorType const & src) const;

private:
  void
  reinit_face_derived(IntegratorFace &   integrator_m,
                      IntegratorFace &   integrator_p,
                      unsigned int const face) const final;

  void
  reinit_boundary_face_derived(IntegratorFace & integrator_m, unsigned int const face) const final;

  void
  reinit_face_cell_based_derived(IntegratorFace &                 integrator_m,
                                 IntegratorFace &                 integrator_p,
                                 unsigned int const               cell,
                                 unsigned int const               face,
                                 dealii::types::boundary_id const boundary_id) const final;

  void
  do_cell_integral(IntegratorCell & integrator) const final;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  // only relevant for discontinuous Galerkin discretization (DG)
  void
  boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           range) const;

  // only relevant for discontinuous Galerkin discretization (DG)
  void
  do_boundary_integral_dirichlet_bc_from_dof_vector(
    IntegratorFace &                   integrator_m,
    OperatorType const &               operator_type,
    dealii::types::boundary_id const & boundary_id) const;

  // continuous FE: calculates Neumann boundary integral
  void
  do_boundary_integral_continuous(IntegratorFace &                   integrator_m,
                                  OperatorType const &               operator_type,
                                  dealii::types::boundary_id const & boundary_id) const final;

  LaplaceOperatorData<rank, dim> operator_data;

  Operators::LaplaceKernel<dim, Number, n_components> kernel;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* EXADG_POISSON_SPATIAL_DISCRETIZATION_LAPLACE_OPERATOR_H_ */
