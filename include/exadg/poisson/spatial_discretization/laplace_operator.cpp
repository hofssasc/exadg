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

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/poisson/spatial_discretization/weak_boundary_conditions.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  LaplaceOperatorData<rank, dim> const &    data,
  bool const                                assemble_matrix)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  AssertThrow(not(data.kernel_data.coefficient_is_variable and this->is_dg),
              dealii::ExcMessage(
                "A variable coefficient is currently only supported for continuous elements. "
                "For DG the coefficient additionally has to enter the gradient and value fluxes "
                "and scale the interior penalty parameter, which is not implemented yet."));

  kernel.reinit(matrix_free, data.kernel_data, data.dof_index, data.quad_index);

  this->integrator_flags = kernel.get_integrator_flags(this->is_dg);

  if(assemble_matrix)
    this->assemble_matrix_if_matrix_based();
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::calculate_penalty_parameter(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  unsigned int const                      dof_index)
{
  kernel.calculate_penalty_parameter(matrix_free, dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::update_penalty_parameter()
{
  calculate_penalty_parameter(this->get_matrix_free(), this->get_data().dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::set_coefficient(dealii::Function<dim> const & function)
{
  AssertThrow(kernel.coefficient_is_variable(),
              dealii::ExcMessage("Set LaplaceKernelData::coefficient_is_variable to true before "
                                 "prescribing a spatially varying coefficient."));

  dealii::MatrixFree<dim, Number> const & matrix_free = this->get_matrix_free();

  IntegratorCell integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
  {
    integrator.reinit(cell);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // FunctionEvaluator unpacks the vectorized quadrature point and evaluates the function
      // separately for every SIMD lane.
      dealii::VectorizedArray<Number> const coefficient =
        FunctionEvaluator<0, dim, Number>::value(function, integrator.quadrature_point(q));

      kernel.set_coefficient_cell(cell, q, coefficient);
    }
  }

  // keep a matrix-based representation, if any, consistent with the new coefficient
  this->assemble_matrix_if_matrix_based();
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::set_coefficient_from_cell_values(
  std::vector<Number> const & cell_values)
{
  AssertThrow(kernel.coefficient_is_variable(),
              dealii::ExcMessage("Set LaplaceKernelData::coefficient_is_variable to true before "
                                 "prescribing a spatially varying coefficient."));

  dealii::MatrixFree<dim, Number> const & matrix_free = this->get_matrix_free();

  IntegratorCell integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
  {
    integrator.reinit(cell);

    // Matrix-free groups cells into vectorized batches, so the value of each SIMD lane has to
    // be looked up separately; the lanes of a batch are unrelated cells.
    dealii::VectorizedArray<Number> coefficient = dealii::make_vectorized_array<Number>(0.0);

    unsigned int const n_lanes =
      matrix_free.n_active_entries_per_cell_batch(cell);

    for(unsigned int lane = 0; lane < n_lanes; ++lane)
    {
      auto const cell_iterator =
        matrix_free.get_cell_iterator(cell, lane, operator_data.dof_index);

      unsigned int const index = cell_iterator->active_cell_index();

      AssertThrow(index < cell_values.size(),
                  dealii::ExcMessage("Expected one coefficient per active cell, but cell " +
                                     std::to_string(index) + " has no value among the " +
                                     std::to_string(cell_values.size()) + " given."));

      coefficient[lane] = cell_values[index];
    }

    // padding lanes of an incomplete batch are never read, but leaving them at zero would make
    // a division by the coefficient elsewhere a surprise; the first lane's value is harmless
    for(unsigned int lane = n_lanes; lane < dealii::VectorizedArray<Number>::size(); ++lane)
      coefficient[lane] = coefficient[0];

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      kernel.set_coefficient_cell(cell, q, coefficient);
  }

  this->assemble_matrix_if_matrix_based();
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::set_coefficient_from_dof_vector(
  VectorType const & coefficient_values,
  unsigned int const coefficient_dof_index,
  bool const         check_positivity)
{
  AssertThrow(kernel.coefficient_is_variable(),
              dealii::ExcMessage("Set LaplaceKernelData::coefficient_is_variable to true before "
                                 "prescribing a spatially varying coefficient."));

  dealii::MatrixFree<dim, Number> const & matrix_free = this->get_matrix_free();

  // A scalar integrator over the *coefficient's* DoFHandler, but this operator's quadrature: the
  // coefficient has to be sampled where the operator integrates, not where it is defined.
  CellIntegrator<dim, 1, Number> coefficient_integrator(matrix_free,
                                                        coefficient_dof_index,
                                                        operator_data.quad_index);

  coefficient_values.update_ghost_values();

  for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
  {
    coefficient_integrator.reinit(cell);
    coefficient_integrator.read_dof_values_plain(coefficient_values);
    coefficient_integrator.evaluate(dealii::EvaluationFlags::values);

    for(unsigned int q = 0; q < coefficient_integrator.n_q_points; ++q)
    {
      dealii::VectorizedArray<Number> const coefficient = coefficient_integrator.get_value(q);

      // A coefficient that is non-positive anywhere destroys coercivity, and it does so
      // invisibly: the expansion coefficients may all be positive while the field between them
      // is not. That is exactly what a Lagrange basis above degree 1 does, which is why
      // Parameters::check() bounds the degree -- but a caller can also simply pass a bad vector.
      //
      // Only for an actual diffusivity, though. Building an affine component means installing a
      // *single basis function*, which is zero outside its support and changes sign above degree
      // one; that component operator is not meant to be coercive, and only the assembled
      // A(c) = sum_i c_i A_i has to be.
      if(check_positivity)
        for(unsigned int lane = 0; lane < dealii::VectorizedArray<Number>::size(); ++lane)
          AssertThrow(coefficient[lane] > 0.0,
                      dealii::ExcMessage(
                        "The coefficient is " + std::to_string(coefficient[lane]) +
                        " at a quadrature point, so the operator is not coercive there. With a "
                        "nodal expansion the field can dip below zero between nodes even when "
                        "every expansion coefficient is positive."));

      kernel.set_coefficient_cell(cell, q, coefficient);
    }
  }

  coefficient_values.zero_out_ghost_values();

  this->assemble_matrix_if_matrix_based();
}

namespace
{
/**
 * Contracts two gradients into the scalar the coefficient multiplies.
 *
 * Two overloads rather than one, because a gradient is a vector for a scalar field and a matrix
 * for a vector-valued one, and ``operator*`` on the rank-2 case is a matrix product rather than
 * the contraction wanted here -- an error that compiles.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE dealii::VectorizedArray<Number>
contract(dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & left,
         dealii::Tensor<1, dim, dealii::VectorizedArray<Number>> const & right)
{
  return left * right;
}

template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE dealii::VectorizedArray<Number>
contract(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & left,
         dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & right)
{
  return dealii::scalar_product(left, right);
}
} // namespace

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::compute_coefficient_sensitivity(
  VectorType const & u,
  VectorType const & p,
  unsigned int const coefficient_dof_index,
  VectorType &       dst) const
{
  AssertThrow(kernel.coefficient_is_variable(),
              dealii::ExcMessage("The coefficient sensitivity is the derivative with respect to "
                                 "the coefficient's degrees of freedom, of which there are none "
                                 "unless LaplaceKernelData::coefficient_is_variable is set."));

  dealii::MatrixFree<dim, Number> const & matrix_free = this->get_matrix_free();

  CellIntegrator<dim, n_components, Number> state(matrix_free,
                                                  operator_data.dof_index,
                                                  operator_data.quad_index);
  CellIntegrator<dim, n_components, Number> adjoint(matrix_free,
                                                    operator_data.dof_index,
                                                    operator_data.quad_index);

  // The test function is the *coefficient's*, sampled at the operator's quadrature points. Both
  // integrators must therefore share the quadrature index and need not share the DoF index --
  // which is the whole point, since the coefficient's space is coarser than the state's.
  CellIntegrator<dim, 1, Number> sensitivity(matrix_free,
                                             coefficient_dof_index,
                                             operator_data.quad_index);

  dst = 0.0;

  u.update_ghost_values();
  p.update_ghost_values();

  for(unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
  {
    state.reinit(cell);
    state.read_dof_values(u);
    state.evaluate(dealii::EvaluationFlags::gradients);

    adjoint.reinit(cell);
    adjoint.read_dof_values(p);
    adjoint.evaluate(dealii::EvaluationFlags::gradients);

    sensitivity.reinit(cell);

    for(unsigned int q = 0; q < sensitivity.n_q_points; ++q)
      sensitivity.submit_value(contract(state.get_gradient(q), adjoint.get_gradient(q)), q);

    sensitivity.integrate(dealii::EvaluationFlags::values);
    sensitivity.distribute_local_to_global(dst);
  }

  u.zero_out_ghost_values();
  p.zero_out_ghost_values();

  dst.compress(dealii::VectorOperation::add);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::rhs_add_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & src) const
{
  AssertThrow(this->is_dg, dealii::ExcMessage("This function is only implemented for DG."));

  VectorType tmp;
  tmp.reinit(dst, false /* init with 0 */);

  this->matrix_free->loop(&This::cell_loop_empty,
                          &This::face_loop_empty,
                          &This::boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector,
                          this,
                          tmp,
                          src,
                          false /* `zero_dst_vector` */);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right-hand side
  dst.add(-1.0, tmp);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_face_derived(IntegratorFace &   integrator_m,
                                                                IntegratorFace &   integrator_p,
                                                                unsigned int const face) const
{
  (void)face;

  kernel.reinit_face(integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_boundary_face_derived(
  IntegratorFace &   integrator_m,
  unsigned int const face) const
{
  (void)face;

  kernel.reinit_boundary_face(integrator_m, operator_data.dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_face_cell_based_derived(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  (void)cell;
  (void)face;

  kernel.reinit_face_cell_based(boundary_id, integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  unsigned int const cell = integrator.get_current_cell_index();

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_gradient(kernel.get_coefficient_cell(cell, q) * integrator.get_gradient(q),
                               q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_integral(IntegratorFace & integrator_m,
                                                             IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m = integrator_m.get_value(q);
    value value_p = integrator_p.get_value(q);

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m = integrator_m.get_normal_derivative(q);
    value normal_gradient_p = integrator_p.get_normal_derivative(q);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_p.submit_normal_derivative(gradient_flux, q);

    integrator_m.submit_value(-value_flux, q);
    integrator_p.submit_value(value_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set exterior value to zero
    value value_m = integrator_m.get_value(q);
    value value_p = value();

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    // set exterior value to zero
    value normal_gradient_m = integrator_m.get_normal_derivative(q);
    value normal_gradient_p = value();

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    value value_p = integrator_p.get_value(q);
    value value_m = value();

    value gradient_flux = kernel.calculate_gradient_flux(value_p, value_m);

    // minus sign to get the correct normal vector n⁺ = -n⁻
    value normal_gradient_p = -integrator_p.get_normal_derivative(q);
    value normal_gradient_m = value();

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_p, normal_gradient_m, value_p, value_m);

    integrator_p.submit_normal_derivative(-gradient_flux, q); // opposite sign since n⁺ = -n⁻
    integrator_p.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m =
      calculate_interior_value<dim, Number, n_components, rank>(q, integrator_m, operator_type);
    value value_p = calculate_exterior_value<dim, Number, n_components, rank>(value_m,
                                                                              q,
                                                                              integrator_m,
                                                                              operator_type,
                                                                              boundary_type,
                                                                              boundary_id,
                                                                              operator_data.bc,
                                                                              this->time);

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m =
      calculate_interior_normal_gradient<dim, Number, n_components, rank>(q,
                                                                          integrator_m,
                                                                          operator_type);
    value normal_gradient_p =
      calculate_exterior_normal_gradient<dim, Number, n_components, rank>(normal_gradient_m,
                                                                          q,
                                                                          integrator_m,
                                                                          operator_type,
                                                                          boundary_type,
                                                                          boundary_id,
                                                                          operator_data.bc,
                                                                          this->time);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::cell_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::face_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::
  boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           range) const
{
  IntegratorFace integrator_m =
    IntegratorFace(*this->matrix_free, true, operator_data.dof_index, operator_data.quad_index);

  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_m, face);

    // deviating from the standard function boundary_face_loop_inhom_operator()
    // because the boundary condition comes from the vector src
    integrator_m.gather_evaluate(src, this->integrator_flags.face_evaluate);

    do_boundary_integral_dirichlet_bc_from_dof_vector(integrator_m,
                                                      OperatorType::inhomogeneous,
                                                      matrix_free.get_boundary_id(face));

    integrator_m.integrate_scatter(this->integrator_flags.face_integrate, dst);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral_dirichlet_bc_from_dof_vector(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m =
      calculate_interior_value<dim, Number, n_components, rank>(q, integrator_m, operator_type);

    // deviating from the standard boundary_face_loop_inhom_operator() function,
    // because the boundary condition comes from the vector src
    Assert(operator_type == OperatorType::inhomogeneous,
           dealii::ExcMessage(
             "This function is only implemented for OperatorType::inhomogeneous."));

    value value_p = value();
    if(boundary_type == BoundaryType::Dirichlet)
    {
      // The desired boundary value g is obtained as integrator_m.get_value(q).
      value_p = 2.0 * integrator_m.get_value(q);
    }
    else if(boundary_type == BoundaryType::Neumann)
    {
      // do nothing
    }
    else
    {
      AssertThrow(false,
                  dealii::ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m =
      calculate_interior_normal_gradient<dim, Number, n_components, rank>(q,
                                                                          integrator_m,
                                                                          operator_type);
    value normal_gradient_p =
      calculate_exterior_normal_gradient<dim, Number, n_components, rank>(normal_gradient_m,
                                                                          q,
                                                                          integrator_m,
                                                                          operator_type,
                                                                          boundary_type,
                                                                          boundary_id,
                                                                          operator_data.bc,
                                                                          this->time);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator_m,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  if(operator_type == OperatorType::inhomogeneous or operator_type == OperatorType::full)
  {
    BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      value neumann_value = calculate_neumann_value<dim, Number, n_components, rank>(
        q, integrator_m, boundary_type, boundary_id, operator_data.bc, this->time);

      integrator_m.submit_value(-neumann_value, q);
    }
  }
  else if(operator_type == OperatorType::homogeneous)
  {
    // `OperatorType::homogeneous` does not involve boundary integrals. We still need to submit
    // zeroes for consistency.
    value zero;
    zero = 0.0;
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      integrator_m.submit_value(zero, q);
    }
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("OperatorType not supported in "
                                   "LaplaceOperator::do_boundary_integral_continuous()"));
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::set_inhomogeneous_constrained_values(
  VectorType & dst) const
{
  // periodicity and hanging node constraints are enforced strongly for continuous spaces
  if(not this->is_dg)
  {
    unsigned int const dof_index_inhomogeneous = this->get_dof_index_inhomogeneous();
    dealii::AffineConstraints<Number> const & constraints =
      this->matrix_free->get_affine_constraints(dof_index_inhomogeneous);
    constraints.distribute(dst);
  }

  // standard Dirichlet boundary conditions
  std::map<dealii::types::global_dof_index, double> boundary_values;
  for(auto dbc : operator_data.bc->dirichlet_bc)
  {
    dbc.second->set_time(this->get_time());

    dealii::ComponentMask mask     = dealii::ComponentMask();
    auto                  dbc_mask = operator_data.bc->dirichlet_bc_component_mask.find(dbc.first);
    if(dbc_mask != operator_data.bc->dirichlet_bc_component_mask.end())
      mask = dbc_mask->second;

    dealii::VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                                     this->matrix_free->get_dof_handler(
                                                       operator_data.dof_index),
                                                     dbc.first,
                                                     *dbc.second,
                                                     boundary_values,
                                                     mask);
  }

  // set Dirichlet values in solution vector
  for(auto m : boundary_values)
    if(dst.get_partitioner()->in_local_range(m.first))
      dst[m.first] = m.second;

  dst.update_ghost_values();

  // DirichletCached type boundary conditions
  if(not(operator_data.bc->dirichlet_cached_bc.empty()))
  {
    unsigned int const dof_index  = operator_data.dof_index;
    unsigned int const quad_index = operator_data.quad_index_gauss_lobatto;

    IntegratorFace integrator(*this->matrix_free, true, dof_index, quad_index);

    for(unsigned int face = this->matrix_free->n_inner_face_batches();
        face <
        this->matrix_free->n_inner_face_batches() + this->matrix_free->n_boundary_face_batches();
        ++face)
    {
      dealii::types::boundary_id const boundary_id = this->matrix_free->get_boundary_id(face);

      BoundaryType const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

      if(boundary_type == BoundaryType::DirichletCached)
      {
        integrator.reinit(face);
        integrator.read_dof_values(dst);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          unsigned int const local_face_number =
            this->matrix_free->get_face_info(face).interior_face_no;

          unsigned int const index = this->matrix_free->get_shape_info(dof_index, quad_index)
                                       .face_to_cell_index_nodal[local_face_number][q];

          dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> g;

          if(boundary_type == BoundaryType::DirichletCached)
          {
            auto bc = operator_data.bc->get_dirichlet_cached_data();

            g = FunctionEvaluator<rank, dim, Number>::value(*bc, face, q, quad_index);
          }
          else
          {
            AssertThrow(false, dealii::ExcMessage("Not implemented."));
          }

          integrator.submit_dof_value(g, index);
        }

        integrator.set_dof_values_plain(dst);
      }
      else
      {
        AssertThrow(boundary_type == BoundaryType::Dirichlet or
                      boundary_type == BoundaryType::Neumann,
                    dealii::ExcMessage("BoundaryType not implemented."));
      }
    }
  }
}

template class LaplaceOperator<2, float, 1>;
template class LaplaceOperator<2, double, 1>;
template class LaplaceOperator<2, float, 2>;
template class LaplaceOperator<2, double, 2>;

template class LaplaceOperator<3, float, 1>;
template class LaplaceOperator<3, double, 1>;
template class LaplaceOperator<3, float, 3>;
template class LaplaceOperator<3, double, 3>;

} // namespace Poisson
} // namespace ExaDG
