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

#ifndef EXADG_PYMOR_INTERFACE_H_
#define EXADG_PYMOR_INTERFACE_H_

// C/C++
#include <map>
#include <memory>
#include <string>
#include <vector>

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>

namespace ExaDG
{
namespace PyMOR
{
/*
 * What an application must provide for pyMOR to reduce it.
 *
 * Implement these classes in applications/<app>/python_bindings.cpp and bind the concrete model
 * with pybind11. python/exadg/mor/ then builds a pyMOR model from what is declared here, so no
 * Python is written per application.
 *
 * The classes are thin renamings of deal.II's own vocabulary:
 *
 *   deal.II / ExaDG                    this file                        implement it when
 *   ---------------------------------------------------------------------------------------
 *   distributed::Vector                (bound in exadg._core)           never
 *   Operator::vmult                    LinearOperator::apply            always
 *   Operator::Tvmult                   LinearOperator::apply_transpose  A != A^T
 *   Krylov solve + preconditioner      LinearOperator::apply_inverse    you want full-order solves
 *   AffineConstraints::set_zero        Space::make_admissible           Dirichlet rows are eliminated
 *   a DoFHandler's function space      Space                            always
 *   MatrixFree + DoFHandler + Driver   FullOrderModel                   a single-field problem
 *   OperatorCoupled's block system     SaddlePointModel                 a velocity/pressure problem
 *   cell matrices from FEValues        RestrictedOperator               you want hyper-reduction
 *   a functional of the solution       Functional                       the model has outputs
 *
 * Two rules run through the file.
 *
 * **Nothing is assumed.** Symmetry, invertibility, a restricted evaluation, the presence of
 * constraints -- each is a virtual method whose default says "no", and whose transposed forms
 * abort rather than improvise. A wrong guess is invisible here: an adjoint solve that quietly
 * solves the wrong system returns plausible numbers.
 *
 * **Structure is declared, naming is not.** parameter_shape() says how many parameters there are
 * and how they group. What they are called, and how the operator depends on them, are modelling
 * choices and live in Python, where changing one is not a recompile.
 */

/**
 * An operator evaluated on a few output degrees of freedom, for empirical interpolation.
 *
 * The contract is an identity, not an approximation:
 *
 *     A.apply(U).dofs(output_dofs) == restricted.apply(U.dofs(source_dofs))
 *
 * The source set is the stencil: every degree of freedom sharing a cell with a requested one.
 * Resolving it is a question about the mesh, so it is answered here rather than by exporting a
 * full-order sized sparsity pattern to Python.
 */
class RestrictedOperator
{
public:
  virtual ~RestrictedOperator() = default;

  /// Stencil degrees of freedom, in the order apply() expects their values.
  virtual std::vector<dealii::types::global_dof_index> const &
  get_source_dofs() const = 0;

  /// Rows of the operator at the output degrees of freedom, given the stencil values.
  virtual std::vector<double>
  apply(std::vector<double> const & source_values) const = 0;

  /**
   * Affine components the stencil actually reads, or empty if the notion does not apply.
   *
   * An interpolated operator is exactly insensitive to every component outside this set, which
   * is what hyper-reduction trades on and what makes it worth measuring.
   */
  virtual std::vector<unsigned int>
  active_components() const
  {
    return {};
  }

  /// Re-parameterise. Only a restriction of a ParametricOperator may; the rest refuse.
  virtual void
  set_coefficients(std::vector<double> const &)
  {
    AssertThrow(false, dealii::ExcMessage("This restriction has fixed coefficients."));
  }
};

/**
 * A linear operator on the state space -- an ExaDG operator's vmult, held by handle.
 *
 * apply() is the only method pyMOR always needs. Each of the others unlocks a capability and
 * costs nothing to leave alone:
 *
 *     apply_transpose            output error estimators, dual-weighted residuals
 *     apply_inverse              full-order solves, greedy basis generation, LSPG, certification
 *     restricted                 empirical interpolation, and hyper-reduction after it
 */
template<typename VectorType>
class LinearOperator
{
public:
  virtual ~LinearOperator() = default;

  /// dst = A src.
  virtual void
  apply(VectorType & dst, VectorType const & src) const = 0;

  /// Declare A == A^T. Opt-in: the default claims nothing.
  virtual bool
  is_symmetric() const
  {
    return false;
  }

  /// Declare that apply_inverse() will do something. Opt-in, like is_symmetric().
  virtual bool
  has_inverse() const
  {
    return false;
  }

  /// dst = A^T src. Override unless the operator is symmetric.
  virtual void
  apply_transpose(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(is_symmetric(),
                dealii::ExcMessage("Override apply_transpose(), or declare is_symmetric()."));

    apply(dst, src);
  }

  /**
   * Solve A x = rhs, or return nullptr for an operator with no solver.
   *
   * nullptr is a legitimate answer: pyMOR then reports an InversionError rather than falling
   * back to building a dense matrix, which cannot work for a vector it never sees the entries of.
   */
  virtual std::shared_ptr<VectorType>
  apply_inverse(VectorType const & /*rhs*/) const
  {
    return nullptr;
  }

  /// Solve A^T x = rhs. Override unless the operator is symmetric.
  virtual std::shared_ptr<VectorType>
  apply_inverse_transpose(VectorType const & rhs) const
  {
    AssertThrow(is_symmetric(),
                dealii::ExcMessage(
                  "Override apply_inverse_transpose(), or declare is_symmetric()."));

    return apply_inverse(rhs);
  }

  /// The operator on these output degrees of freedom, or nullptr if it cannot be restricted.
  virtual std::shared_ptr<RestrictedOperator>
  restricted(std::vector<dealii::types::global_dof_index> const & /*output_dofs*/) const
  {
    return nullptr;
  }

  /// Label used for pyMOR's operator name and in diagnostics.
  virtual std::string
  get_name() const
  {
    return "A";
  }
};

/**
 * An operator whose coefficients arrive at apply time rather than at construction.
 *
 * The alternative to declaring affine components, and an application may offer either, both or
 * neither. As components, pyMOR projects each one exactly, at one full-order apply per component
 * per basis vector -- linear in a parameter count that grows with the mesh once the coefficient
 * is per cell. As one of these, the parameter dependence is opaque and pyMOR reaches for
 * empirical interpolation, whose cost is set by the interpolation instead.
 */
template<typename VectorType>
class ParametricOperator : public LinearOperator<VectorType>
{
public:
  /// Coefficients for every subsequent apply(), indexed like operator_components().
  virtual void
  set_coefficients(std::vector<double> const & coefficients) = 0;

  /// Which entry of parameter_shape() those coefficients belong to.
  virtual unsigned int
  parameter_slot() const
  {
    return 0;
  }
};

/**
 * A linear map from the state to a fixed number of scalar outputs: sensor readings, a drag
 * coefficient, an average over a subdomain.
 *
 * Used as the pyMOR model's output functional, so its projection onto the reduced basis comes out
 * of the reductor with the operators and the reduced model predicts outputs without ever
 * reconstructing a full field.
 */
template<typename VectorType>
class Functional
{
public:
  virtual ~Functional() = default;

  virtual unsigned int
  n_outputs() const = 0;

  virtual std::vector<double>
  apply(VectorType const & src) const = 0;

  /// B^T w. pyMOR's output error estimator forms the Riesz representative of B^T and needs this.
  virtual std::shared_ptr<VectorType>
  apply_transpose(std::vector<double> const & weights) const = 0;
};

/**
 * A discrete function space -- what one pyMOR VectorArray lives in.
 *
 * Split out of FullOrderModel because a saddle-point problem has two of them, velocity and
 * pressure, and the Python wrapper needs a space rather than a model.
 */
template<typename VectorType>
class Space
{
public:
  virtual ~Space() = default;

  /// Global number of degrees of freedom, summed over ranks.
  virtual dealii::types::global_dof_index
  n_dofs() const = 0;

  /// A zero vector with this space's partitioning and ghosting.
  virtual std::shared_ptr<VectorType>
  zero_vector() const = 0;

  /**
   * Project onto the subspace the operators are valid on.
   *
   * Called on every vector the Python layer builds that did not come out of a solve: random
   * probes, interpolation candidates, data read from NumPy. Eliminate Dirichlet rows here if the
   * discretisation has them, because an affine decomposition holds only on that subspace -- on a
   * constrained row every component acts as the identity, so summing P of them scales the entry
   * by the sum of the coefficients instead of leaving it alone.
   *
   * The default is a no-op, which is correct for a discretisation that constrains nothing -- a
   * discontinuous Galerkin velocity space, for instance, imposes its boundary conditions weakly.
   */
  virtual void
  make_admissible(VectorType & /*vector*/) const
  {
  }

  /**
   * Write fields as a VTU/PVTU record and return the record's path.
   *
   * A file writer rather than a plot window: the model may be on a compute node, and under MPI
   * each rank holds a piece of the field.
   */
  virtual std::string
  write_vtu(std::string const & /*directory*/,
            std::string const & /*basename*/,
            std::vector<std::shared_ptr<VectorType>> const & /*fields*/,
            std::vector<std::string> const & /*names*/) const
  {
    AssertThrow(false, dealii::ExcMessage("This space does not implement write_vtu()."));

    return {};
  }
};

/**
 * One affine component together with the parameter entry its coefficient comes from.
 *
 * A(mu) = sum_i c(mu, slot_i, index_i) A_i, with the coefficient function chosen in Python. slot
 * selects an entry of parameter_shape() and index a position within it, so a model with a scalar
 * viscosity and sixteen forcing amplitudes declares shape {1, 16} and tags its components (0, 0)
 * and (1, 0..15).
 */
template<typename VectorType>
struct AffineComponent
{
  std::shared_ptr<LinearOperator<VectorType>> op;

  unsigned int slot  = 0;
  unsigned int index = 0;
};

/// One affine component of a parameter-dependent right-hand side.
template<typename VectorType>
struct AffineVector
{
  std::shared_ptr<VectorType> vector;

  unsigned int slot  = 0;
  unsigned int index = 0;
};

/**
 * The model an application exposes: whatever owns the MatrixFree, the DoFHandler and the solver.
 *
 * Pure virtuals are what no application can avoid; everything else declares a capability and
 * defaults to not having it. The methods handing out operators are non-const because an operator
 * returned here holds the model and will mutate it -- installing a coefficient, rebuilding a
 * preconditioner.
 */
template<typename VectorType>
class FullOrderModel : public Space<VectorType>,
                       public std::enable_shared_from_this<FullOrderModel<VectorType>>
{
public:
  virtual ~FullOrderModel() = default;

  // --- structure ---------------------------------------------------------------------------

  /// Sizes of the parameter groups: {16} for one field, {1, 16} for a scalar beside a field.
  virtual std::vector<unsigned int>
  parameter_shape() const
  {
    return {};
  }

  /// Affine components of the operator. Empty if only a parametric operator is offered.
  virtual std::vector<AffineComponent<VectorType>>
  operator_components()
  {
    return {};
  }

  /// The operator as a single parametric object, or nullptr.
  virtual std::shared_ptr<ParametricOperator<VectorType>>
  parametric_operator()
  {
    return nullptr;
  }

  /**
   * The operator at fixed component coefficients, or nullptr if it cannot be assembled.
   *
   * What makes a model solvable rather than only multipliable: pyMOR collapses a linear
   * combination of the affine components at a parameter and asks for it here, and an operator
   * carrying a solver routes full-order solves through this application's own Krylov method.
   *
   * Coefficients are indexed like operator_components() and may be **signed** -- a reductor
   * legitimately forms differences of operators. Return nullptr for anything this application
   * cannot represent, such as a negative diffusivity, and pyMOR falls back to its generic path.
   */
  virtual std::shared_ptr<LinearOperator<VectorType>>
  assemble(std::vector<double> const & /*coefficients*/)
  {
    return nullptr;
  }

  // --- right-hand side ---------------------------------------------------------------------

  /// The parameter-independent right-hand side, or nullptr.
  virtual std::shared_ptr<VectorType>
  rhs()
  {
    return nullptr;
  }

  /**
   * Affine components of a parameter-dependent right-hand side.
   *
   * Both this and rhs() may be present, in which case the right-hand side is their sum. A model
   * whose parameters live entirely in the forcing declares nothing in rhs() and everything here.
   */
  virtual std::vector<AffineVector<VectorType>>
  rhs_components()
  {
    return {};
  }

  // --- optional extras ---------------------------------------------------------------------

  /**
   * Inner products by name. "mass" is conventionally the L2 product, and the one a proper
   * orthogonal decomposition should be taken in; "energy" the product a coercive error estimator
   * is certified in.
   */
  virtual std::map<std::string, std::shared_ptr<LinearOperator<VectorType>>>
  products()
  {
    return {};
  }

  /// The output functional, or nullptr for a model with no quantity of interest.
  virtual std::shared_ptr<Functional<VectorType>>
  output_functional()
  {
    return nullptr;
  }

};

/**
 * A velocity/pressure problem, as pyMOR's SaddlePointModel wants it.
 *
 *     [ A   B* ] [u]   [f]
 *     [ B   0  ] [p] = [g]
 *
 * pyMOR assembles that block operator itself from A and B, and forms the (1,2) block as
 * AdjointOperator(B) -- so B::apply_transpose has to reproduce ExaDG's (1,2) block exactly,
 * sign and scaling included. It is not the same operator as B, and the interface will not
 * pretend otherwise: divergence() must declare is_symmetric() false and implement the transpose.
 *
 * Two spaces rather than one block space, because that is what the reductor needs. Supremizer
 * enrichment computes velocity_product^-1 B^T p for each pressure basis vector, which is what
 * restores the inf-sup condition the reduced spaces would otherwise lose.
 *
 * A and B are not parametric here. Every parameter this interface carries lives in the
 * right-hand side, which is what a body force expanded in modes gives. Making the viscosity a
 * parameter as well would need a setter on ExaDG's viscous kernel -- it bakes the value in at
 * setup, together with the interior penalty parameter derived from it -- so it is left out until
 * it buys something.
 */
template<typename VectorType>
class SaddlePointModel : public std::enable_shared_from_this<SaddlePointModel<VectorType>>
{
public:
  virtual ~SaddlePointModel() = default;

  virtual std::shared_ptr<Space<VectorType>>
  velocity_space() = 0;

  virtual std::shared_ptr<Space<VectorType>>
  pressure_space() = 0;

  /// Sizes of the parameter groups; see FullOrderModel::parameter_shape.
  virtual std::vector<unsigned int>
  parameter_shape() const
  {
    return {};
  }

  /// A, the (1,1) block: velocity in, velocity out.
  virtual std::shared_ptr<LinearOperator<VectorType>>
  momentum() = 0;

  /// B, the (2,1) block: velocity in, pressure out. Its transpose is the (1,2) block.
  virtual std::shared_ptr<LinearOperator<VectorType>>
  divergence() = 0;

  /// The velocity inner product. Needed for supremizers, so effectively required.
  virtual std::shared_ptr<LinearOperator<VectorType>>
  velocity_product()
  {
    return nullptr;
  }

  virtual std::shared_ptr<LinearOperator<VectorType>>
  pressure_product()
  {
    return nullptr;
  }

  /// Parameter-independent part of the velocity right-hand side f, or nullptr.
  virtual std::shared_ptr<VectorType>
  velocity_rhs()
  {
    return nullptr;
  }

  /// Affine components of f.
  virtual std::vector<AffineVector<VectorType>>
  velocity_rhs_components()
  {
    return {};
  }

  /// The pressure right-hand side g, or nullptr for zero.
  virtual std::shared_ptr<VectorType>
  pressure_rhs()
  {
    return nullptr;
  }

  /**
   * The full-order coupled solve, for snapshots.
   *
   * Returns false if this model cannot solve at those coefficients, the way
   * FullOrderModel::assemble returns nullptr. Kept as a single call rather than composed from
   * the blocks because the block preconditioner is the application's business, and because for
   * Navier-Stokes this is where its Newton iteration lives.
   */
  virtual bool
  solve(std::vector<double> const & /*coefficients*/,
        VectorType & /*velocity*/,
        VectorType & /*pressure*/)
  {
    return false;
  }
};

} // namespace PyMOR
} // namespace ExaDG

#endif /* EXADG_PYMOR_INTERFACE_H_ */
