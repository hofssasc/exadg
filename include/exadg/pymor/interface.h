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
/**
 * The vocabulary a full-order model has to speak to be driven by pyMOR.
 *
 * Everything in this file is physics-free on purpose. An application implements these classes;
 * python/exadg/mor/binding.py wraps them in pyMOR's interfaces without knowing what they
 * discretise, and python/exadg/mor/models/ assembles them into a pyMOR Model. Adding an
 * application therefore means writing C++ and nothing else.
 *
 * Two rules make that possible, and they are the reason for the shape of what follows.
 *
 * **Nothing is assumed; capabilities are declared.** Symmetry, invertibility, the existence of
 * a restricted evaluation, the presence of Dirichlet constraints -- each is a virtual method
 * whose default says "not available". A default that guesses would be silently wrong in exactly
 * the cases nobody checks, so the defaults abort or return nullptr instead. Whoever writes the
 * operator knows the answer; whoever writes the Python layer does not.
 *
 * **Structure is declared here, naming is not.** parameter_shape() says how many parameters
 * there are and how they group; it does not say what they are called. Names are a modelling
 * choice that belongs in Python, where changing one does not mean a recompile.
 */

/**
 * An operator evaluated on a small set of output degrees of freedom.
 *
 * This is what empirical interpolation needs, and pyMOR's contract on it is
 *
 *     A.apply(U).dofs(output_dofs) == restricted.apply(U.dofs(source_dofs))
 *
 * exactly, not approximately. The source set is the stencil: every degree of freedom sharing a
 * cell with a requested one. Resolving it is a question about the mesh, which is why it is
 * answered here and not in Python -- answering it there would mean exporting a full-order sized
 * sparsity pattern, which is what this whole interface exists to avoid.
 */
class RestrictedOperator
{
public:
  virtual ~RestrictedOperator() = default;

  /**
   * The stencil: degrees of freedom whose values apply() needs, in the order it expects them.
   */
  virtual std::vector<dealii::types::global_dof_index> const &
  get_source_dofs() const = 0;

  /**
   * Rows of the operator at the output degrees of freedom, given the stencil values.
   */
  virtual std::vector<double>
  apply(std::vector<double> const & source_values) const = 0;

  /**
   * Which affine components the stencil actually reads, or empty if the notion does not apply.
   *
   * The sparsity hyper-reduction is trading on: an interpolated operator is *exactly*
   * insensitive to every component outside this set. For a forward solve that is the intended
   * approximation; for a parameter Jacobian it is a claim that the others do not matter, which
   * is worth measuring rather than assuming.
   */
  virtual std::vector<unsigned int>
  active_components() const
  {
    return {};
  }

  /**
   * Installs the coefficients of a parametric operator's affine components.
   *
   * Only called for a restriction obtained from a ParametricOperator, whose coefficient is not
   * known until apply time. A restriction of a fixed operator already carries its coefficients
   * and aborts here rather than silently ignoring an argument that was meant to change the
   * answer.
   */
  virtual void
  set_coefficients(std::vector<double> const &)
  {
    AssertThrow(false,
                dealii::ExcMessage("This restricted operator has fixed coefficients. Only a "
                                   "restriction of a ParametricOperator can be re-parameterised."));
  }
};

/**
 * A linear operator on the state space, held by handle.
 *
 * apply() is the only method pyMOR always needs. Each of the others unlocks a capability and
 * costs nothing to leave alone:
 *
 *     apply_transpose            output error estimators, dual-weighted residuals
 *     apply_inverse              full-order solves, greedy basis generation, LSPG, certification
 *     apply_inverse_transpose    adjoint solves
 *     restricted                 empirical interpolation, and hyper-reduction after it
 *
 * The two transposes default to forwarding to their non-transposed counterpart *and abort
 * unless is_symmetric() says that is legitimate*. So a symmetric operator writes nothing extra,
 * a non-symmetric one is told to implement the transpose, and neither gets a wrong answer. This
 * matters at the first non-symmetric operator -- a Navier-Stokes momentum block -- where the
 * old assumption would have produced plausible numbers rather than a failure.
 */
template<typename VectorType>
class LinearOperator
{
public:
  virtual ~LinearOperator() = default;

  /**
   * dst = A src.
   */
  virtual void
  apply(VectorType & dst, VectorType const & src) const = 0;

  /**
   * Whether A == A^T. Opt-in: the default claims nothing.
   */
  virtual bool
  is_symmetric() const
  {
    return false;
  }

  /**
   * dst = A^T src.
   */
  virtual void
  apply_transpose(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(is_symmetric(),
                dealii::ExcMessage("apply_transpose() is not implemented for this operator and "
                                   "it does not declare is_symmetric()."));

    apply(dst, src);
  }

  /**
   * Whether apply_inverse() will do anything. Opt-in, like is_symmetric().
   *
   * Declared rather than probed because probing costs a solve, and because pyMOR decides *once*,
   * when the operator is built, whether to route inversions through this application's solver or
   * through its own generic path.
   */
  virtual bool
  has_inverse() const
  {
    return false;
  }

  /**
   * Solves A x = rhs, or returns nullptr if this operator has no solver.
   *
   * nullptr is a legitimate answer and pyMOR handles it: the operator is then one it can
   * multiply but not invert, and an algorithm needing a solve reports an InversionError rather
   * than falling back to converting the operator into a dense matrix -- which cannot work for a
   * vector type whose entries never enter Python.
   */
  virtual std::shared_ptr<VectorType>
  apply_inverse(VectorType const & /*rhs*/) const
  {
    return nullptr;
  }

  /**
   * Solves A^T x = rhs.
   */
  virtual std::shared_ptr<VectorType>
  apply_inverse_transpose(VectorType const & rhs) const
  {
    AssertThrow(is_symmetric(),
                dealii::ExcMessage("apply_inverse_transpose() is not implemented for this "
                                   "operator and it does not declare is_symmetric()."));

    return apply_inverse(rhs);
  }

  /**
   * The operator restricted to these output degrees of freedom, or nullptr if unsupported.
   */
  virtual std::shared_ptr<RestrictedOperator>
  restricted(std::vector<dealii::types::global_dof_index> const & /*output_dofs*/) const
  {
    return nullptr;
  }

  /**
   * A label for diagnostics and for pyMOR's operator names.
   */
  virtual std::string
  get_name() const
  {
    return "A";
  }
};

/**
 * An operator whose coefficients arrive at apply time rather than at construction.
 *
 * The alternative presentation of a parametric operator. Given as a sum of affine components,
 * pyMOR sees the full parametric structure and projects each component exactly -- at a cost of
 * one full-order apply per component per basis vector, which is linear in a parameter dimension
 * that may grow with the mesh. Given as one of these instead, the parameter dependence is opaque
 * and pyMOR reaches for empirical interpolation, whose cost is set by the interpolation rather
 * than by the mesh. An application may offer either, both, or neither.
 */
template<typename VectorType>
class ParametricOperator : public LinearOperator<VectorType>
{
public:
  /**
   * Installs the coefficients used by every subsequent apply().
   *
   * Indexed exactly like FullOrderModel::operator_components(), so a caller that can build the
   * affine form can drive this one with the same vector and get the same operator.
   */
  virtual void
  set_coefficients(std::vector<double> const & coefficients) = 0;

  /**
   * Which entry of parameter_shape() the coefficients belong to.
   */
  virtual unsigned int
  parameter_slot() const
  {
    return 0;
  }
};

/**
 * A linear map from the state space to a fixed number of scalar outputs.
 *
 * Sensor readings, a drag coefficient, an average over a subdomain -- whatever the quantity of
 * interest is. Used as a pyMOR Model's output functional, so that its projection onto the
 * reduced basis comes out of the reductor along with the operators and the reduced model can
 * predict outputs without ever reconstructing a full field.
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

  /**
   * B^T w, mapping output weights back to the state space.
   *
   * pyMOR needs this as apply_adjoint: the output error estimator forms the Riesz representative
   * of B^T and fails without it.
   */
  virtual std::shared_ptr<VectorType>
  apply_transpose(std::vector<double> const & weights) const = 0;
};

/**
 * One affine component together with the parameter entry its coefficient comes from.
 *
 * A(mu) = sum_i c(mu, slot_i, index_i) * A_i, where the coefficient function c is chosen in
 * Python. slot selects an entry of parameter_shape() and index a position within it, so an
 * application with a scalar viscosity and sixteen forcing amplitudes declares shape {1, 16} and
 * tags its components (0, 0) and (1, 0..15).
 */
template<typename VectorType>
struct AffineComponent
{
  std::shared_ptr<LinearOperator<VectorType>> op;

  unsigned int slot  = 0;
  unsigned int index = 0;
};

/**
 * One affine component of a parameter-dependent right-hand side.
 */
template<typename VectorType>
struct AffineVector
{
  std::shared_ptr<VectorType> vector;

  unsigned int slot  = 0;
  unsigned int index = 0;
};

/**
 * The full-order model an application exposes to pyMOR.
 *
 * Implement this, bind the concrete class with pybind11, and the Python layer builds a pyMOR
 * Model from it without further help. Pure virtual methods are the ones no application can
 * avoid; everything else declares a capability and defaults to not having it.
 *
 * The methods handing out operators are deliberately non-const. An operator returned here holds
 * the model and will mutate it -- installing a coefficient, rebuilding a preconditioner -- so
 * producing one is not a const operation, and saying so beats a const method that hands out a
 * mutable reference to itself.
 */
template<typename VectorType>
class FullOrderModel : public std::enable_shared_from_this<FullOrderModel<VectorType>>
{
public:
  virtual ~FullOrderModel() = default;

  // --- the state space ---------------------------------------------------------------------

  /**
   * Global number of degrees of freedom, summed over ranks.
   */
  virtual dealii::types::global_dof_index
  n_dofs() const = 0;

  /**
   * A zero vector with this model's partitioning and ghosting.
   */
  virtual std::shared_ptr<VectorType>
  zero_vector() const = 0;

  /**
   * Projects a vector onto the subspace the operators are valid on.
   *
   * Called on every vector the Python layer creates that did not come out of a solve -- random
   * probes, empirical interpolation candidates, data read from NumPy. A discretisation with
   * eliminated Dirichlet rows zeroes them here, because an affine decomposition holds only on
   * that subspace: on a constrained row every component acts as the identity, so summing P of
   * them scales the entry by the sum of the coefficients rather than leaving it alone.
   *
   * The default is a no-op, which is correct for a discretisation that constrains nothing.
   */
  virtual void
  make_admissible(VectorType & /*vector*/) const
  {
  }

  // --- structure ---------------------------------------------------------------------------

  /**
   * Sizes of the parameter groups, e.g. {16} for one field of sixteen values, or {1, 16} for a
   * scalar alongside a field. Empty for a model with no parameters.
   *
   * Deliberately shapes and not names: what the parameters are *called* is a modelling choice
   * and lives in Python, where renaming one does not mean a recompile.
   */
  virtual std::vector<unsigned int>
  parameter_shape() const
  {
    return {};
  }

  /**
   * Affine components of the operator. Empty if the model offers only a parametric operator.
   */
  virtual std::vector<AffineComponent<VectorType>>
  operator_components()
  {
    return {};
  }

  /**
   * The operator as a single parametric object, or nullptr.
   */
  virtual std::shared_ptr<ParametricOperator<VectorType>>
  parametric_operator()
  {
    return nullptr;
  }

  /**
   * The operator at fixed component coefficients, or nullptr if it cannot be assembled.
   *
   * What makes a reduced model solvable rather than merely multipliable. pyMOR collapses a
   * linear combination of the affine components at a parameter and asks for it here; returning
   * an operator carrying a solver is what routes full-order solves through this application's
   * own preconditioned Krylov method.
   *
   * Coefficients are indexed like operator_components() and may be **signed**: a reductor
   * legitimately forms differences of operators. Return nullptr for a combination this
   * application cannot represent -- a negative diffusivity, say -- and pyMOR falls back to its
   * generic path. Deciding that here is the point: whoever wrote the coefficient knows what it
   * means, and the Python layer does not.
   */
  virtual std::shared_ptr<LinearOperator<VectorType>>
  assemble(std::vector<double> const & /*coefficients*/)
  {
    return nullptr;
  }

  // --- right-hand side ---------------------------------------------------------------------

  /**
   * The parameter-independent right-hand side, or nullptr.
   */
  virtual std::shared_ptr<VectorType>
  rhs()
  {
    return nullptr;
  }

  /**
   * Affine components of a parameter-dependent right-hand side.
   *
   * Both this and rhs() may be present, in which case the right-hand side is their sum. An
   * application whose parameters live entirely in the forcing declares nothing in rhs() and
   * everything here.
   */
  virtual std::vector<AffineVector<VectorType>>
  rhs_components()
  {
    return {};
  }

  // --- optional extras ---------------------------------------------------------------------

  /**
   * Inner products, by name. "mass" is conventionally the L2 product and the one a proper
   * orthogonal decomposition should be taken in; "energy" the product a coercive error
   * estimator is certified in.
   */
  virtual std::map<std::string, std::shared_ptr<LinearOperator<VectorType>>>
  products()
  {
    return {};
  }

  /**
   * The output functional, or nullptr for a model with no quantity of interest.
   */
  virtual std::shared_ptr<Functional<VectorType>>
  output_functional()
  {
    return nullptr;
  }

  /**
   * Writes fields as a VTU/PVTU record and returns the path of the record written.
   *
   * A file writer rather than a plot window: the model may be on a compute node, and under MPI
   * each rank holds a piece of the field, so there is nothing for one process to draw.
   */
  virtual std::string
  write_vtu(std::string const & /*directory*/,
            std::string const & /*basename*/,
            std::vector<std::shared_ptr<VectorType>> const & /*fields*/,
            std::vector<std::string> const & /*names*/) const
  {
    AssertThrow(false, dealii::ExcMessage("This model does not implement write_vtu()."));

    return {};
  }
};

} // namespace PyMOR
} // namespace ExaDG

#endif /* EXADG_PYMOR_INTERFACE_H_ */
