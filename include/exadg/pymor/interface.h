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
 *   one implicit time step             SaddlePointModel(mass_scaling)   the problem is transient
 *   cell matrices from FEValues        RestrictedOperator               you want empirical interpolation
 *   a nonlinear term, part polynomial  SplitOperator                    you want an exact tensor
 *   a residual summed over faces/cells SampledOperator                  you want ECSW
 *   -- (the arrays it compiles to)     CompiledOperator                 with SampledOperator
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
 * The evaluation half of a SampledOperator: arrays, and nothing else.
 *
 * Deliberately **not** templated on a vector type, because it has no use for one. Once the
 * weights are installed the sampled operator is a fixed quantity of data -- weights, quadrature
 * measures, normals, the basis traces on the entities that survived -- and evaluating it is
 * arithmetic on that data. It needs no mesh, no MatrixFree, no DoFHandler and no solver.
 *
 * Splitting it out is what makes a reduced model a deliverable rather than a view onto a resident
 * full-order one. Held on its own, it can be moved, kept after the model is destroyed, and (once
 * its arrays are gathered) evaluated on one rank with no communication at all.
 */
class CompiledOperator
{
public:
  virtual ~CompiledOperator() = default;

  /// Entities the installed weights visit.
  virtual std::size_t
  n_selected() const = 0;

  /// V^T sum_e w_e R_e(V a), length r.
  virtual std::vector<double>
  projected(std::vector<double> const & coefficients) const = 0;

  /// V^T (sum_e w_e R'_e(V a)) V, row-major (r, r).
  virtual std::vector<double>
  jacobian(std::vector<double> const & coefficients) const = 0;

  /**
   * Scale the inhomogeneous boundary data this operator holds, relative to how it was compiled.
   *
   * A sampled term that touches a Dirichlet boundary carries the prescribed data with it -- in a
   * Lax-Friedrichs flux it is inside the lambda, which is a maximum of absolute values and so not
   * a polynomial in anything. There is nothing to decompose, and the only way a detached operator
   * can follow a schedule is to be told where on it the caller is. Compiled at an amplitude of
   * one, so this is the amplitude itself and not a ratio.
   *
   * Defaults to doing nothing, which is right for fixed or homogeneous data.
   */
  virtual void
  set_boundary_amplitude(double const /*amplitude*/)
  {
  }
};

/**
 * An operator evaluated as a weighted sum over a few mesh entities, for hyper-reduction.
 *
 * The object a reduced model needs once its residual is too nonlinear to project exactly. An
 * application supplies its residual as a sum of per-entity contributions -- faces of a numerical
 * flux, cells of a nonlinear material law -- and a reductor fits weights that reproduce the whole
 * from a few of them. Which entities they are is the application's business; nothing in Python
 * needs to know.
 *
 * Everything here speaks **reduced coefficients**, not degrees of freedom. That is deliberate and
 * is what makes the online cost independent of the mesh: given a vector, an implementation would
 * have to reconstruct V a everywhere before looking at a dozen entities, and that reconstruction
 * would then be the dominant cost. Given coefficients, it can reconstruct on the sampled entities
 * alone.
 *
 * The evaluator owns its basis and weights, so a model may hand out several and they do not
 * interfere -- a hyper-reduced reduced model and the exact one it is measured against hold one
 * each.
 */
template<typename VectorType>
class SampledOperator
{
public:
  virtual ~SampledOperator() = default;

  /// Entities available to sample. Local to a rank, and to a partitioning.
  virtual std::size_t
  n_entities() const = 0;

  /// Install one weight per entity. Zero means "do not evaluate".
  virtual void
  set_weights(std::vector<double> const & weights) = 0;

  /**
   * The evaluation half, holding no reference to this model.
   *
   * Compiling reads the installed weights and gathers what the entities they select contribute,
   * so it costs a pass over those entities and is invalidated by set_weights(). Keep the result
   * and drop everything else: it is the whole of what a reduced model needs at run time.
   */
  virtual std::shared_ptr<CompiledOperator>
  compiled() = 0;

  /**
   * V^T R_e(V a) for **every** entity, row-major (n_entities, r).
   *
   * The training data: its column sums are the exact projected residual, and a sparse
   * non-negative weight vector reproducing them is a rule for evaluating on a few entities.
   * Ignores the installed weights, and is an offline quantity -- it touches the whole mesh.
   */
  virtual std::vector<double>
  contributions(std::vector<double> const & coefficients) = 0;

  /**
   * Where on the boundary data's schedule the next contributions() is taken.
   *
   * The counterpart of CompiledOperator::set_boundary_amplitude, and needed for the same reason:
   * a term reaching a Dirichlet boundary carries the prescribed data, and the weights are fitted
   * to reproduce *that* term. Training every state at one amplitude while the states came from a
   * schedule fits the wrong operator -- quietly, because the fit still converges.
   *
   * Defaults to doing nothing, which is right for fixed or homogeneous data.
   */
  virtual void
  set_boundary_amplitude(double const /*amplitude*/)
  {
  }

  /**
   * Draw the entities the installed weights select, as a VTU/PVTU record, and return its path.
   *
   * Optional, and a diagnostic: which entities a fit chose is the one thing about a
   * hyper-reduction that a number cannot show. An entity is not in general a cell, so how it is
   * drawn is the application's business -- a face term may well write cell data.
   */
  virtual std::string
  write_selection(std::string const & /*directory*/, std::string const & /*basename*/)
  {
    AssertThrow(false, dealii::ExcMessage("This operator does not implement write_selection()."));

    return {};
  }
};

/**
 * A nonlinear term, split into the half that projects exactly and the half that does not.
 *
 * The counterpart of SampledOperator, and the two are meant to be read together:
 *
 *     N(u) = Q(u) + S(u)
 *
 * Q is a polynomial in the state, so its Galerkin projection is a fixed tensor -- built once
 * offline by polarisation, contracted online at a cost independent of the mesh, and exact rather
 * than fitted. S is whatever is left, and is what SampledOperator hyper-reduces.
 *
 * Both halves are needed, and for different reasons. Q determines the tensor. N determines what
 * is *not* nonlinear: subtracting it from the momentum operator leaves the affine remainder, so r
 * applications settle the rest of the block. An application that offers only one of them offers
 * neither, which is why they are one object.
 *
 * A model returning nullptr here is not saying its operator is linear -- it is saying it has no
 * polynomial half to exploit, so a reductor must project the whole nonlinearity generically.
 */
template<typename VectorType>
class SplitOperator
{
public:
  virtual ~SplitOperator() = default;

  /// N(u), the whole nonlinear term, exactly as the solver evaluates it.
  virtual std::shared_ptr<VectorType>
  apply(VectorType const & u) const = 0;

  /**
   * Q(u), the polynomial half.
   *
   * Polarising this is what gives the tensor, so it must be the polynomial the *nonlinear*
   * operator contains -- not a linearised operator that happens to be multilinear. The two can
   * be different bilinear maps at a given mesh even when they agree in the limit, and a reduced
   * model should reproduce the operator its own snapshots came from.
   */
  virtual std::shared_ptr<VectorType>
  apply_polynomial(VectorType const & u) const = 0;

  /// Degree of the polynomial half, so a reductor knows how many indices its tensor has.
  virtual unsigned int
  polynomial_degree() const
  {
    return 2;
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

  /**
   * Name of the operator coefficient this component scales with, or empty when it belongs to a
   * parameter group instead and slot/index name one of its entries.
   *
   * A right-hand side is usually parameterised by amplitudes of its own. An inhomogeneous
   * boundary condition is different: it puts the *same* scalar into the operator and into the
   * right-hand side, and the two then have to move together. Naming the coefficient here is what
   * keeps one value in front of both, rather than a parameter that happens to be set twice.
   */
  std::string coefficient;
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
 *
 * **The step is the operator; the time loop is not.** A time integrator is an algorithm -- it
 * chooses the sequence of problems to solve -- and an algorithm cannot be projected. What can is
 * one implicitly discretised step,
 *
 *     s M u + N(u, p) = f,     s = gamma_0 / dt
 *
 * which is the steady problem again with a mass term and a right-hand side carrying the history.
 * So every method that evaluates or solves the momentum equation takes (mass_scaling, time), and
 * a caller stepping in time supplies them. s = 0 is the steady problem, and is what every steady
 * caller passes; nothing here special-cases it.
 *
 * The loop itself belongs to pyMOR, for both models. A reduced model has no ExaDG object to step
 * it, so its loop must be pyMOR's -- and a full-order model stepped by ExaDG against a reduced
 * one stepped by pyMOR would be two discretisations rather than a measurement. The history term
 * therefore arrives in f, already assembled by whoever owns the loop.
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

  /**
   * Coefficients of the operator that a caller may set, by name.
   *
   * A parameter carried by the right-hand side needs nothing here: velocity_rhs_components()
   * gives its affine decomposition and the Python layer forms the combination itself, never
   * telling the application which parameter it is solving at. A parameter that changes the
   * *operator* cannot be handled that way. The residual, the Jacobian and the solve all belong to
   * this application, and its Newton iteration reads the coefficient out of objects the Python
   * layer does not own -- so the only way to solve at a new value is to install it first.
   *
   * The operator is expected to be affine in each coefficient declared here. That is what lets a
   * reduced model project once per coefficient and assemble online, rather than reprojecting at
   * every parameter value -- which would need the full-order operator and would be no reduction
   * at all. An application that cannot honour this should not declare the coefficient.
   */
  virtual std::vector<std::string>
  coefficients() const
  {
    return {};
  }

  virtual double
  get_coefficient(std::string const & name) const
  {
    AssertThrow(false, dealii::ExcMessage("There is no coefficient named '" + name + "'."));

    return 0.0;
  }

  virtual void
  set_coefficient(std::string const & name, double const /*value*/)
  {
    AssertThrow(false, dealii::ExcMessage("There is no coefficient named '" + name + "'."));
  }

  /**
   * The degree of the polynomial the operator is in this coefficient.
   *
   * One is the usual answer and the default: a coefficient that multiplies one term of the
   * operator once, as a viscosity multiplies every viscous flux. Two is what a Dirichlet
   * amplitude needs, and for a reason worth stating rather than discovering -- the convective
   * flux is quadratic in the velocity, and prescribed boundary data *is* part of that velocity,
   * so the same scalar appears in the flux's linear part and again, squared, in its constant.
   *
   * A reduced model probes at ``degree + 1`` values of each coefficient and interpolates. Declare
   * too low and the fit is silently wrong away from the nodes; too high costs probes and is
   * harmless, since the surplus monomials come out zero.
   */
  virtual unsigned int
  coefficient_degree(std::string const & /*name*/) const
  {
    return 1;
  }

  /**
   * The coefficient the inhomogeneous Dirichlet data scales with, or empty when it is fixed.
   *
   * Everything else a coefficient touches can be decomposed and projected once. This one cannot
   * be, entirely: a Lax-Friedrichs lambda is a maximum of absolute values of a velocity that
   * includes the boundary data, so a sampled operator holding that term has to be handed the
   * number itself. Naming it here is how a reduced model knows which of its coefficients to pass
   * on -- see CompiledOperator::set_boundary_amplitude.
   */
  virtual std::string
  boundary_amplitude_coefficient() const
  {
    return {};
  }

  /// A, the (1,1) block at this step's mass scaling: velocity in, velocity out.
  virtual std::shared_ptr<LinearOperator<VectorType>>
  momentum(double mass_scaling) = 0;

  /// B, the (2,1) block: velocity in, pressure out. Its transpose is the (1,2) block.
  virtual std::shared_ptr<LinearOperator<VectorType>>
  divergence() = 0;

  /// The velocity inner product. Needed for supremizers, so effectively required.
  virtual std::shared_ptr<LinearOperator<VectorType>>
  velocity_product()
  {
    return nullptr;
  }

  /**
   * M, the velocity mass matrix -- the operator multiplying du/dt.
   *
   * Separate from velocity_product() even where an application returns the same handle, because
   * they are the same object only by coincidence. A model is free to make its velocity product
   * the H1 product, which is a reasonable choice for a proper orthogonal decomposition of
   * velocity fields and is not the mass matrix. Reading the time derivative's operator off the
   * inner product would then be wrong, and wrong in the way this interface exists to prevent:
   * plausibly, without failing.
   */
  virtual std::shared_ptr<LinearOperator<VectorType>>
  velocity_mass()
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

  /// The parameter-independent part of the pressure right-hand side g, or nullptr for zero.
  virtual std::shared_ptr<VectorType>
  pressure_rhs()
  {
    return nullptr;
  }

  /**
   * Affine components of g.
   *
   * The counterpart of velocity_rhs_components(), and needed for the same reason once the
   * Dirichlet data is not fixed: the continuity equation's right-hand side is the boundary term
   * of the divergence operator, so it carries whatever scales that data. It is *exactly* linear
   * in such a scalar, which the momentum equation's constant is not -- there the same data also
   * enters the convective flux quadratically.
   */
  virtual std::vector<AffineVector<VectorType>>
  pressure_rhs_components()
  {
    return {};
  }

  /**
   * Whether the operator depends on the state, i.e. whether solving needs a Newton iteration.
   *
   * The Python layer builds a different pyMOR model for the two: a linear saddle point is
   * projected once, a nonlinear one has to carry its residual and Jacobian so that the *reduced*
   * model can run its own Newton iteration -- which it must, since the reduced Jacobian is a
   * small dense matrix that ExaDG knows nothing about.
   */
  virtual bool
  is_nonlinear() const
  {
    return false;
  }

  /**
   * s M u + N(u, p), one step's operator, *without* the right-hand side.
   *
   * pyMOR's convention is that a model solves operator(U) = rhs, so the forcing is subtracted by
   * the model rather than by the operator -- and so is the history, which reaches the caller's
   * right-hand side rather than this method. Returns false for a model that is linear, where the
   * blocks say everything already.
   */
  virtual bool
  apply_nonlinear(VectorType const & /*u*/,
                  VectorType const & /*p*/,
                  VectorType & /*du*/,
                  VectorType & /*dp*/,
                  double const /*mass_scaling*/,
                  double const /*time*/)
  {
    return false;
  }

  /**
   * The (1,1) block of the Jacobian, linearised at the given velocity.
   *
   * Only that block depends on the state: B is linear, so the Jacobian of the whole system is
   * [[s M + A'(u), B*], [B, 0]] and the rest is unchanged. The mass term is linear, so it enters
   * the Jacobian at the same scaling it entered the residual -- pass the step's. Returns nullptr
   * for a linear model, whose momentum() is already its own Jacobian.
   */
  virtual std::shared_ptr<LinearOperator<VectorType>>
  jacobian_momentum(VectorType const & /*velocity*/, double const /*mass_scaling*/)
  {
    return nullptr;
  }

  /**
   * The momentum operator's nonlinear term, split into its polynomial and remaining halves.
   *
   * Returns nullptr for a linear model, or one whose nonlinearity has no polynomial part worth
   * separating. What the halves are used for is in SplitOperator; sampled_momentum() below
   * hyper-reduces the second of them.
   */
  virtual std::shared_ptr<SplitOperator<VectorType>>
  split_momentum()
  {
    return nullptr;
  }

  /**
   * The part of the momentum operator that cannot be projected exactly, as a sampled operator.
   *
   * Returns nullptr for a model with no such part, or one that does not offer hyper-reduction.
   * The basis is the one the reduced model projects onto -- after supremizer enrichment, if there
   * is any -- and the evaluator keeps it, so it must be complete before this is called.
   */
  virtual std::shared_ptr<SampledOperator<VectorType>>
  sampled_momentum(std::vector<std::shared_ptr<VectorType>> const & /*basis*/)
  {
    return nullptr;
  }

  /**
   * The full-order coupled solve for the given right-hand side.
   *
   * Takes the right-hand side rather than the parameters, because that is pyMOR's contract:
   * apply_inverse may be asked to solve with any vector, not only the one the model assembles at
   * a parameter. Kept as a single call rather than composed from the blocks because the block
   * preconditioner is the application's business -- and because for Navier-Stokes this is where
   * the Newton iteration lives.
   *
   * Returns false if this model cannot solve that system, the way FullOrderModel::assemble
   * returns nullptr.
   *
   * initial_guess is nullptr for a cold start, which is what a snapshot needs: it has to be a
   * function of its parameter alone, and warm-starting from the previous one would make it a
   * function of the order they were solved in. A time loop wants the opposite -- the previous
   * step is the best guess there is -- so the choice belongs to the caller.
   */
  virtual bool
  solve(VectorType const & /*f*/,
        VectorType const & /*g*/,
        VectorType & /*velocity*/,
        VectorType & /*pressure*/,
        double const /*mass_scaling*/,
        double const /*time*/,
        VectorType const * /*initial_guess*/)
  {
    return false;
  }
};

} // namespace PyMOR
} // namespace ExaDG

#endif /* EXADG_PYMOR_INTERFACE_H_ */
