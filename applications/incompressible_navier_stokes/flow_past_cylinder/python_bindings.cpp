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
 * Flow past a cylinder as a PyMOR::SaddlePointModel.
 *
 * Everything about ExaDG's coupled operator is in IncNSSaddlePoint. What is left here is what
 * this problem is: a channel with a cylinder in it, driven by a parabolic inflow rather than by a
 * body force, and with the Reynolds number as the quantity worth varying.
 *
 * Two things make it a harder case than the forced box, and both are the point of running it.
 *
 * **There is no right-hand side.** The flow is driven entirely through an inhomogeneous Dirichlet
 * inflow. That would be a problem for a reduced model built on a constrained space -- a
 * combination of snapshots would not satisfy the boundary condition -- and is not one here,
 * because the velocity space is discontinuous and the condition is imposed *weakly*. No degree of
 * freedom is constrained; the inflow enters the residual as a boundary flux whose
 * state-independent part is exactly the constant term the reduced operator already carries.
 *
 * Test cases 2 and 3 ramp and oscillate the inflow, which makes that constant a function of time.
 * What keeps it tractable is that the schedule is *separable*: one scalar in front of a fixed
 * spatial profile. Set InflowIsParameter and the scalar becomes a coefficient like the viscosity,
 * so the schedule lives in the caller's parameter rather than in this application's clock -- which
 * is what a reduced model needs, since it has to evaluate the operator at an amplitude of its
 * choosing and not at whatever the clock says.
 *
 * The two right-hand sides then behave differently, and the difference is not cosmetic. The
 * continuity equation's is *exactly* linear in that scalar, so it is declared as an affine
 * component. The momentum equation's is not: the same boundary data enters the convective flux
 * quadratically, so its constant carries the amplitude squared.
 *
 * **The flow has dynamics of its own.** Above a Reynolds number of about 47 a cylinder wake is
 * unsteady whatever the inflow does, so this is the first case here whose reduced model has to
 * represent a limit cycle rather than a relaxation -- and the first where the Kolmogorov width of
 * the solution manifold is a real constraint rather than a remark.
 */

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// ExaDG
#include <exadg/pymor/incompressible_flow.h>

// application
#include "application.h"

namespace py = pybind11;

namespace ExaDG
{
namespace IncNS
{
/**
 * Flow past a cylinder, driven by a parabolic inflow.
 */
template<int dim>
class CylinderFOM : public IncNSSaddlePoint<dim, Application<dim, Number>>
{
public:
  using Base = IncNSSaddlePoint<dim, Application<dim, Number>>;

  CylinderFOM(std::string const & input_file,
              unsigned int const  degree,
              unsigned int const  refinements,
              bool const          verbose)
    : Base(input_file, degree, refinements, verbose)
  {
    // An amplitude that is a parameter starts at the top of the profile rather than at whatever
    // the schedule says, so that a caller which never sets it gets the steady inflow and not a
    // NaN. The schedule is still reachable through scheduled_inflow_amplitude().
    if(this->application->inflow_is_parameter())
      this->application->set_inflow_amplitude(1.0);
  }

  /**
   * No parameters yet.
   *
   * The Reynolds number is the one worth having, and it enters the *operator* rather than the
   * right-hand side -- the viscous block is exactly linear in the viscosity, since the interior
   * penalty parameter is purely geometric and every viscous flux carries one factor of it. That
   * is an affine decomposition of the momentum block, which PyMOR::SaddlePointModel does not yet
   * express; FullOrderModel's operator_components() is the shape it will take.
   */
  std::vector<unsigned int>
  parameter_shape() const override
  {
    return {};
  }

  /**
   * Nothing. The flow is driven through the boundary, not by a force.
   *
   * ExaDG's residual already carries the inflow's contribution, so there is nothing to add and
   * nothing to subtract -- see the note on weak boundary conditions above.
   */
  std::shared_ptr<VectorType>
  velocity_rhs() override
  {
    return nullptr;
  }

  /** The Reynolds number is not a forcing amplitude, so there are no components here. */
  std::vector<PyMOR::AffineVector<VectorType>>
  velocity_rhs_components() override
  {
    return {};
  }

  /*
   * Reynolds number, on the cylinder diameter and the mean inflow, as the benchmark defines it.
   *
   * Settable, and it is the viscosity that moves: the inflow is what the boundary condition
   * prescribes, and scaling that would rescale the convective term as well, leaving the momentum
   * operator no longer affine in the parameter. Viscosity is the coefficient the operator is
   * linear in, so a Reynolds sweep is an affine sweep.
   */
  /** The Reynolds number is the parameter here, and the viscosity is how it is set. */
  bool
  viscosity_is_parameter() const override
  {
    return true;
  }

  /*
   * The inflow amplitude joins the viscosity as a coefficient, when the application says it is a
   * parameter rather than a schedule.
   *
   * It is one scalar in front of a fixed spatial profile, and that separability is what makes it
   * usable: the operator is a polynomial in it -- degree one in the viscous lift and the linear
   * part of the convective flux, degree two in that flux's constant -- and the continuity
   * equation's right-hand side is exactly linear in it.
   */
  std::vector<std::string>
  coefficients() const override
  {
    auto names = Base::coefficients();

    if(this->application->inflow_is_parameter())
      names.push_back("inflow");

    return names;
  }

  /*
   * The inflow amplitude is what the Dirichlet data scales with, so the sampled stabilisation can
   * be compiled at a known one and told the rest later. Its lambda is a maximum of absolute
   * values, so there is nothing to decompose there and the schedule has to arrive as a number.
   */
  /*
   * The inflow amplitude enters twice over, so a reduced model has to fit a parabola in it.
   *
   * Once through the convective flux's linear part -- the prescribed value multiplies the
   * interior velocity there -- and once through its constant, where it multiplies itself. The
   * viscosity keeps the default degree of one.
   */
  unsigned int
  coefficient_degree(std::string const & name) const override
  {
    if(name == "inflow")
      return 2;

    return Base::coefficient_degree(name);
  }

  std::string
  boundary_amplitude_coefficient() const override
  {
    if(not this->application->inflow_is_parameter())
      return {};

    return "inflow";
  }

  double
  boundary_amplitude() const override
  {
    if(not this->application->inflow_is_parameter())
      return 1.0;

    return this->application->get_inflow_amplitude();
  }

  void
  set_boundary_amplitude(double const amplitude) override
  {
    if(this->application->inflow_is_parameter())
      this->application->set_inflow_amplitude(amplitude);
  }

  double
  get_coefficient(std::string const & name) const override
  {
    if(name == "inflow")
      return this->application->get_inflow_amplitude();

    return Base::get_coefficient(name);
  }

  void
  set_coefficient(std::string const & name, double const value) override
  {
    if(name == "inflow")
    {
      this->application->set_inflow_amplitude(value);
      return;
    }

    Base::set_coefficient(name, value);
  }

  /*
   * g, split so that the amplitude sits in front of it rather than inside it.
   *
   * With a fixed inflow the whole of g is a constant and the base class hands it over as one.
   * With the amplitude as a parameter none of it is: g is exactly linear in that scalar, so the
   * component is g at an amplitude of one and the coefficient carries the rest. Returning both
   * would count the inflow twice.
   */
  std::shared_ptr<VectorType>
  pressure_rhs() override
  {
    if(this->application->inflow_is_parameter())
      return nullptr;

    return Base::pressure_rhs();
  }

  std::vector<PyMOR::AffineVector<VectorType>>
  pressure_rhs_components() override
  {
    if(not this->application->inflow_is_parameter())
      return {};

    double const restore = this->application->get_inflow_amplitude();
    this->application->set_inflow_amplitude(1.0);

    PyMOR::AffineVector<VectorType> component;
    component.vector = std::make_shared<VectorType>(
      this->continuity_rhs(this->application->get_parameters().start_time));
    component.coefficient = "inflow";

    this->application->set_inflow_amplitude(restore);

    return {component};
  }

  /** The schedule this test case would follow on its own, for a caller reproducing it. */
  double
  scheduled_inflow_amplitude(double const time) const
  {
    return this->application->scheduled_inflow_amplitude(time);
  }

  double
  reynolds_number() const
  {
    return reference_velocity() * DIAMETER / this->application->get_parameters().viscosity;
  }

  void
  set_reynolds_number(double const reynolds_number)
  {
    AssertThrow(reynolds_number > 0.0,
                dealii::ExcMessage("The Reynolds number has to be positive."));

    this->set_viscosity(reference_velocity() * DIAMETER / reynolds_number);
  }

private:
  static constexpr double DIAMETER = 0.1;

  /** Mean inflow over the channel height, which is what the benchmark's Reynolds number uses. */
  double
  reference_velocity() const
  {
    return this->application->get_max_inflow() * (dim == 2 ? 2.0 / 3.0 : 4.0 / 9.0);
  }
};

template<int dim>
void
register_model(py::module_ & module, std::string const & name)
{
  py::class_<CylinderFOM<dim>,
             PyMOR::SaddlePointModel<VectorType>,
             std::shared_ptr<CylinderFOM<dim>>>(module, name.c_str())
    .def(py::init<std::string const &, unsigned int, unsigned int, bool>(),
         py::arg("input_file"),
         py::arg("degree")      = 2,
         py::arg("refinements") = 0,
         py::arg("verbose")     = false)
    .def_property("reynolds_number",
                  &CylinderFOM<dim>::reynolds_number,
                  &CylinderFOM<dim>::set_reynolds_number)
    .def_property("viscosity", &CylinderFOM<dim>::get_viscosity, &CylinderFOM<dim>::set_viscosity)
    .def("scheduled_inflow_amplitude",
         &CylinderFOM<dim>::scheduled_inflow_amplitude,
         py::arg("time"),
         "The fraction of the inflow profile this test case applies at that time, for a caller "
         "reproducing its schedule as a parameter.")
    .def_property_readonly("upwind_factor", &CylinderFOM<dim>::get_upwind_factor)
    .def_property_readonly("n_faces", &CylinderFOM<dim>::n_faces)
    .def("time_step_for_cfl",
         &CylinderFOM<dim>::time_step_for_cfl,
         py::arg("cfl"),
         "The time step this mesh admits at that CFL number, from ExaDG's own criterion.")
    .def("apply_convective",
         &CylinderFOM<dim>::apply_convective,
         py::arg("u"),
         "N(u), the convective operator as the solver evaluates it.");
}

} // namespace IncNS
} // namespace ExaDG

PYBIND11_MODULE(flow_past_cylinder, module)
{
  using namespace ExaDG::IncNS;

  py::module_::import("exadg._core");

  module.doc() =
    "Flow past a cylinder, driven by a parabolic inflow.\n\n"
    "A velocity/pressure saddle point exposed through exadg/pymor/interface.h. Needs "
    "Formulation = Coupled in the input file; wrap it with "
    "exadg.mor.models.instationary_saddle_point.mpi_instationary_saddle_point_model().";

  register_model<2>(module, "CylinderFOM2D");
  register_model<3>(module, "CylinderFOM3D");
}
