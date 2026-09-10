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
 * That holds only while the inflow does not depend on time. Test cases 2 and 3 ramp and oscillate
 * it, which would make that constant a function of t; test case 1's inflow is steady, and at the
 * Reynolds number of test case 2 it still sheds.
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
  using Base::Base;

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
