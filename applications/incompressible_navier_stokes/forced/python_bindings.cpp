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
 * The forced box as a PyMOR::SaddlePointModel.
 *
 * Everything about ExaDG's coupled operator -- the three blocks, the convective split, the
 * sampled stabilisation, the step -- is in IncNSSaddlePoint, because none of it is about this
 * problem. What is left here is the parameterisation: the parameters are the amplitudes of a body
 * force expanded in Gaussian modes, and they enter the right-hand side alone.
 *
 * That is the smallest problem that still poses the saddle-point question, and one whose answer
 * is known for Stokes: the solution is linear in the amplitudes, so a basis of the P mode
 * solutions has to reproduce the full-order model exactly.
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
 * Incompressible flow in a box, driven by a body force expanded in modes.
 */
template<int dim>
class ForcedFOM : public IncNSSaddlePoint<dim, Application<dim, Number>>
{
public:
  using Base = IncNSSaddlePoint<dim, Application<dim, Number>>;
  using Base::Base;

    /// One group, holding the forcing amplitudes.
    std::vector<unsigned int>
    parameter_shape() const override
    {
      return {this->application->get_forcing()->n_modes()};
    }

    /**
     * The velocity right-hand side at zero amplitudes.
     *
     * Not assumed to vanish. The boundary conditions here are homogeneous, so it should, but
     * rhs_linear_problem() also collects the boundary terms of the gradient and viscous operators
     * and this is the honest constant term of the affine decomposition either way. The example
     * checks that it is zero rather than this file asserting it.
     */
    std::shared_ptr<VectorType>
    velocity_rhs() override
    {
      return assemble_rhs(std::vector<double>(n_modes(), 0.0)).first;
    }

    /// f_i = rhs(e_i) - rhs(0), one per forcing mode.
    std::vector<PyMOR::AffineVector<VectorType>>
    velocity_rhs_components() override
    {
      auto const constant = velocity_rhs();

      std::vector<PyMOR::AffineVector<VectorType>> components;
      for(unsigned int i = 0; i < n_modes(); ++i)
      {
        std::vector<double> amplitudes(n_modes(), 0.0);
        amplitudes[i] = 1.0;

        auto mode = assemble_rhs(amplitudes).first;
        mode->add(-1.0, *constant);

        components.push_back({mode, 0 /* slot */, i});
      }

      return components;
    }

    /**
     * Writes the body force of a parameter as its own VTU record.
     *
     * The forcing is a smooth function, so interpolating it onto the velocity space loses nothing
     * and puts it on the same mesh as the field it drives.
     */
    std::string
    write_forcing(std::string const &         directory,
                  std::string const &         basename,
                  std::vector<double> const & amplitudes)
    {
      AssertThrow(amplitudes.size() == n_modes(),
                  dealii::ExcMessage("Expected " + std::to_string(n_modes()) +
                                     " amplitudes, got " + std::to_string(amplitudes.size()) + "."));

      this->application->get_forcing()->set_amplitudes(amplitudes);

      auto field = std::make_shared<VectorType>();
      this->pde_operator->initialize_vector_velocity(*field);

      dealii::VectorTools::interpolate(*this->pde_operator->get_mapping(),
                                       this->pde_operator->get_dof_handler_u(),
                                       *this->application->get_forcing(),
                                       *field);

      return this->write_fields(this->pde_operator->get_dof_handler_u(),
                          directory,
                          basename,
                          {field},
                          {"forcing"},
                          true /* vector valued */);
    }

    /// Number of forcing modes, i.e. of parameters.
    unsigned int
    n_modes() const
    {
      return this->application->get_forcing()->n_modes();
    }

    /// The coupled right-hand side at the given amplitudes, as (velocity, pressure).
    std::pair<std::shared_ptr<VectorType>, std::shared_ptr<VectorType>>
    assemble_rhs(std::vector<double> const & amplitudes)
    {
      this->application->get_forcing()->set_amplitudes(amplitudes);

      BlockVectorType rhs;
      this->pde_operator->initialize_block_vector_velocity_pressure(rhs);

      VectorType transport_velocity;
      this->pde_operator->rhs_linear_problem(rhs, transport_velocity, 0.0);

      auto velocity = std::make_shared<VectorType>(rhs.block(0));
      auto pressure = std::make_shared<VectorType>(rhs.block(1));

      return {velocity, pressure};
    }
};


template<int dim>
void
register_model(py::module_ & module, std::string const & name)
{
  py::class_<ForcedFOM<dim>,
             PyMOR::SaddlePointModel<VectorType>,
             std::shared_ptr<ForcedFOM<dim>>>(module, name.c_str())
    .def(py::init<std::string const &, unsigned int, unsigned int, bool>(),
         py::arg("input_file"),
         py::arg("degree")      = 2,
         py::arg("refinements") = 4,
         py::arg("verbose")     = false)
    .def_property_readonly("n_modes", &ForcedFOM<dim>::n_modes)
    .def_property_readonly("upwind_factor", &ForcedFOM<dim>::get_upwind_factor)
    .def_property_readonly("max_velocity", &ForcedFOM<dim>::get_max_velocity)
    .def("time_step_for_cfl",
         &ForcedFOM<dim>::time_step_for_cfl,
         py::arg("cfl"),
         "The time step this mesh admits at that CFL number, from ExaDG's own criterion.")
    .def_property_readonly("quadrature_indices", &ForcedFOM<dim>::quadrature_indices)
    .def("apply_convective",
         &ForcedFOM<dim>::apply_convective,
         py::arg("u"),
         "N(u), the convective operator as the solver evaluates it.")
    .def("apply_convective_central",
         &ForcedFOM<dim>::apply_convective_central,
         py::arg("u"),
         "N(u) with a central flux, i.e. at upwind_factor = 0.")
    .def_property_readonly("n_faces", &ForcedFOM<dim>::n_faces)
    .def("apply_stabilisation", &ForcedFOM<dim>::apply_stabilisation, py::arg("u"),
         "S(u) summed over every face, as a full-order vector.")

    .def("apply_trilinear",
         &ForcedFOM<dim>::apply_trilinear,
         py::arg("w"),
         py::arg("v"),
         "C(w, v), the trilinear part of the convective operator.")
    .def("write_forcing",
         &ForcedFOM<dim>::write_forcing,
         py::arg("directory"),
         py::arg("basename"),
         py::arg("amplitudes"),
         "Write the body force of a parameter as a VTU record.");
}

} // namespace IncNS
} // namespace ExaDG

PYBIND11_MODULE(forced, module)
{
  using namespace ExaDG::IncNS;

  py::module_::import("exadg._core");

  module.doc() =
    "Steady incompressible flow in a box, driven by a parameterised body force.\n\n"
    "A velocity/pressure saddle point exposed through exadg/pymor/interface.h. Wrap it with "
    "exadg.mor.models.saddle_point.saddle_point_model().";

  register_model<2>(module, "ForcedFOM2D");
  register_model<3>(module, "ForcedFOM3D");
}

