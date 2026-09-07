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
 * exadg._core -- the vector and operator vocabulary of exadg/pymor/interface.h, bound once.
 *
 * Every application module imports this one and returns objects of the types registered here.
 *
 * Nothing here knows any physics; it only makes the abstract base classes visible to Python, so
 * that python/exadg/mor/binding.py can wrap any of them without knowing what it is wrapping.
 */

// C/C++
#include <string>
#include <vector>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// deal.II
#include <deal.II/base/init_finalize.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/pymor/interface.h>

namespace py = pybind11;

namespace ExaDG
{
namespace PyMOR
{
using Number     = double;
using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

/**
 * Initializes deal.II's external libraries exactly once per interpreter, starting MPI only if
 * nobody else has.
 *
 * Under pyMOR's event loop mpi4py is imported first and has already called MPI_Init by the time
 * this module loads. deal.II's MPI_InitFinalize asserts MPI_Initialized() == 0 and would abort
 * with "MPI error. You can only start MPI once!". Kokkos, PETSc, Zoltan and p4est still have to
 * be brought up, so the fix is to initialize the same set of libraries as MPI_InitFinalize with
 * the MPI bit cleared -- deal.II guards only the MPI branch with that assertion.
 *
 * The object is deliberately leaked. Its destructor finalizes MPI and the libraries, and running
 * that during interpreter teardown -- after an arbitrary amount of other cleanup, and possibly
 * after deal.II statics have gone -- is a reliable source of crashes at exit.
 */
void
ensure_mpi_initialized()
{
  static bool initialized = false;

  if(initialized)
    return;

  static char   program_name[] = "exadg";
  static char * argv_storage[] = {program_name, nullptr};

  int     argc = 1;
  char ** argv = argv_storage;

  int mpi_already_started = 0;
  MPI_Initialized(&mpi_already_started);

  // the set MPI_InitFinalize uses, so that behaviour is identical either way
  auto libraries = dealii::InitializeLibrary::Kokkos | dealii::InitializeLibrary::SLEPc |
                   dealii::InitializeLibrary::PETSc | dealii::InitializeLibrary::Zoltan |
                   dealii::InitializeLibrary::P4EST | dealii::InitializeLibrary::PSBLAS;

  if(not mpi_already_started)
    libraries = libraries | dealii::InitializeLibrary::MPI;

  new dealii::InitFinalize(argc, argv, libraries, 1);

  initialized = true;
}

/**
 * Registers a vector type and everything templated on it, under the given name prefix.
 *
 * Templated for other vector types, e.g. a block vector for the saddle-point models.
 */
template<typename V>
void
register_vector_type(py::module_ & module, std::string const & prefix)
{
  py::class_<V, std::shared_ptr<V>>(module, (prefix + "Vector").c_str())
    .def_property_readonly("dim", [](V const & v) { return v.size(); })
    .def("copy", [](V const & v) { return std::make_shared<V>(v); }, "Deep copy.")
    .def("scal", [](V & v, double const a) { v *= a; }, py::arg("alpha"))
    .def(
      "axpy",
      [](V & v, double const a, V const & x) { v.add(a, x); },
      py::arg("alpha"),
      py::arg("x"))
    .def(
      "inner",
      [](V const & v, V const & x) { return v * x; },
      py::arg("other"),
      "Euclidean inner product; reduces over the communicator internally.")
    .def("norm", [](V const & v) { return v.l2_norm(); })
    .def("norm2", [](V const & v) { return v.norm_sqr(); })
    .def("sup_norm", [](V const & v) { return v.linfty_norm(); })
    .def(
      "amax",
      [](V const & v) {
        // Index and magnitude of the largest entry, which is what empirical interpolation uses
        // to pick its next interpolation point. The index is *global*, because that is the
        // index space dofs() is addressed in.
        //
        // Two reductions rather than one MPI_MAXLOC: a maximum over the values, then a minimum
        // over the global indices of the ranks that attain it. That costs one extra allreduce
        // but ensures a deterministic tie-break: the smallest global index always wins, whatever
        // the partitioning. MAXLOC would break ties by rank, so the interpolation point chosen
        // would depend on the number of ranks, and a basis built on two ranks would differ from
        // one built on four.
        auto const & partitioner = *v.get_partitioner();

        double local_value = 0.0;
        for(unsigned int i = 0; i < v.locally_owned_size(); ++i)
          local_value = std::max(local_value, std::abs(v.local_element(i)));

        double const value = dealii::Utilities::MPI::max(local_value, v.get_mpi_communicator());

        auto local_index = dealii::numbers::invalid_dof_index;
        for(unsigned int i = 0; i < v.locally_owned_size(); ++i)
          if(std::abs(v.local_element(i)) == value)
          {
            local_index = partitioner.local_to_global(i);
            break;
          }

        auto const index = dealii::Utilities::MPI::min(local_index, v.get_mpi_communicator());

        return py::make_tuple(index, value);
      })
    .def(
      "dofs",
      [](V const & v, std::vector<dealii::types::global_dof_index> const & indices) {
        // Collective. pyMOR reaches this through mpi.call, which runs the same call on every
        // rank and keeps rank 0's return value, so every rank has to come back with the whole
        // answer rather than with its own slice.
        //
        // Exactly one rank owns each index, so a sum over ranks with zero from the others
        // gathers the values without anyone needing to know who owns what.
        auto const & partitioner = *v.get_partitioner();

        std::vector<double> values(indices.size(), 0.0);
        for(unsigned int i = 0; i < indices.size(); ++i)
          if(partitioner.in_local_range(indices[i]))
            values[i] = v.local_element(partitioner.global_to_local(indices[i]));

        std::vector<double> gathered(indices.size());
        dealii::Utilities::MPI::sum(values, v.get_mpi_communicator(), gathered);

        return gathered;
      },
      py::arg("indices"),
      "Selected entries by global index, as pyMOR's empirical interpolation requires. "
      "Collective: every rank returns the same values.")
    .def(
      "assign_numpy",
      [](V & v, py::array_t<double, py::array::c_style | py::array::forcecast> const & data) {
        AssertThrow(static_cast<dealii::types::global_dof_index>(data.size()) == v.size(),
                    dealii::ExcMessage("Expected " + std::to_string(v.size()) + " entries, got " +
                                       std::to_string(data.size()) + "."));

        // Every rank receives the whole array and keeps the slice it owns. That is pyMOR's model
        // under mpi.call: the same call runs on every rank with the same arguments, so writing a
        // non-owned entry would be both wrong and a deal.II assertion.
        auto view = data.template unchecked<1>();
        for(auto const index : v.locally_owned_elements())
          v[index] = view(index);

        v.compress(dealii::VectorOperation::insert);
      },
      py::arg("data"),
      "Overwrite from a NumPy array holding the whole global vector.")
    .def(
      "to_numpy",
      [](V const & v) {
        // Refuses rather than returning the local slice. pyMOR never calls this on more than
        // one rank -- MPIVectorArrayImpl.to_numpy raises -- so a caller who gets here in
        // parallel is a test or a debug print that would otherwise silently compare slices.
        AssertThrow(dealii::Utilities::MPI::n_mpi_processes(v.get_mpi_communicator()) == 1,
                    dealii::ExcMessage("to_numpy() would gather the whole vector onto every "
                                       "rank, which is exactly what this interface exists to "
                                       "avoid. Use dofs() for selected entries."));

        py::array_t<double> array(v.size());
        auto                view = array.mutable_unchecked<1>();
        for(unsigned int i = 0; i < v.size(); ++i)
          view(i) = v[i];

        return array;
      },
      "Copy into a NumPy array. Single rank only; for debugging and tests.");

  py::class_<LinearOperator<V>, std::shared_ptr<LinearOperator<V>>>(
    module, (prefix + "LinearOperator").c_str())
    .def_property_readonly("name", &LinearOperator<V>::get_name)
    .def_property_readonly("is_symmetric", &LinearOperator<V>::is_symmetric)
    .def_property_readonly("has_inverse", &LinearOperator<V>::has_inverse)
    .def(
      "apply",
      [](LinearOperator<V> const & op, V & dst, V const & src) { op.apply(dst, src); },
      py::arg("dst"),
      py::arg("src"))
    .def(
      "apply_transpose",
      [](LinearOperator<V> const & op, V & dst, V const & src) { op.apply_transpose(dst, src); },
      py::arg("dst"),
      py::arg("src"))
    .def("apply_inverse", &LinearOperator<V>::apply_inverse, py::arg("rhs"))
    .def("apply_inverse_transpose", &LinearOperator<V>::apply_inverse_transpose, py::arg("rhs"))
    .def("restricted", &LinearOperator<V>::restricted, py::arg("output_dofs"));

  py::class_<ParametricOperator<V>, LinearOperator<V>, std::shared_ptr<ParametricOperator<V>>>(
    module, (prefix + "ParametricOperator").c_str())
    .def_property_readonly("parameter_slot", &ParametricOperator<V>::parameter_slot)
    .def("set_coefficients", &ParametricOperator<V>::set_coefficients, py::arg("coefficients"));

  py::class_<Functional<V>, std::shared_ptr<Functional<V>>>(module,
                                                            (prefix + "Functional").c_str())
    .def_property_readonly("n_outputs", &Functional<V>::n_outputs)
    .def("apply", &Functional<V>::apply, py::arg("src"))
    .def("apply_transpose", &Functional<V>::apply_transpose, py::arg("weights"));

  py::class_<AffineComponent<V>>(module, (prefix + "AffineComponent").c_str())
    .def_readonly("op", &AffineComponent<V>::op)
    .def_readonly("slot", &AffineComponent<V>::slot)
    .def_readonly("index", &AffineComponent<V>::index);

  py::class_<AffineVector<V>>(module, (prefix + "AffineVector").c_str())
    .def_readonly("vector", &AffineVector<V>::vector)
    .def_readonly("slot", &AffineVector<V>::slot)
    .def_readonly("index", &AffineVector<V>::index);

  py::class_<FullOrderModel<V>, std::shared_ptr<FullOrderModel<V>>>(
    module, (prefix + "FullOrderModel").c_str())
    .def_property_readonly("n_dofs", &FullOrderModel<V>::n_dofs)
    .def_property_readonly("parameter_shape", &FullOrderModel<V>::parameter_shape)
    .def("zero_vector", &FullOrderModel<V>::zero_vector)
    .def(
      "make_admissible",
      [](FullOrderModel<V> const & m, V & v) { m.make_admissible(v); },
      py::arg("vector"))
    .def("operator_components", &FullOrderModel<V>::operator_components)
    .def("parametric_operator", &FullOrderModel<V>::parametric_operator)
    .def("assemble", &FullOrderModel<V>::assemble, py::arg("coefficients"))
    .def("rhs", &FullOrderModel<V>::rhs)
    .def("rhs_components", &FullOrderModel<V>::rhs_components)
    .def("products", &FullOrderModel<V>::products)
    .def("output_functional", &FullOrderModel<V>::output_functional)
    .def("write_vtu",
         &FullOrderModel<V>::write_vtu,
         py::arg("directory"),
         py::arg("basename"),
         py::arg("fields"),
         py::arg("names"));
}

} // namespace PyMOR
} // namespace ExaDG

PYBIND11_MODULE(_core, module)
{
  using namespace ExaDG::PyMOR;

  module.doc() =
    "ExaDG's vector and operator vocabulary for pyMOR, bound once for the whole process.\n\n"
    "Application modules import this one and return objects of these types; "
    "exadg.mor.binding wraps them in pyMOR's interfaces without knowing what they discretise.";

  ensure_mpi_initialized();

  module.def("mpi_size",
             [] { return dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); },
             "Number of MPI ranks in MPI_COMM_WORLD.");

  module.def("mpi_rank",
             [] { return dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD); },
             "This process's rank in MPI_COMM_WORLD.");

  py::class_<RestrictedOperator, std::shared_ptr<RestrictedOperator>>(module, "RestrictedOperator")
    .def_property_readonly("source_dofs", &RestrictedOperator::get_source_dofs)
    .def("apply", &RestrictedOperator::apply, py::arg("source_values"))
    .def_property_readonly("active_components", &RestrictedOperator::active_components)
    .def("set_coefficients", &RestrictedOperator::set_coefficients, py::arg("coefficients"));

  register_vector_type<VectorType>(module, "");
}
