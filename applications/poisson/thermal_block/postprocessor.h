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

#ifndef APPLICATIONS_POISSON_THERMAL_BLOCK_POSTPROCESSOR_H_
#define APPLICATIONS_POISSON_THERMAL_BLOCK_POSTPROCESSOR_H_

// C/C++
#include <fstream>
#include <iomanip>

// ExaDG
#include <exadg/poisson/postprocessor/postprocessor.h>
#include <exadg/pymor/sensor_operator.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace Poisson
{
struct ThermalBlockOutputData
{
  // Write the values at the sensor points, which form the data of the inverse problem.
  bool write_sensors = false;

  // Number of sensors per coordinate direction; the total is sensors_per_dim^dim.
  unsigned int sensors_per_dim = 4;

  std::string directory = "output/";
  std::string filename  = "thermal_block";
};

/**
 * Postprocessor of the thermal block benchmark.
 *
 * Beyond the usual output it writes the values at the sensor points, which are the full-order
 * counterpart of ExaDGObservationOperator on the Python side. Snapshots are not written here:
 * pyMOR drives the solves itself and keeps the resulting vectors in a VectorArray, so they
 * never need to reach the file system.
 */
template<int dim, int n_components, typename Number>
class ThermalBlockPostProcessor : public PostProcessor<dim, n_components, Number>
{
private:
  typedef PostProcessor<dim, n_components, Number> Base;
  typedef typename Base::VectorType                VectorType;

public:
  ThermalBlockPostProcessor(PostProcessorData<dim> const &  pp_data,
                            MPI_Comm const &                mpi_comm,
                            ThermalBlockOutputData const &  output_data)
    : Base(pp_data, mpi_comm), output_data(output_data)
  {
  }

  void
  setup(Operator<dim, n_components, Number> const & pde_operator) override
  {
    Base::setup(pde_operator);

    dof_handler = &pde_operator.get_dof_handler();
    mapping     = pde_operator.get_mapping();

    if(output_data.write_sensors)
      create_directories(output_data.directory, this->mpi_comm);

    if(output_data.write_sensors)
    {
      sensors.setup(dof_handler->get_triangulation(),
                    *mapping,
                    SensorOperator<dim, Number>::interior_cartesian_grid(
                      output_data.sensors_per_dim));
    }
  }

  void
  do_postprocessing(VectorType const &     solution,
                    double const           time             = 0.0,
                    types::time_step const time_step_number = numbers::steady_timestep) override
  {
    Base::do_postprocessing(solution, time, time_step_number);

    if(output_data.write_sensors)
      write_sensor_values(solution);
  }

private:
  void
  write_sensor_values(VectorType const & solution)
  {
    std::vector<Number> const values = sensors.evaluate(*dof_handler, solution);

    // rank 0 writes; the values are identical on all ranks
    if(dealii::Utilities::MPI::this_mpi_process(this->mpi_comm) != 0)
      return;

    std::string const path = output_data.directory + output_data.filename + "_sensors.txt";

    std::ofstream file(path);
    AssertThrow(file.is_open(), dealii::ExcMessage("Could not open " + path + " for writing."));

    file << "# sensor values of the thermal block benchmark\n";
    file << "#";
    for(unsigned int d = 0; d < dim; ++d)
      file << " x" << d;
    file << " value\n";

    file << std::scientific << std::setprecision(16);

    auto const & points = sensors.get_points();
    for(unsigned int i = 0; i < values.size(); ++i)
    {
      for(unsigned int d = 0; d < dim; ++d)
        file << points[i][d] << " ";

      file << values[i] << "\n";
    }
  }

  ThermalBlockOutputData output_data;

  dealii::DoFHandler<dim> const *              dof_handler = nullptr;
  std::shared_ptr<dealii::Mapping<dim> const>  mapping;

  SensorOperator<dim, Number> sensors;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* APPLICATIONS_POISSON_THERMAL_BLOCK_POSTPROCESSOR_H_ */
