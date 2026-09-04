#########################################################################
#
#                 #######               ######  #######
#                 ##                    ##   ## ##
#                 #####   ##  ## #####  ##   ## ## ####
#                 ##       ####  ## ##  ##   ## ##   ##
#                 ####### ##  ## ###### ######  #######
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2021 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#########################################################################

TARGETNAME(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR})

PROJECT(${TARGET_NAME})

EXADG_PICKUP_EXE(solver.cpp ${TARGET_NAME} solver)


# Python bindings, staged into the exadg package by the shared macro.
IF (EXADG_WITH_PYTHON)
  EXADG_ADD_PYTHON_MODULE(thermal_block python_bindings.cpp)
ENDIF()
