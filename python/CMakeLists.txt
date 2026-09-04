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
#########################################################################

# Stages an installable Python package in the build tree. Extension modules are placed here by
# EXADG_ADD_PYTHON_MODULE(); this directory only provides the package metadata around them, so
# that the result can be installed with pip instead of requiring PYTHONPATH.

# pybind11 is located by the top-level CMakeLists so its targets are globally visible.

SET(EXADG_PYTHON_PACKAGE_DIR ${CMAKE_BINARY_DIR}/python CACHE INTERNAL
    "Staging directory of the exadg Python package")
SET(EXADG_PYTHON_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/python CACHE INTERNAL
    "Install directory of the exadg Python package")

SET(EXADG_PYTHON_VERSION "0.1.0")

FILE(MAKE_DIRECTORY ${EXADG_PYTHON_PACKAGE_DIR}/exadg)
FILE(MAKE_DIRECTORY ${EXADG_PYTHON_PACKAGE_DIR}/exadg/mor)

CONFIGURE_FILE(pyproject.toml.in ${EXADG_PYTHON_PACKAGE_DIR}/pyproject.toml @ONLY)
CONFIGURE_FILE(exadg/__init__.py.in ${EXADG_PYTHON_PACKAGE_DIR}/exadg/__init__.py @ONLY)

# The model order reduction subpackage is plain Python and is copied rather than configured.
# COPYONLY keeps the build tree in step when a source file changes.
FOREACH(_mor_source __init__.py pymor_binding.py)
  CONFIGURE_FILE(exadg/mor/${_mor_source}
                 ${EXADG_PYTHON_PACKAGE_DIR}/exadg/mor/${_mor_source} COPYONLY)
ENDFOREACH()

INSTALL(FILES ${EXADG_PYTHON_PACKAGE_DIR}/pyproject.toml
        DESTINATION ${EXADG_PYTHON_INSTALL_DIR})
INSTALL(FILES ${EXADG_PYTHON_PACKAGE_DIR}/exadg/__init__.py
        DESTINATION ${EXADG_PYTHON_INSTALL_DIR}/exadg)
INSTALL(DIRECTORY ${EXADG_PYTHON_PACKAGE_DIR}/exadg/mor
        DESTINATION ${EXADG_PYTHON_INSTALL_DIR}/exadg)

MESSAGE(STATUS "Python bindings enabled; package staged in ${EXADG_PYTHON_PACKAGE_DIR}")
