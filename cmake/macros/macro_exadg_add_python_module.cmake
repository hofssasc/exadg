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

# Builds a pybind11 extension module and stages it inside the "exadg" Python package in the
# build tree, so that the result is a normal installable package rather than something that
# needs PYTHONPATH.
#
#   EXADG_ADD_PYTHON_MODULE(<module_name> <source>)
#
# produces exadg/<module_name> importable as "from exadg import <module_name>".
#
# DEAL_II_SETUP_TARGET() is deliberately not used: it calls target_link_libraries() with the
# plain signature whereas pybind11_add_module() has already used the keyword signature, and
# CMake forbids mixing the two on one target.
#
# The macro is not needed either. It is a compatibility shim; since deal.II exports imported
# targets, linking dealii::dealii is the supported modern usage (see the comment above the
# target configuration in deal.IIConfig.cmake). That target carries the whole configuration as
# usage requirements -- include directories, compile definitions, cxx_std_17, the CXX flags
# including -march=native, the per-configuration optimization flags, the linker options, and
# the debug/release library selected by a $<CONFIG:> generator expression. Nothing is copied
# out of DEAL_II_* variables here, so nothing goes stale when deal.II changes.
#
# Link dealii::dealii, NOT dealii::dealii_release. The flavour targets carry includes,
# definitions and cxx_std_17 but no INTERFACE_COMPILE_OPTIONS, so -march=native is absent and
# the build dies in vectorization.h with "Mismatch in vectorization capabilities: AVX was
# detected during configuration of deal.II ... but it is apparently not available for the file
# you are trying to compile". deal.II hard-codes the vectorization width detected at its own
# configure time, so this is an error rather than a slower build.

MACRO(EXADG_ADD_PYTHON_MODULE MODULE_NAME SOURCE_FILE)

    # NO_EXTRAS disables pybind11's link-time optimization. It is not a preference: deal.II
    # carries -fuse-ld=lld in its INTERFACE_LINK_OPTIONS, and lld does not load GCC's LTO
    # plugin, so the two together link an object that holds only LTO IR. The link reports
    # success, the .so is produced, and it exports no PyInit_<module> symbol -- the failure
    # surfaces much later as "dynamic module does not define module export function".
    #
    # Setting INTERPROCEDURAL_OPTIMIZATION OFF does not help: pybind11 adds -flto through its
    # own pybind11::lto interface target rather than through CMake's IPO property.
    pybind11_add_module(${MODULE_NAME} ${SOURCE_FILE} NO_EXTRAS)

    SET_TARGET_PROPERTIES(${MODULE_NAME} PROPERTIES
        OUTPUT_NAME ${MODULE_NAME}
        LIBRARY_OUTPUT_DIRECTORY ${EXADG_PYTHON_PACKAGE_DIR}/exadg)

    IF (TARGET exadg)
      TARGET_LINK_LIBRARIES(${MODULE_NAME} PRIVATE exadg dealii::dealii)
    ELSE()
      TARGET_LINK_LIBRARIES(${MODULE_NAME} PRIVATE EXADG::exadg dealii::dealii)
    ENDIF()

    INSTALL(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${EXADG_PYTHON_INSTALL_DIR}/exadg)

ENDMACRO()
