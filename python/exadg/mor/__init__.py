#  ______________________________________________________________________
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
#  ______________________________________________________________________

"""pyMOR bindings for ExaDG full-order models.

Two layers live here, and the split is what keeps an application from needing Python at all.

:mod:`exadg.mor.binding` dresses ExaDG's vectors and operators in pyMOR's interfaces. It is
physics-free: every class wraps one of the abstract types of ``exadg/pymor/interface.h``, so the
same wrappers serve every application. Degrees of freedom never enter Python.

:mod:`exadg.mor.models` assembles those into pyMOR ``Model`` objects, one module per model type
rather than per application. A new application therefore costs a ``python_bindings.cpp`` and
nothing here.

Neither is re-exported, so importing :mod:`exadg.mor` does not pull in pyMOR.

Why this sits in ExaDG rather than in a solver-agnostic package: the binding tracks the C++
interface one-for-one, and the two sides change together. Exposing the linear solve to pyMOR, for
instance, meant declaring ``has_inverse`` in C++ and attaching a ``Solver`` in Python in the same
change. Keeping them in one repository means they are versioned and reviewed together.

All reduction -- basis generation, projection, hyper-reduction, error estimation -- is pyMOR's;
nothing in this package reimplements any of it.
"""
