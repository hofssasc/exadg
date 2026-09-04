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

One module lives here. :mod:`exadg.mor.pymor_binding` dresses the ExaDG vectors and operators
of the compiled extension in pyMOR's interfaces, so that pyMOR's algorithms run against the
full-order model without degrees of freedom ever entering Python. All reduction -- basis
generation, projection, hyper-reduction, error estimation -- is pyMOR's; nothing in this
package reimplements any of it.

It is imported explicitly rather than re-exported here, so that importing :mod:`exadg.mor`
does not pull in pyMOR.

Why this sits in ExaDG rather than in a solver-agnostic package: the binding tracks the C++
operator API one-for-one. Exposing the linear solve to pyMOR, for instance, meant adding
``apply_inverse_jacobian`` in C++ and a ``Solver`` in Python in the same change. Keeping the
two sides in one repository means they are versioned and reviewed together.
"""
