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

"""Ready-made pyMOR models over ExaDG full-order models.

One module per pyMOR ``Model`` type, not per application. What distinguishes these is
mathematical structure -- stationary and linear, a saddle point, a Newton iteration, a time
integrator -- and pyMOR's own taxonomy of models is small and closed, so this package is too. Two
applications sharing a structure share a module; one application with two structures uses two.

Each factory reads everything it needs from the model's declared structure, so a new application
costs a ``python_bindings.cpp`` and nothing here. Modelling choices that a discretisation cannot
know -- what the parameters are called, how a coefficient depends on them -- are arguments with
defaults rather than compiled-in facts.
"""
