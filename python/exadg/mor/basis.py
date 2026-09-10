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

"""A basis built from trajectories that are never all held at once, and the coefficients with it.

pyMOR's ``inc_hapod`` already compresses a stream of snapshots under a certified bound. What it
does not do is tell you where each snapshot *went*: it returns modes and singular values and
discards the coefficients. For a proper orthogonal decomposition that is the right choice, and for
hyper-reduction it is the whole problem.

**Why the coefficients are the problem.** ECSW trains on ``a_i = V^T M u_i`` -- where the reduced
model will actually be evaluated -- and ``V`` is not known until every trajectory has been seen.
So the obvious ordering needs the snapshots twice: once to build the basis, once to project onto
it. At 24 parameters by 641 BDF-2 levels and refinement 6 that is 9 GB to keep, or a second pass
of full-order solves to avoid keeping it. Neither is acceptable.

**Why it is avoidable.** Projection is linear. If a snapshot is already known as ``u = W c`` in
some basis ``W``, then its coefficients in any later basis are

    c' = (V^T M W) c

-- a small matrix times a small vector, with no full-order object anywhere. So the coefficients
can be *carried* through every basis update instead of being recomputed from snapshots that no
longer exist. That is all this module does.

**What it costs.** The remap is exact; the basis update it follows is not. ``V`` keeps only what
survived its threshold, so each compression discards a little and the losses compound over the
updates. Measured on one trajectory of 129 levels at a local tolerance of 1e-6, going from one
update to 129: peak storage falls from 258 vectors to 12, and the error against the snapshots
grows from 5.5e-07 to 3.9e-06. Seven times worse for twenty-one times less memory -- and the error
stays at the order of the tolerance asked for, so one extra decade (about five more modes) buys it
back several times over. Compress often and ask for more accuracy than you need.

The tolerance split across levels is pyMOR's, from :cite:`HLR18` -- see :func:`_level_tolerances`.
"""

import numpy as np
from pymor.algorithms.hapod import inc_hapod_tree
from pymor.algorithms.pod import pod
from pymor.vectorarrays.constructions import cat_arrays


def _level_tolerances(steps, eps, omega):
    """The per-level l2 budget, as ``pymor.algorithms.hapod.std_local_eps`` sets it.

    Reproduced rather than called because that function wants tree nodes and this walks a chain,
    but the formula is the same one and the constants are what make the overall bound hold: the
    root gets ``omega`` of the budget and the intermediate levels share ``sqrt(1 - omega^2)``.

    Returns ``(inner, root)``, each a function of how many snapshots have been seen so far.
    """
    levels = inc_hapod_tree(steps).depth - 1

    def root(seen):
        return np.sqrt(seen) * omega * eps

    if levels <= 1:
        return (lambda seen: 0.0), root

    def inner(seen):
        return np.sqrt(seen) / np.sqrt(levels - 1) * np.sqrt(1.0 - omega**2) * eps

    return inner, root


def carried_incremental_pod(chunks, steps, products, eps, omega=0.9):
    """Per-field bases for one trajectory, with every level's coefficients carried along.

    Several fields at once, because one pass over a trajectory has to serve all of them: a saddle
    point has a velocity and a pressure, they compress separately, and solving twice to build two
    bases would defeat the point.

    Args:
        chunks: Iterable of exactly ``steps`` items, each a tuple of one ``VectorArray`` per
            field -- consecutive pieces of one trajectory. Consumed lazily, so a generator that
            computes them on demand is never asked to hold more than one.
        steps: How many chunks there will be. Needed in advance, because the error budget is
            split across the levels before the first one is seen.
        products: One inner product per field.
        eps: One absolute l2-mean error per field; they have different scales.
        omega: Balance between the intermediate levels and the last one, in ``(0, 1)``.

    Returns:
        One ``(basis, singular_values, coefficients)`` per field, coefficients of shape
        ``(n_levels, r)``. Peak storage is one chunk plus the running bases, never the trajectory.
    """
    assert steps >= 1
    assert 0.0 < omega < 1.0
    assert len(products) == len(eps)

    budgets = [_level_tolerances(steps, e, omega) for e in eps]
    state = [{"basis": None, "svals": None, "carried": []} for _ in products]
    seen = 0

    for step, chunk in enumerate(chunks):
        last = step == steps - 1
        seen += len(chunk[0])

        for field, (block, product, (inner, root)) in enumerate(zip(chunk, products, budgets)):
            current = state[field]
            basis, svals = current["basis"], current["svals"]

            # The running basis re-enters weighted by its singular values, which is how much
            # energy each of its modes stands for. Unweighted, the next compression treats a
            # trajectory's last mode as it treats its first -- measured once by accident, and the
            # basis inflated from 7 modes to 38.
            stacked = (
                block.copy() if basis is None
                else cat_arrays([basis.lincomb(np.diag(svals)), block.copy()])
            )

            updated, new_svals = pod(
                stacked, atol=0.0, rtol=0.0, l2_err=root(seen) if last else inner(seen),
                product=product, orth_tol=None if last else np.inf,
            )

            # Everything already seen moves to the new basis without being looked at again. Exact
            # for what the update kept; what it dropped is gone, and that is the compression
            # rather than the map.
            if basis is not None and current["carried"]:
                remap = product.apply2(updated, basis)
                current["carried"] = [remap @ c for c in current["carried"]]

            current["carried"].extend(product.apply2(updated, block).T)
            current["basis"], current["svals"] = updated, new_svals

    return [(c["basis"], c["svals"], np.array(c["carried"])) for c in state]


class CompressedSnapshots:
    """Every snapshot of one field, as coefficients in a per-trajectory basis rather than itself.

    What survives a streaming run: a handful of full-order vectors per trajectory, and one small
    coefficient vector per level. Enough to project every snapshot onto **any** basis afterwards,
    which is the whole point -- the basis that matters is not known while the trajectories are
    being computed.

    Args:
        bases: One ``VectorArray`` per trajectory.
        coefficients: One ``(n_levels, m_k)`` array per trajectory, in the matching basis.
    """

    def __init__(self, bases, coefficients):
        self.bases = list(bases)
        self.coefficients = list(coefficients)

    def __len__(self):
        return sum(len(c) for c in self.coefficients)

    @property
    def n_vectors(self):
        """Full-order vectors held -- what this cost, against one per level for keeping them."""
        return sum(len(basis) for basis in self.bases)

    def project(self, basis, product):
        """Coefficients of every level on ``basis``: ``(basis^T M W_k) c``, snapshot-free.

        Exact for whatever the compression kept, which is why the local tolerance is the knob
        that matters. Note this projects onto the basis it is *given* -- for hyper-reduction that
        must be the **enriched** velocity basis, supremizers included, or the training points sit
        in a subspace the reduced model does not live in. Which is also why the compressed form is
        kept rather than coefficients on some earlier basis: the supremizers are directions no
        velocity POD contains, so a projection made too early could not be extended to them.
        """
        return np.vstack([
            (product.apply2(basis, local) @ carried.T).T
            for local, carried in zip(self.bases, self.coefficients)
        ])


def streaming_basis(trajectories, steps, products, eps, omega=0.9):
    """Global per-field bases over many trajectories, plus what projects onto anything else.

    The two-level form of :func:`carried_incremental_pod`: each trajectory is compressed as it
    arrives and only its bases and coefficients are kept, then those bases are compressed into
    one per field.

    Args:
        trajectories: Iterable of ``(chunks, n_chunks)`` pairs, one per trajectory, where
            ``chunks`` yields tuples of one array per field. Consumed lazily, so a generator that
            solves on demand never holds two.
        steps: How many trajectories there will be.
        products: One inner product per field.
        eps: One absolute l2-mean error per field.
        omega: Budget split, as above.

    Returns:
        One ``(basis, compressed)`` pair per field.
    """
    budgets = [_level_tolerances(steps, e, omega) for e in eps]

    held = [[] for _ in products]
    seen = 0

    for chunks, n_chunks in trajectories:
        local_eps = [
            inner(1) if steps > 1 else root(1) for inner, root in budgets
        ]
        results = carried_incremental_pod(chunks, n_chunks, products, local_eps, omega)
        seen += len(results[0][2])

        for field, result in enumerate(results):
            held[field].append(result)

    combined = []
    for field, (product, (_, root)) in enumerate(zip(products, budgets)):
        stacked = cat_arrays([b.lincomb(np.diag(s)) for b, s, _ in held[field]])
        basis, _ = pod(stacked, atol=0.0, rtol=0.0, l2_err=root(seen), product=product)

        combined.append((
            basis,
            CompressedSnapshots([b for b, _, _ in held[field]],
                                [c for _, _, c in held[field]]),
        ))

    return combined


def trajectory_chunks(model, mu, chunk=1, blocks=(0, 1)):
    """Levels of one trajectory as the time stepper produces them, in chunks, split by block.

    ``model.solve(mu)`` materialises the whole trajectory, which is the thing being avoided. The
    stepper's ``iterate`` is a generator, so driving it directly is what makes the peak one chunk
    rather than one trajectory.

    Yields tuples of one ``VectorArray`` per entry of ``blocks``, each of at most ``chunk``
    consecutive levels -- the shape :func:`carried_incremental_pod` consumes.
    """
    from pymor.operators.constructions import IdentityOperator, ZeroOperator

    iterator = model.time_stepper.iterate(
        0.0,
        model.T,
        model.initial_data.as_range_array(mu),
        model.operator,
        rhs=None if isinstance(model.rhs, ZeroOperator) else model.rhs,
        mass=None if isinstance(model.mass, IdentityOperator) else model.mass,
        mu=mu,
        num_values=model.num_values,
    )

    batch = []
    for level, _ in iterator:
        batch.append(level)
        if len(batch) == chunk:
            yield _split(batch, blocks)
            batch = []

    if batch:
        yield _split(batch, blocks)


def _split(batch, blocks):
    joined = cat_arrays(batch)

    return tuple(joined.blocks[index] for index in blocks)


def chunk_count(levels, chunk):
    """How many chunks ``trajectory_chunks`` will yield, which the tolerance split needs first."""
    return -(-levels // chunk)
