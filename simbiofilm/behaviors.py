"""
behaviors.py

Contains classes to run core simulation behaviors
"""
import csv
import os
from collections import OrderedDict
from random import choice, choices, random

import numba
import numpy as np

from fipy import CellVariable, DiffusionTerm, ImplicitSourceTerm
from fipy.matrices.scipyMatrix import _ScipyMeshMatrix
from fipy.solvers import LinearCGSSolver, LinearGMRESSolver, LinearPCGSolver

from scipy.sparse.csgraph import dijkstra

from .containers import Matrix, ParticleContainer
from .core import Behavior, biofilm_edge, count, to_grid, to_list
# import skfmm


class MonodRateReaction(object):
    """Rate reaction object for diffusion.

    Parameters
    ----------

    solute : Solute (see containers.py for documentation)
        Container for utilized solute

    bacteria : Bacteria (see containers.py for documentation)
        Container for bacteria utilizing the solute


    """

    def __init__(self, solute, bacteria):
        self.bacteria = bacteria
        self.solute = solute

    def utilization_rate(self):
        S = self.solute.value.reshape(-1)[self.bacteria.location]
        return (
            self.bacteria.mu
            * self.bacteria.yield_s
            * S
            / (S + self.solute.P.k / self.solute.P.max)
        )

    def uptake_rate(self):
        # TODO: generalize to_grid to enable this more easily
        coeff = np.zeros(self.solute.space.shape).reshape(-1)
        np.add.at(
            coeff.reshape(-1),
            self.bacteria.location,
            self.bacteria.mass * self.bacteria.mu,
        )
        return coeff


class Biomass_growth(Behavior):
    """Grows each container in activeContainers according to a corresponding MonodRateReaction
    See core.py for Behavior documentation

    Parameters
    ----------

    rateReactions : list of MonodRateReactions
        Corresponding to the growth of each container in activeContainers.
        rateReactions and activeContainers must be the same length, otherwise will raise error

    """

    def __init__(self, rateReactions, log=False, **kwargs):
        super().__init__(rateReactions=rateReactions, log=log, **kwargs)

    def initialize(self, sim):
        na = len(self.activeContainers)
        nr = len(self.rateReactions)
        if na != nr:
            msg = f"Number of active containers {na} != number of rate reactions {nr}"
            raise ValueError(msg)
        if not hasattr(self, "dt_nlayers"):
            setattr(self, "dt_nlayers", 2)

    def __call__(self, dt):
        """Perform biomass growth on container at rate.utilization_rate()."""
        for c, r in zip(self.activeContainers, self.rateReactions):
            dXdt = r.utilization_rate() * dt * c.mass
            c.mass += dXdt

    def calculate_dt(self):
        vals = []
        dvN = self.space.dV * np.prod(self.space.shape[1:])
        for c, r in zip(self.activeContainers, self.rateReactions):
            if c.n == 0:
                vals.append(np.inf)
                continue
            maxmass = c.P["density"] * dvN * self.dt_nlayers
            vals.append(maxmass / np.sum(r.utilization_rate() * c.mass))
        return min(vals)


class Bacterial_division(Behavior):
    """Divides bacteria in activeContainers when their division_mass is reached
    and adds the new individuals

    See core.py for Behavior documentation
    """

    def initialize(self, sim):
        self._divisions = {c.name: 0 for c in self.activeContainers}

    def __call__(self, dt):
        """Divide bacteria that have reached their division_mass."""
        self._divisions = {c.name: 0 for c in self.activeContainers}
        for container in self.activeContainers:  # TODO multi-container iterators
            n = container.n
            for individual in container:
                if individual.mass > individual.division_mass:
                    individual.mass = individual.mass / 2
                    clone = container.clone(individual)
                    self._divisions[container.name] += 1

    def summarize(self):
        msg, dat = [], {}

        for cname, divisions in self._divisions.items():
            msg.append(f'{cname} divisions: {divisions}')
            dat[f'{cname}_divisions'] = divisions

        return ', '.join(msg), dat


class Erode_biomass(Behavior):
    """Erodes the edge of the biomass, following the method described in 'A
    general description of detachment for multidimensional modelling of biofilms'
    (Xavier 2005 Biotechnology and Bioenginnering), reduce biomass.
    Conceptually, there is a liquid flow off biomass that removes biomass over time, scaling with
    the square of the height of the biofilm.

    Parameters
    ----------

    cutoff : float
        [0,1], defaults to .1

    """

    def __init__(self, rate, log=False, **kwargs):
        super().__init__(rate=rate, log=log, **kwargs)

    def initialize(self, sim):
        self.cutoff = 0.1 if not hasattr(self, "cutoff") else self.cutoff
        self.cutoff = int(sim.space.shape[0] * (1 - self.cutoff))

    def __call__(self, dt):
        biomass = to_grid(self.biomassContainers, "mass")

        # TODO: 'top' should be from space
        zeros = biofilm_edge(self.space, biomass)

        # FIXME what is this math?
        erosion_force = (
            np.square(self.space.meshgrid[0] + self.space.dl)
            * self.rate
            / self.space.dl
        )
        time_to_erode = np.array(
            skfmm.travel_time(
                zeros, erosion_force, dx=1, periodic=self.space.periodic, order=1
            )
        )
        # TODO: use higher order. skfmm has a bug, #18

        time_to_erode[self.cutoff :] = 0
        zeros = time_to_erode == 0
        if np.any(zeros):
            time_to_erode[zeros] = dt / 1e10
        shrunken_volume = np.exp(-dt / time_to_erode).reshape(-1)
        for c in self.biomassContainers:
            c.mass *= shrunken_volume[c.location]
            c.multi_remove(c.mass < (c.division_mass / 4))


def _biofilm_edge(space, biomass, substratum=False):
    # TODO: this should be what biofilm_edge provides?
    edge = biofilm_edge(space, biomass)  # want 1 on edge, 0 elsewhere.
    is_edge = np.zeros_like(edge, dtype=bool)
    is_edge[(edge + np.roll(edge, -1, axis=0)) == 0] = True  # Top
    for i in range(1, space.dimensions):  # other sides
        is_edge[(edge + np.roll(edge, -1, axis=i)) == 0] = True
        is_edge[(edge + np.roll(edge, 1, axis=i)) == 0] = True
    is_edge[biomass == 0] = False
    if substratum:
        is_edge[0][(biomass[0] == 0) & (edge[0] == -1)] = True
    return is_edge


class Erode_individuals(Erode_biomass):
    """Requires: rate. Optional: cutoff (0-1)"""

    def __init__(self, erosionRate, patch=None, log=False, **kwargs):
        super().__init__(rate=erosionRate, patch=patch, log=log, **kwargs)

    def __call__(self, dt):
        biomass = to_grid(self.biomassContainers, "mass")
        is_edge = _biofilm_edge(self.space, biomass).reshape(-1)

        erosionRate = np.square(self.space.meshgrid[0] + self.space.dl) * self.rate
        erosionRate = erosionRate.reshape(-1)
        for container in self.biomassContainers:
            eroded = is_edge[container.location] & (
                np.random.rand(container.n)
                < 1 - np.exp(-erosionRate[container.location] * dt)
            )
            if self.patch:
                for ind in container.iter(eroded):
                    self.patch.disperse(container, ind, dt)
            container.multi_remove(eroded)


class Detach_biomass(Behavior):
    """Removes unattached biomass. Method described by Xavier 2005"""

    def __init__(self, patch=None, log=False, **kwargs):
        super().__init__(patch=patch, log=log, **kwargs)

    def __call__(self, dt):
        reachable = self.space.breadth_first_search(to_grid(self.biomassContainers, "mass"))
        self.isdetached = (reachable != -1).reshape(-1)

        if self.patch:
            for container in self.activeContainers:
                for ind in container.iter(self.isdetached[container.location]):
                    self.patch.disperse(container, ind, dt)

        for container in self.biomassContainers:
            if container not in self.phageContainers:
                container.multi_remove(self.isdetached[container.location])

    def post_iteration(self, dt):  # post output, for phage visualization
        for c in self.phageContainers:
            c.multi_remove(self.isdetached[c.location])

    def well_mixed(self, dt):
        maxvol = 0.9 if not hasattr(self, "maxvol") else self.maxvol
        volumes = [c.volume.sum() for c in self.biomassContainers]
        total = np.sum(volumes)
        if total <= maxvol:
            return

        avg_volume = [np.average(c.volume) for c in self.biomassContainers]
        volume_tolose = volumes / total * (total - maxvol)
        ntolose = [
            ((v / avg) if avg > 0 else 0) for v, avg in zip(volume_tolose, avg_volume)
        ]
        ntolose = np.ceil(ntolose).astype(int)
        for n, container in zip(ntolose, self.biomassContainers):
            if n == 0:
                continue
            removals = np.zeros(container.n, dtype=bool)
            removals[np.random.choice(container.n, n, replace=False)] = True
            container.multi_remove(removals)


class Lysis(Behavior):
    """Reduces 'incubation time' of infected cells and lyses those which have no time left.
    Phage 'remainder' is set to the fraction of the remaining time of dt, for use in subsequent
    phage behaviors.

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    phage : Phage (see containers.py for documentation)
        Infecting phage

    infected : InfectedBacteria (see containers.py for documentation)
        Container of infected bacteria

    """

    def __init__(self, pairs, log=False, **kwargs):
        super().__init__(pairs={v: k for k, v in pairs.items()}, log=log, **kwargs)
        self._produced = {phage.name: 0 for phage in self.pairs.values()}
        self._lysed = {infected.name: 0 for infected in self.pairs.keys()}

    def __call__(self, dt):
        """Lyse cells which have passed their incubation_time."""
        self._produced = {phage.name: 0 for phage in self.pairs.values()}
        self._lysed = {infected.name: 0 for infected in self.pairs.keys()}
        for infected, phage in self.pairs.items():
            if infected.n == 0:
                continue
            infected.incubation_time -= dt

            for individual in infected:
                if individual.incubation_time <= 0:
                    params = {}
                    for name in individual.dtype.names:
                        if "phage_" in name:
                            params[name[6:]] = getattr(individual, name)
                        params["remainder"] = -individual.incubation_time / dt

                    # TODO: int parameters
                    phage.add_multiple_individuals(
                        int(individual.phage_burst), individual.location, params
                    )
                    # TODO: multi_remove
                    self._produced[phage.name] += individual.phage_burst
                    self._lysed[infected.name] += 1
                    infected.remove(individual)
            phage.remainder = np.clip(phage.remainder, 0, None)

    def calculate_dt(self):
        return np.inf
        # return np.inf if self.infected.n == 0 else np.min(self.infected.incubation_time)

    def summarize(self):
        msg, dat = [], {}
        for phg, produced in self._produced.items():
            msg.append(f'{phg} produced: {produced}')
            dat[f'{phg}_produced'] = produced

        for inf, lysed in self._lysed.items():
            msg.append(f'{inf} lysed: {lysed}')
            dat[f'{inf}_lysed'] = lysed

        return ', '.join(msg), dat



class Phage_interaction(Behavior):
    """Calculates and performs interactions between phage and other particles.
    Non active or non infectable containers default to inert interactions.

    Parameters
    ----------

    infectedContainer : InfectedBacteria (see containers.py for documentation)
        Container of bacteria infected by the phage

    """

    def __init__(self, pairs, log=False, **kwargs):
        super().__init__(pairs={v: k for k, v in pairs.items()}, log=log, **kwargs)
        self._infected = {phage.name: 0 for phage in self.pairs.values()}

    def _build_biomass(self, containers):
        """Generates
        {location: [(container, particle, volume_p), (c, p, v) ...], location: [...]}."""
        biomass = {}
        for container in containers:
            rate = container.impedance * container.mass
            for particle, rate in zip(container, rate):
                if particle.location in biomass:
                    biomass[particle.location].append((container, particle, rate))
                else:
                    biomass[particle.location] = [(container, particle, rate)]
        return biomass

    def __call__(self, dt):
        """Infect susceptible bacteria, without resistance."""
        self._infected = {phage.name: 0 for phage in self.pairs.values()}
        containers = [c for c in self.biomassContainers if c not in self.phageContainers]
        biomass = self._build_biomass(containers)

        bremovals = {c: [] for c in containers}

        for infected, phage in self.pairs.items():
            interactions = [
                choices(population=biomass[loc], weights=[v for c, p, v in biomass[loc]])[0]
                if loc in biomass
                else (False, False, False)
                for loc in phage.location
            ]

            adsorptions = np.random.rand(phage.n) < (
                1 - np.exp(-phage.adsorption_rate * phage.remainder * dt)
            )

            removals = np.zeros(phage.n, dtype=bool)
            # FIXME: use a proper iterator  manager to iterate and remove particles
            for (i, phg), (c, bac, _), ads in zip(
                enumerate(phage), interactions, adsorptions
            ):
                if not c or not ads:  # objects are true, False is not.
                    continue
                bac_infected, phg_removed = c.phage_interacted(bac, phg)
                if bac_infected and not phg_removed:
                    msg = "Infected bacteria must remove phage.\n"
                    msg +=  'Check phage_interacted implementation and adsorbable parameters'
                    raise RuntimeError(msg)
                # Multiple phage can infect the same thing, so have to make sure its not removed yet
                if bac_infected and bac.id not in bremovals[c]:
                    bremovals[c].append(bac.id)
                    infected.infect(bac, phg)
                    self._infected[phage.name] += 1
                removals[i] = phg_removed

            phage.multi_remove(removals)

        for container, ids in bremovals.items():
            for id in ids:
                container.remove(container.with_id(id))


    def summarize(self):
        msg, dat = [], {}

        for phg, infected in self._infected.items():
            msg.append(f'{phg} infected: {infected}')
            dat[f'{phg}_infected'] = infected

        return ', '.join(msg), dat


class Cull_underpopulated(Behavior):
    """Remove cells on the substratum if they have no neighbors and volume < Vmin

    Parameters
    ----------

    space : Space (see core.py for documentation)
         Space object for the simulation

    vmin : float
        Minimum volume (0-1)

    """

    def __init__(self, vmin = 0.5, **kwargs):
        self.vmin = vmin
        super().__init__(**kwargs)

    def __call__(self, dt):

        volume = to_grid(self.biomassContainers, "volume")
        cull = []
        for ix in np.where(volume[0] < self.vmin)[0]:
            if (volume[0][nix] > 0 for nix in self.space.neighbors[ix]):
                continue
            cull.append(ix)

        for c in self.biomassContainers:
            c.multi_remove(np.isin(c.location, cull))


class Phage_randomwalk(Behavior):
    """Calculates the brownian motion of a phage, modeled as a random walk.

    Parameters
    ----------

    space : Space (see core.py for documentation)
         Space object for the simulation

    phage : Phage (see containers.py for documentation)
        Phage container for phage of interest

    rate : float
        Rate at which to remove phages

    adjacency : np.array
        Adjacency matrix for the biomass
    """

    def __init__(self, removalRate, log=False, **kwargs):
        super().__init__(rate=removalRate, log=log, **kwargs)

    def initialize(self, sim):
        self.adjacency = self.space.construct_neighbors()

    def __call__(self, dt):
        if sum([p.n for p in self.phageContainers]) == 0:
            return

        # Setup biomass for impedance
        containers = [c for c in self.biomassContainers if 'impedance' in c.P]
        biomass = to_grid(containers, "mass")
        mass_pts = np.ravel_multi_index(np.where((biomass != 0)), self.space.shape)

        # Calc distance away from biomass for advection
        distance = np.array(
            dijkstra(self.adjacency, indices=mass_pts, min_only=True), dtype=float,
        ).reshape(self.space.shape)
        distance = np.multiply(distance, self.space.dl)
        distance = distance ** 2
        distance[distance < 0] = 0

        # walk
        for phage in self.phageContainers:
            if phage.n == 0:
                continue
            step_dt = (2 * self.space.dl * self.space.dl) / phage.diffusivity
            remove = 1 - np.exp(-self.rate * distance * step_dt[0])  # TODO remove [0]
            nsteps = dt / step_dt * phage.remainder

            zps = np.zeros(self.space.shape).reshape(-1)
            for cntnr in containers:
                np.add.at(zps, cntnr.location, cntnr.impedance * cntnr.mass)
            zps = zps.reshape(self.space.shape) / self.space.dV
            location = np.array(np.unravel_index(phage.location, self.space.shape)).T

            args = [location, nsteps, step_dt, zps, np.array(self.space.shape), remove]
            if self.space.dimensions == 2:
                location, remainder = _rw_stuck_2D(*args)
            elif self.space.dimensions == 3:
                location, remainder = _rw_stuck_3D(*args)

            remainder[nsteps > 0] = remainder[nsteps > 0] / nsteps[nsteps > 0]
            phage.location = np.ravel_multi_index(location.T, self.space.shape)
            phage.remainder = remainder

    def calculate_dt(self):
        return np.inf
        # return np.inf if self.phage.n == 0 else np.min(self.phage.incubation_period / 2)


@numba.njit(parallel=True)
def _rw_stuck_2D(location, nsteps, step_dt, zps, shape, premoval):
    remainder = np.zeros_like(nsteps)
    for i in numba.prange(location.shape[0]):
        loc = location[i]
        for step in range(nsteps[i]):

            # sp.special.erf(1/(np.sqrt(2 * np.pi))) for 2D
            if np.random.rand() < 0.42737488393046696:
                continue

            if np.random.rand() < premoval[loc[0], loc[1]]:
                break

            loc_ = loc.copy()
            loc_[np.random.randint(2)] += -1 if np.random.rand() < 0.5 else 1

            # check for boundaries. Bump into wall and fail to move.
            if loc_[0] == shape[0] or loc_[0] == -1:
                continue

            # check for periodicity
            # FIXME: pass in periodicity
            for dim in range(1, len(loc_)):
                if loc_[dim] == shape[dim] or loc_[dim] == -1:
                    loc_[dim] = loc_[dim] % shape[dim]

            zp = zps[loc[0], loc[1]] + zps[loc_[0], loc_[1]]
            P = 1 - np.exp(-zp * step_dt[i])
            if np.random.rand() < P:  # interact
                break

            # make the move!
            loc = loc_
        remainder[i] = nsteps[i] - step

        location[i] = loc
    return (location, remainder)


@numba.njit(parallel=True)
def _rw_stuck_3D(location, nsteps, step_dt, zps, shape, premoval):
    remainder = np.zeros_like(nsteps)
    for i in numba.prange(location.shape[0]):
        loc = location[i]
        for step in range(nsteps[i]):

            # sp.special.erf(1/(2 * np.sqrt(2 * np.pi))) for 3D
            if np.random.rand() < 0.22212917330884568:
                continue

            if np.random.rand() < premoval[loc[0], loc[1], loc[2]]:
                break

            loc_ = loc.copy()
            loc_[np.random.randint(3)] += -1 if np.random.rand() < 0.5 else 1

            # check for boundaries. Bump into wall and fail to move.
            if loc_[0] == shape[0] or loc_[0] == -1:
                continue

            # check for periodicity
            # FIXME: pass in periodicity
            for dim in range(1, len(loc_)):
                if loc_[dim] == shape[dim] or loc_[dim] == -1:
                    loc_[dim] = loc_[dim] % shape[dim]

            zp = zps[loc[0], loc[1], loc[2]] + zps[loc_[0], loc_[1], loc_[2]]
            P = 1 - np.exp(-zp * step_dt[i])
            if np.random.rand() < P:  # interact
                break

            # make the move!
            loc = loc_
        remainder[i] = nsteps[i] - step

        location[i] = loc
    return (location, remainder)


@numba.njit
def _distance(p1, p2, shape):
    total = 0
    for i, (a, b) in enumerate(zip(p1, p2)):
        delta = abs(b - a)
        if i == 0:  # no periodic y
            total += delta ** 2
            continue
        if delta > shape[i] - delta:
            delta = shape[i] - delta
        total += delta ** 2
    return np.sqrt(total)


class Relax_line_shove(Behavior):
    """Expands the biomass according to rules in Simmons 2017 (ISME), shoving
    only biomass whose total volume is greater than the cube can hold."""

    def __call__(self, dt):
        """Shove overfull grid points along the path to the nearest empty point."""
        if sum([c.n for c in self.biomassContainers]) == 0:
            return
        maxvol = 1  # packing ratio?

        location, volume = _unpack_containers(
            self.biomassContainers, ["location", "volume"]
        )
        total_volume = np.zeros(self.space.shape)
        np.add.at(total_volume.reshape(-1), location, volume)

        while True:
            updates = list(_get_updates(total_volume.reshape(-1), maxvol))
            shoved = set()
            if not updates:
                break

            while updates:
                ix = updates.pop()
                path = _get_shove_path(total_volume, ix)

                if any(ix in shoved for ix in path):
                    break

                shoved.update(path)
                layer_size = np.prod(self.space.shape[1:])
                _shove_along_path(
                    path, location, volume, total_volume.reshape(-1), layer_size
                )

        _repack_containers(location, self.biomassContainers)


def _unpack_containers(biomassContainers, params):
    return [
        np.concatenate([c[param] for c in biomassContainers if c.n > 0])
        for param in params
    ]


@numba.njit(parallel=True)
def _get_updates(volumes, maxvol):
    return np.where(volumes > maxvol)[0]


def _get_shove_path(volumes, ix):
    loc = list(np.unravel_index(ix, volumes.shape))
    if len(loc) == 2:
        candidates = _get_candidates_2d(volumes, loc)
    if len(loc) == 3:
        candidates = _get_candidates_3d(volumes, loc)
    target = list(candidates[np.random.randint(len(candidates))])
    shift = []
    for i in range(1, len(volumes.shape)):
        shift.append(volumes.shape[i] // 2 - loc[i])
        loc[i] = (loc[i] + shift[i - 1]) % volumes.shape[i]
        target[i] = (target[i] + shift[i - 1]) % volumes.shape[i]
    path = _get_line(loc, target)
    for i in range(1, len(volumes.shape)):
        path[:, i] -= shift[i - 1]
        path[:, i] %= volumes.shape[i]
    return np.ravel_multi_index(list(zip(*path)), volumes.shape)


# FIXME: should be easy to fix nD into one function
@numba.njit
def _get_candidates_2d(volumes, ix):
    yinit, xinit = ix
    explored = np.zeros_like(volumes, dtype=np.bool8)
    node = [(yinit, xinit)]

    candidates = []
    candistance = volumes.shape[0] ** 2 + volumes.shape[1] ** 2
    while node:
        _ix = node.pop(0)
        y, x = _ix
        if explored[y, x]:
            continue
        explored[y, x] = True
        if _distance(ix, _ix, volumes.shape) > candistance:
            continue

        if volumes[y, x] <= 0.01:
            candidates.append(_ix)
            candistance = _distance(ix, _ix, volumes.shape)
            continue

        for j, i in ((1, 0), (0, -1), (0, 1), (-1, 0)):
            _y, _x = y + j, x + i
            if _y >= 0 and _y < volumes.shape[0]:
                if _x == -1:
                    _x = volumes.shape[1] - 1
                elif _x == volumes.shape[1]:
                    _x = 0
                node.append((_y, _x))
    return candidates


@numba.njit
def _get_candidates_3d(volumes, ix):
    yinit, xinit, zinit = ix
    explored = np.zeros_like(volumes, dtype=np.bool8)
    node = [(yinit, xinit, zinit)]

    candidates = []
    candistance = volumes.shape[0] ** 2 + volumes.shape[1] ** 2 + volumes.shape[2] ** 2
    while node:
        _ix = node.pop(0)
        y, x, z = _ix
        if explored[y, x, z]:
            continue
        explored[y, x, z] = True
        if _distance(ix, _ix, volumes.shape) > candistance:
            continue

        if volumes[y, x, z] <= 0.01:
            candidates.append(_ix)
            candistance = _distance(ix, _ix, volumes.shape)
            continue

        for j, i, k in (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1),
        ):
            _y, _x, _z = y + j, x + i, z + k
            if _y >= 0 and _y < volumes.shape[0]:
                if _x == -1:
                    _x = volumes.shape[1] - 1
                elif _x == volumes.shape[1]:
                    _x = 0
                if _z == -1:
                    _z = volumes.shape[1] - 1
                elif _z == volumes.shape[1]:
                    _z = 0
                node.append((_y, _x, _z))
    return candidates


def _get_line(start, end):
    start = np.array(start, dtype=int)
    end = np.array(end, dtype=int)
    n_steps = np.max(np.abs(end - start)) + 1
    dim = start.size
    slope = (end - start).astype(float)
    scale = np.max(np.abs(slope))
    if scale != 0:
        slope = slope / scale
    stepmat = np.arange(n_steps).reshape((n_steps, 1)).repeat(dim, axis=1)
    return np.rint(start + slope * stepmat).astype(int)


@numba.njit
def _shove_along_path(path, location, volume, total_volume, layersize):
    for start, target in zip(path[:-1], path[1:]):
        y, yt = start // layersize, target // layersize
        at_start_indices = np.where(location == start)[0]
        np.random.shuffle(at_start_indices)  # helps to prevent bias
        for ix in at_start_indices:
            if volume[ix] == 0:
                location[ix] = target
            if total_volume[start] > 1:
                total_volume[start] -= volume[ix]
                location[ix] = target
                total_volume[target] += volume[ix]


def _repack_containers(location, biomassContainers):
    count = 0
    for container in biomassContainers:
        container.location = location[count : count + container.n]
        count += container.n


@numba.njit
def _distance(p1, p2, shape):
    total = 0
    for i, (a, b) in enumerate(zip(p1, p2)):
        delta = abs(b - a)
        if i == 0:  # no periodic y
            total += delta ** 2
            continue
        if delta > shape[i] - delta:
            delta = shape[i] - delta
        total += delta ** 2
    return np.sqrt(total)


class Relax_biofilm(Behavior):
    """Relax biofilm (if a gridpoint is overfilled, 'shove' some biomass into adjacent points)
    according to the method in Xavier 2005 (enviornmental microbiology) adapted
    for well mixed cubes instead of individuals on a continuum.
    """

    def __call__(self, dt):
        _relax_biofilm_nojit(dt, self.space, self.biomassContainers)


def _relax_biofilm_nojit(dt, space, biomassContainers):
    """Shove overfull grid points along the path to the nearest empty point."""
    # VAR_ix is VAR by the index.
    # VAR_ind is VAR by the individual.
    maxvol = 1  # packing ratio?
    if sum([c.n for c in biomassContainers]) == 0:
        return

    unpacked = _unpack_containers(biomassContainers, ["location", "volume", "adhesion"])
    location_ind, volume_ind, adhesion_ind = unpacked
    locked_ind = np.zeros_like(location_ind, dtype=bool)
    volume_ix = np.zeros(space.shape).reshape(-1)
    locked_volume_ix = np.zeros_like(volume_ix)
    np.add.at(volume_ix, location_ind, volume_ind)

    # TODO: check for max height?
    ix = np.argmax(volume_ix)

    while volume_ix[ix] > maxvol:

        # Randomly decide the priority of moving, weighted by adhesion
        ind_ix = np.where(location_ind == ix)[0]
        adhesion = adhesion_ind[ind_ix]
        prob = np.random.rand(len(ind_ix)) * adhesion / (1 + min(adhesion))
        ind_indexes = [x for _, x in sorted(zip(prob, ind_ix)) if not locked_ind[x]]

        # Find the particles at this index who will remain in this index
        while ind_indexes:
            vol = volume_ind[ind_indexes[-1]]
            if (locked_volume_ix[ix] + vol) > maxvol:
                break
            ind = ind_indexes.pop()
            locked_volume_ix[ix] += vol
            locked_ind[ind] = True

        # move the rest
        for ind in ind_indexes:
            vol = volume_ind[ind]
            next_ix, lockit = _get_overflow_target(space, ix, vol, maxvol, volume_ix)
            location_ind[ind] = next_ix
            volume_ix[ix] -= vol
            volume_ix[next_ix] += vol
            if lockit:
                locked_volume_ix[next_ix] += vol
                locked_ind[ind] = True

        ix = np.argmax(volume_ix)
    _repack_containers(location_ind, biomassContainers)


def _get_overflow_target(space, ix, vol, maxvol, volume_ix):
    candidates = sorted([(volume_ix[nix] + vol, nix) for nix in space.neighbors[ix]])
    if candidates[0][0] <= maxvol:
        return (candidates[0][1], True)
    else:
        return (choice(candidates)[1], False)


def relax_biofilm_jit(dt, space, biomassContainers):
    """Shove overfull grid points along the path to the nearest empty point."""
    # VAR_ix is VAR by the index.
    # VAR_ind is VAR by the individual.
    maxvol = 1  # packing ratio?

    unpacked = _unpack_containers(biomassContainers)
    neighbors = np.array([space.neighbors[x] for x in range(space.N)]).reshape(-1)

    start = 0
    neighbor_index = []
    for x in range(space.N):
        n = len(space.neighbors[x])
        neighbor_index.append((start, start + n))
        start += n
    neighbor_index = np.array(neighbor_index)

    # TODO: check for max height
    _flow(space.shape, maxvol, neighbors, neighbor_index, *unpacked)
    _repack_containers(unpacked[0], biomassContainers)


@numba.njit
def _flow(
    shape,
    maxvol,
    neighbors,
    neighbor_index,
    location_ind,
    volume_ind,
    adhesion_ind,
    locked_ind,
):
    # Setup
    volume_ix = np.zeros(np.prod(shape), dtype=np.float32)
    locked_volume_ix = np.zeros_like(volume_ix)
    for ix, vol in zip(location_ind, volume_ind):
        volume_ix[ix] += vol
    locked = np.zeros(len(location_ind), dtype=np.bool8)

    # While there is an index to explore
    ix = np.argmax(volume_ix)
    while volume_ix[ix] > maxvol:

        # Randomly decide the priority of moving, weighted by adhesion
        ind_ix = np.where(location_ind == ix)[0]
        adhesion = adhesion_ind[ind_ix]
        prob = np.random.rand(len(ind_ix)) * 2 / (1 + adhesion / adhesion.min())
        ind_indexes = [x for _, x in sorted(zip(prob, ind_ix)) if not locked[x]]

        neigh = neighbors[neighbor_index[ix]]

        # Find the particles at this index who will remain in this index
        while ind_indexes:
            vol = volume_ind[ind_indexes[-1]]
            if (locked_volume_ix[ix] + vol) > maxvol:
                break
            ind = ind_indexes.pop()
            locked_volume_ix[ix] += vol
            locked[ind] = True

        # move the rest
        for ind in ind_indexes:
            vol = volume_ind[ind]
            candidates = sorted([(locked_volume_ix[nix] + vol, nix) for nix in neigh])
            lockit = candidates[0][0] <= maxvol
            next_ix = candidates[0][1] if lockit else np.random.choice(candidates)[1]
            # next_ix = candidates[0][1]
            location_ind[ind] = next_ix
            volume_ix[ix] -= vol
            volume_ix[next_ix] += vol
            if lockit:
                locked_volume_ix[next_ix] += vol
                locked[ind] = True

        ix = np.argmax(volume_ix)


def _get_updates(flat_volume, maxvol):
    updates = np.where(flat_volume > maxvol)[0]
    updates = [(ix, flat_volume[ix]) for ix in updates]
    return [x[0] for x in sorted(updates, key=lambda x: 1 / x[1])]


class DiffusionReaction(Behavior):
    """Calculates the diffusion reaction with given rate reactions at steady state with
    respect to the time step of the simulation.

    Parameters
    ----------

    solute : Solute (see containers.py for documentation)
        Container of solute particles

    diffusivity : float
        Diffusivity constant for calculations

    reactions : list of MonodRateReactions
        Reactions for diffusion

    """

    def __init__(self, solute, ratereactions, tol=1e-8, log=False, **kwargs):
        super().__init__(solute=solute, reactions=ratereactions, _tolerance=tol, log=log, **kwargs)
        # where do we put the mesh and constraining?
        # mesh goes in Space. constraints go in solute.
        self.diffusivity = solute.P.diffusivity * solute.P.max
        self.solvers = [LinearGMRESSolver(), LinearCGSSolver(), LinearPCGSolver()]
        self._layer = np.prod(solute.space.shape[1:])
        self._layer_height = round(solute.P.h / solute.space.dl) + 1
        self._sr = solute.k / solute.max
        self.maxiter = 10

    def __call__(self, dt):
        if sum([c.n for c in self.activeContainers]) == 0:
            return

        _, size = self._get_size()
        default = self.solute.value.reshape(-1)[:size]
        best = {"res": 1e8, "val": default}
        for solver in self.solvers:
            success, res, val = self.solve(solver, default)
            if success:
                break
            if res < best["res"]:  # If unsuccessful, store best res, val
                best = {"res": res, "val": val}
        else:
            val = best["val"]

        self.solute.value.reshape(-1)[size:] = 1
        self.solute.value.reshape(-1)[:size] = val

    def solve(self, solver, init):
        equation, phi = self._setup_equation()
        phi.setValue(init)
        res = [1e8]
        solutions = [phi.value.copy()]
        for i in range(self.maxiter):
            res.append(equation.sweep(var=phi, solver=solver))
            solutions.append(phi.value.copy())

            if phi.value.max() > 1.001 or phi.value.min() < -0.001 or res[-1] > res[0]:
                ix = np.argmin(res[:-1])
                phi.value = solutions[ix]
                return False, res[ix], phi.value

            phi.updateOld()
            if res[-1] <= self._tolerance:
                break
        else:
            return False, res[-1], phi.value  # loop went to max iterations
        return True, res[-1], phi.value

    def _get_size(self):
        height = np.max(np.where(sum([r.uptake_rate() for r in self.reactions]) > 0)[0])
        height = np.unravel_index(height, self.space.shape)[0] + self._layer_height
        size = height * self._layer
        return height, size

    def _setup_equation(self):
        height, size = self._get_size()
        mesh = self.space.construct_mesh(height)
        variables, terms = [], []
        phi = CellVariable(name=self.solute.name, mesh=mesh, hasOld=True)
        for r in self.reactions:
            variables.append(
                CellVariable(name=f"{r.bacteria.name}_rate", mesh=mesh, value=0.0)
            )
            terms.append(ImplicitSourceTerm(coeff=(variables[-1] / (phi + self._sr))))
        equation = DiffusionTerm(coeff=self.diffusivity) - sum(terms)
        phi.constrain(1, where=mesh.facesTop)

        for var, coef in zip(
            variables, [r.uptake_rate()[:size] for r in self.reactions]
        ):
            try:
                var.setValue(coef / self.space.dV)
            except ValueError as err:
                # FIXME resize system or boundary layer or something. Also in __call__
                print("Boundary layer height greater than system size")
                raise err
        return equation, phi


# TODO: 'spatial interaction general behavior'
# tools for creation of iterable of inds at locations and neighbors
