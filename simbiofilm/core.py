"""
core.py

Contains Base Classes and function for simbiofilm
Back-end for handling and transfering simulation information
"""

import csv
import enum
import gc
import os
import re
from collections import OrderedDict, deque
from configparser import ConfigParser
from functools import total_ordering
from random import seed
from time import time

import numba
import numpy as np
import pyamg
from fipy import Grid2D, Grid3D
from scipy import sparse as sp


class ndict(dict):
    """Named dictionary that allows ndict.item for getting, but not setting."""

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise KeyError(name)

    def copy(self):
        """Shallow copy of nD."""
        return ndict(super().copy())


def _convert(value):
    """Convert a string to what makes sense."""
    if not isinstance(value, str):
        return value
    try:  # float next
        return float(value)
    except ValueError:  # bool finally
        if value == "":
            return ""
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if (value[0] + value[-1]) == "()":
            return tuple(_convert(x) for x in value[1:-1].split(","))
    return value  # remains string if nothing else works


def _get_from_npz(filename):
    with np.load(filename) as f:
        configdata = f["config"]
        cfg = ndict()
        for name, value in configdata:
            section, item = name.split(":")
            if section not in cfg:
                cfg[section] = ndict()
            cfg[section][item] = _convert(value)
    return cfg


def cfg_from_dict(cfg_dict):
    """Makes a named dict from cfg dictionary. Converts from string to float, bool, tuples where it can.

    Parameters
    ----------

    cfg_dict : dict
        Dictionary containing the simulation configuration properties

    -------
    Returns: named dict: An equivalent named dict such that cfg.param is equivalent to cfg_dict[param]

    """
    cfg = ndict()
    for section in cfg_dict:
        if section.lower() in cfg:
            msg = f"Duplicate section: {section.lower()}. Already in cfg:\n{cfg}"
            raise KeyError(msg)
        cfg[section.lower()] = ndict()
        for option in cfg_dict[section]:
            if option in cfg[section.lower()]:
                msg = f"Duplicate option: {option.lower()}. Already in section: {section.lower()}"
                raise KeyError(msg)
            cfg[section.lower()][option.lower()] = _convert(cfg_dict[section][option])
    return cfg


def getcfg(filename):
    """From a file path, reads in a lowercase dict and returns a named cfg dict.

    Parameters
    ----------

    filename : str
        Path to a file containing simulation configuration properties

    -------
    Returns: named dict : containing configuration data from the file
    """
    if filename.endswith(".npz"):
        return _get_from_npz(filename)
    cfg = ConfigParser()
    cfg.read(filename)
    return cfg_from_dict(cfg)


def parse_params(names, values, cfg=None):
    """Parse line of values, intended to be from command line.

    names is expected to be of form:
        general:seed,substrate:max,matrix:density
    The first arg per pair is the category of the config, second is the item.

    values is a string of the form:
        1,10,22e3

    If cfg is provided (from simbiofilm.getcfg), it will overwrite or add the
    values from the param line
    """
    cfg = ndict() if cfg is None else cfg.copy()
    if isinstance(names, str):
        names = names.split(",")
    names = [n.lower() for n in names]
    if isinstance(values, str):
        values = values.split(",")
    for (name, val) in zip(names, values):
        section, item = name.split(":")
        if section not in cfg:
            cfg[section] = ndict()
        if re.match('.*\[\d+\]$', item):
            item, _ = item[:-1].split('[')  # index unnecessary
            cfg[section] = cfg[section].get(item, default=[])
            cfg[section][item].append
            if item not in cfg[section]:
                cfg[section] = []
            cfg[section].apppend(val)
        else:
            cfg[section][item] = _convert(val)
    return cfg


def count(space, containers, fill_ones=False):
    """Counts the number of particles at each grid point.

    Parameters
    ----------

    space : Space
        Space object for the simulation

    containers : list of Containers
        List of containers of particles to be counted

    -------
    Returns: np.array: Containing the number of particles at each gridpoint
    """
    containers = to_list(containers)
    total = np.zeros(space.shape, dtype=int)
    for container in containers:
        np.add.at(total.reshape(-1), container.location, 1)
    if fill_ones:  # TODO: determine if fill_ones is necessary.
        total[total == 0] = 1
    return total


def to_grid(containers, param, fill_ones=False):
    """Sum the specified particle property at each grid point.

    Parameters
    ----------

    containers : list of Containers
        List of containers of particles with properity to be summed

    param : str
        Property to be summed at each grid point

    -------
    Returns: np.array: Containing the sum of the property for all particles at each gridpoint
    """
    # TODO make this better, probably with numexpr. Do math on these per individual BEFORE doing
    # the sum
    containers = to_list(containers)
    total = np.zeros(containers[0].space.shape)
    for container in containers:
        np.add.at(total.reshape(-1), container.location, getattr(container, param))
    if fill_ones:
        total[total == 0] = 1
    return total


def to_list(container):
    """If input is a Container, return a list of the single input container"""
    if isinstance(container, Container):
        return [container]
    elif container is None:
        return []
    return container


def biofilm_edge(space, biomass):
    """Find the biofilm edge, where 1 corresponds to a gridpoint with mass, -1 corresponds
    to a gridpoint without mass and the transition between 1 and -1 is an edge.

    Parameters
    ----------

    space : Space
        Space object for the simulation
    biomass : np.array
        Sum of biomass at each grid point

    -------
    Returns: np.array: Representing the edges in the biofilm
    """
    # Returns an array where the transition from 1 to -1 is an edge.
    grid = space.meshgrid[0].copy()
    grid[biomass == 0] = 0
    height = grid.argmax(axis=0)
    if height.max() == 0:
        edge = -np.ones(space.shape)
        edge[np.where(biomass > 0)] = 1
        return edge
    height[(height == 0) & (biomass[0] == 0)] = -2
    height += 1
    nodes = (
        np.ravel_multi_index(np.where(height >= 0), space.shape[1:])
        + (height[height >= 0]) * np.prod(space.shape[1:])
    ).tolist()

    edge = np.zeros(space.shape, dtype=int)
    edge[height.max() + 1 :] = -1

    edge, biomass = edge.reshape(-1), biomass.reshape(-1)
    while nodes:
        ix = nodes.pop(0)
        for ix_n in space.neighbors[ix]:
            if biomass[ix_n] == 0 and edge[ix_n] == 0:
                edge[ix_n] = -1
                nodes.append(ix_n)
    edge[edge == 0] = 1
    return edge.reshape(space.shape)


def _make_time_condition(inftime):
    def infection_condition(space, time, *args, **kwargs):
        return time >= inftime

    return infection_condition


def _make_dt_constraint_function(value):
    def dt_constraint(*args, **kwargs):
        return value

    return dt_constraint


class Behavior:
    """Base class for Behavior. Automatically gains access to containerGroups
    once the Simulation.initialize is called. See Simulation documentation.

    Parameters
    ----------

    sim : Simulation
        The Simulation object itself

    space : Space
        Space object for the simulation

    """

    def _register(self, sim):
        for group, containers in sim.containerGroups.items():
            setattr(self, group, containers)
        self.space = sim.space
        self.sim = sim

    def initialize(self, sim):
        pass

    def __init__(self, **kwargs):
        self.log = False
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, dt):
        msg = f"{self.__class__} has no implementation for __call__"
        raise NotImplementedError(msg)

    def pre_iteration(self, dt):
        """Setup for anything (simultaneous behaviors)"""
        pass

    def post_iteration(self, dt):
        """Teardown or anything else post iteration"""
        pass

    def calculate_dt(self):
        """Calculate max dt allowed. inf is no effect."""
        return np.inf

    def well_mixed(self, dt):
        self(dt)

    def summarize(self):
        """Returns a dict of name: value to log"""
        msg = "'{}' behavior subclass has not implemented 'summarize'."
        raise NotImplementedError(msg.format(type(self).__name__))

    @staticmethod
    def verify_parameters(self, params, containers):
        raise NotImplementedError("oof")
        for c in to_list(containers):
            for p in params:
                if p not in c.P:
                    msg = f"Container {c.name} does not have required parameter '{p}' "
                    msg += f"in behavior {self.__class__}"
                    raise AttributeError(msg)


class Simulation:
    """Back-end object for handling and transferring information

    Parameters
    ----------

    all_end_conditions : list
        Conditions under which the simulation terminates

    containerGroups : dict
        Maps types of containers (str) to a list of the corresponding containers in the Simulation

    allContainers: list of Containers
        List of all Containers in the simulation

    allBehaviors: list of Behaviors
        List of all Behaviors in the simulation

    allEvents : list of Events
        List of all Events in the simulation

    space : Space
        The simulation space

    dt : float
        Timestep for the simulation

    t : float
        Current time of the simulation

    iteration : int
        Current iteration number

    initialized : bool
        True if the simulation has been initialized, false otherwise

    out_dir : str
        Output directory pathway

    config : dict
        Configuration dict

    """

    def __init__(self, space, cfg):
        self.all_end_conditions = []
        self.containerGroups = {  # Default groups:
            "activeContainers": [],
            "inertContainers": [],
            "phageContainers": [],
            "matrixContainers": [],
            "soluteContainers": [],
            "biomassContainers": [],
        }
        self.allContainers = []
        self.allBehaviors = []
        self.allEvents = []
        self.space = space
        self.dt = 0
        self.t = 0
        self.iteration = 0
        self.initialized = False
        self.out_dir = None
        self.config = cfg
        self._output_header = ["iteration", "time", "dt"]
        self._summary_writer = None
        self._summary_file = None
        self._save_frequency = 0
        self._event_at = []
        self._max_sim_time = (
            cfg.general.max_sim_time if "max_sim_time" in cfg.general else None
        )

        if "seed" in cfg.general:
            np.random.seed(int(cfg.general.seed))
            seed(int(cfg.general.seed))
            numba.jit(lambda x: np.random.seed(x), nopython=True)(int(cfg.general.seed))

        self.forceGC = cfg.general.forcegc if "forcegc" in cfg.general else False
        self.fixed_dt = cfg.general.fixed_dt if "fixed_dt" in cfg.general else None
        self.verbose = cfg.general.verbose if 'verbose' in cfg.general else False

    def _stop_if_initialized(self):
        if self.initialized:
            msg = "Attempting to add simulation components when initialized."
            raise RuntimeError(msg)

    def add_container(self, *containers):
        """containerGroups should be a dictionary: {group: [c1, c2, c3]}"""
        self._stop_if_initialized()

        self.allContainers.extend(containers)
        for container in containers:
            for g in container.groups:
                try:
                    self.containerGroups[g].append(container)
                except KeyError:
                    self.containerGroups[g] = [container]

    def add_behavior(self, *behaviors, position=None):
        self._stop_if_initialized()
        if any([not callable(b) for b in behaviors]):
            msg = f"All behaviors must be callable objects: {1}"  # TODO better msg
            raise RuntimeError(msg)
        if position is None:
            self.allBehaviors.extend(behaviors)
        else:
            self.allBehaviors = (
                self.allBehaviors[:position] + list(behaviors) + self.allBehaviors[position:]
            )

    def add_event(self, event, condition, args, once=True):
        """Add an event when condition is met."""
        self._stop_if_initialized()
        if not callable(condition):
            self._event_at.append(float(condition))
            condition = _make_time_condition(condition)

        self.allEvents.append((event, condition, args, once))

    def add_end_condition(self, condition, message, *args):
        """Adds an end condition with given message."""
        self._stop_if_initialized()
        self.all_end_conditions.append((condition, message, args))

    def _check_end_conditions(self):
        endmsg = ""
        for cond, message, args in self.all_end_conditions:
            endmsg += message if cond(self.space, self.t, *args) else ""
        return endmsg

    def initialize(self, output_directory, log_header=[], save_frequency=0):
        """Initialize the containers, marking the sim ready to iterate."""
        self._event_at = sorted(self._event_at, reverse=True)
        for b in self.allBehaviors:
            b._register(self)
            b.initialize(self)
        self._setup_output(output_directory, log_header, save_frequency)
        self.initialized = True

        names = [c.name for c in self.allContainers]
        uniqNames = set(names)
        if len(uniqNames) < len(names):
            raise ValueError(f"Not all container names are unique: {names}")

        print("Initialized.")

    def finish(self):
        """Clean up sim I/O operations. Called by perform_iterations."""
        self._summary_file.close()

    def iterate(self, t, dt_min=0.005, dt_max=np.inf, cleanup=False):
        """Initialize the containers, marking the sim ready to iterate.

        Parameters
        ----------
        t : float
            How long to iterate

        """
        start_time = time()
        t = self.t + t
        t_msg = "End time reached:{}".format(t)
        # self.add_end_condition(lambda _, y: y >= t, t_msg)
        self.all_end_conditions.append((lambda _, y: y >= t, t_msg, []))

        while True:
            _max = dt_max if dt_max < (t - self.t) else t - self.t

            iteration_start = time()
            self._single_iteration(dt_min, _max)
            iteration_end = time()
            # avoid 'zero' dt times, for events. 1e-5 is about 1 second
            if self.dt > 1e-5:
                self.summarize_state(iteration_end - iteration_start)

            if self._save_frequency > 0 and self.iteration % self._save_frequency == 0:
                self._save()

            if (
                self._max_sim_time
                and iteration_end - iteration_start >= self._max_sim_time
            ):
                print(f"FIN-T: iteration time >= {iteration_end - iteration_start}")
                break

            message = self._check_end_conditions()
            if message:
                print(message)
                self.all_end_conditions = [
                    x for x in self.all_end_conditions if x[1] != t_msg
                ]
                break

        end_time = time()
        print("Total time: {}".format(end_time - start_time))
        if cleanup:
            self.finish()

    def _single_iteration(self, dt_min=0.005, dt_max=np.inf):
        if not self.initialized:
            msg = "Attempting to iterate before initializing simulation."
            raise RuntimeError(msg)

        if self.fixed_dt > 0:
            dt = self.fixed_dt
        else:
            dt = min([b.calculate_dt() for b in self.allBehaviors])
            dt = min(dt_max, max(dt_min, dt))

        if self._event_at:  # supersedes fixed_dt
            if dt > (self._event_at[-1] - self.t):
                dt = self._event_at[-1] - self.t
                self._event_at.pop()
        self.dt = dt
        self.t += dt
        if self.dt < 1e-5:
            return

        self.iteration += 1

        for b in self.allBehaviors:
            self._current_behavior = b
            b.pre_iteration(dt)

        for b in self.allBehaviors:
            self._current_behavior = b
            try:
                if self.space.well_mixed:
                    b.well_mixed(dt)
                else:
                    b(dt)
            except:
                print(f"Behavior: {b}")
                raise

        for b in self.allBehaviors:
            self._current_behavior = b
            b.post_iteration(dt)

        removal = []
        for i, (event, condition, args, once) in enumerate(self.allEvents):
            if condition(self.space, self.t, *args):
                event(self.space, self.t, *args)
                if once:
                    removal.append(i)
        for i in sorted(removal, reverse=True):
            del self.allEvents[i]
        if self.forceGC:
            gc.collect()

    def _setup_output(self, directory, log_header, frequency):
        """Prepare the output directory, and if supplied output frequency."""
        self.out_dir = directory
        self._save_frequency = frequency
        np.set_printoptions(precision=3, linewidth=120)

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            for c in self.allContainers:
                try:
                    _, summary = c.summarize()
                except TypeError:
                    continue
                self._output_header.extend(summary.keys())

            for behavior in [b for b in self.allBehaviors if b.log or self.verbose]:
                try:
                    _, summary = behavior.summarize()
                except NotImplementedError:
                    continue
                self._output_header.extend(summary.keys())

            if log_header:  # TODO: behavior objects?
                self._output_header.extend(log_header)

            fname = "{0}/summary.csv".format(self.out_dir)
            self._summary_file = open(fname, "w")
            self._summary_writer = csv.writer(
                self._summary_file, quoting=csv.QUOTE_NONE, lineterminator="\n"
            )
            self._summary_writer.writerow(self._output_header)

    def summarize_state(self, iteration_timing=None):
        """Summarize the state of the simulation."""
        stdout_summary = []
        stdout_summary.append("it: {}".format(self.iteration))
        if iteration_timing:
            stdout_summary.append("sim: {0:0.3f}s".format(iteration_timing))
        stdout_summary.append("t: {0:0.3f}".format(self.t))
        stdout_summary.append("dt: {0:0.3f}".format(self.dt))
        iteration_summary = {"iteration": self.iteration, "time": self.t, "dt": self.dt}

        for container in self.allContainers:
            try:
                sout, summary = container.summarize()
            except TypeError:
                continue
            stdout_summary.append(sout)  # _prefix_name_to_dict(container.name, summary)
            iteration_summary.update(summary)

        for behavior in [b for b in self.allBehaviors if b.log or self.verbose]:
            try:
                sout, summary = behavior.summarize()
            except NotImplementedError:
                continue
            stdout_summary.append(sout)
            iteration_summary.update(summary)

        # get values in order for csv summary
        summary_row = [iteration_summary[name] for name in self._output_header]
        print(", ".join(stdout_summary), flush=True)
        self._summary_writer.writerow(summary_row)
        self._summary_file.flush()

    def _save(self, compressed=True):
        """Save data to npz file. Lookup numpy.savez for more information."""

        fname = "{0}/iteration{1:04}".format(self.out_dir, self.iteration)
        config = []
        for section in self.config:
            for item in self.config[section]:
                config.append(
                    (":".join((section, item)), str(self.config[section][item]))
                )

        savedata = {"config": np.array(config)}
        for container in self.allContainers:
            savedata.update(container.get_save_data())  # _prefix_name_to_dict

        if savedata:
            savedata["t"] = self.t
            savedata["iteration"] = self.iteration
            save = np.savez
            if compressed:
                save = np.savez_compressed
            save(fname, **savedata)
        self._summary_file.flush()

    def load(self, fname):
        """Load files matching 'prefix*'."""
        with np.load(fname, allow_pickle=True) as f:
            self.iteration = int(f["iteration"])
            self.t = float(f["t"])
            for c in self.allContainers:
                # these_data = {key: f[key] for key in f.keys() if key.startswith(c.name)}
                # these_data = _strip_name_from_dict(c.name, these_data)
                c.load({key: f[key] for key in f.keys() if key.startswith(c.name)})


class Container:
    """Container base class."""

    def get_save_data(self, fname):
        """Get data we need to store as a dict."""
        msg = "'{}' container subclass has not implemented 'get_save_data'."
        raise NotImplementedError(msg.format(type(self).__name__))

    def load(self, data):
        """Populate container with values from data dict."""
        msg = "'{}' container subclass has not implemented 'load'."
        raise NotImplementedError(msg.format(type(self).__name__))

    def summarize(self):
        """Return dict of summarized container."""
        msg = "'{}' container subclass has not implemented 'summarize'."
        raise NotImplementedError(msg.format(type(self).__name__))


class Space:
    """Contains the properties of the Simulation space.

    Parameters
    ----------

    params: ndict containing 'dl' 'shape' and 'well_mixed' boolean
        Otherwise, params as a tuple for shape and provide dl and well mixed explicitly

    """

    def __init__(self, params, dl=None, well_mixed=False):
        """See Space documentation for __init__ documentation."""
        if dl is not None:
            shape = params
        else:
            shape = params.shape
            dl = params.dl
            well_mixed = params.well_mixed

        self.well_mixed = well_mixed
        self.shape = tuple(int(x) for x in shape)  # TODO: change 1d, 2d to 3d.
        self.dl = dl
        self.dimensions = len(shape)

        if self.dimensions > 3:
            raise ValueError("dimensions must be 1, 2, or 3")

        self.sides = tuple(np.arange(side) * dl for side in shape)
        self._laplacian_cache = {}

        self.periodic = (0,) + (1,) * (self.dimensions - 1)
        self.N = np.prod(self.shape)
        if not well_mixed:
            self.dV = self.dl ** 3
            self.neighbors = self._construct_neighbors_and_laplacian()
            self.meshgrid = np.meshgrid(*self.sides, indexing="ij")
            if self.dimensions == 2:
                self.mesh = Grid2D(dx=dl, dy=dl, nx=shape[1], ny=shape[0])
            else:
                self.mesh = Grid3D(
                    dx=dl, dy=dl, dz=dl, nx=shape[1], ny=shape[0], nz=shape[2]
                )
        else:
            self.dV = np.prod(dl * np.array(self.shape))
            if len(self.shape) == 2:
                self.dV *= dl
            self.dl = np.cbrt(self.dV)

            self.shape = (1,)
            self.L = np.atleast_2d(self.shape)
            self.neighbors = np.array([0])
            self.meshgrid = [np.array([dl])]

    def construct_mesh(self, height):
        if self.dimensions == 2:
            mesh = Grid2D(dx=self.dl, dy=self.dl, nx=self.shape[1], ny=height)
        else:
            mesh = Grid3D(
                dx=self.dl,
                dy=self.dl,
                dz=self.dl,
                nx=self.shape[1],
                ny=height,
                nz=self.shape[2],
            )
        return mesh

    def _construct_neighbors_and_laplacian(self):
        Lneighbors = self._construct_laplacian(self.shape).todia()
        Lneighbors.setdiag(0, 0)
        neighbors = {}
        for k, v in Lneighbors.todok().keys():
            neighbors.setdefault(k, []).append(v)
        return neighbors

    def _construct_laplacian(self, shape):
        L = -pyamg.gallery.laplacian.poisson(shape, dtype=np.int8)
        L.data[self.dimensions][: np.prod(shape[1:])] += 1  # no flux
        L.data[self.dimensions][-np.prod(shape[1:]) :] += 1

        # for dealing with periodic boundaries
        for dim, size in enumerate(shape[1:], 1):
            ar = np.zeros(shape)
            ar.swapaxes(0, dim)[0] = 1
            ar = ar.reshape(-1)
            L.setdiag(ar, -(size - 1))
            L.setdiag(ar, size - 1)

        return L.tocsr()

    def construct_neighbors(self):
        Lneighbors = self._construct_laplacian(self.shape).todia()
        Lneighbors.setdiag(0, 0)
        return Lneighbors.tocsr()

    def laplacian(self, height=None):
        if height is None:
            height = self.shape[0]
        if height not in self._laplacian_cache:
            shape = (height,) + self.shape[1:]
            L = self._construct_laplacian(shape).tolil()
            L[-np.prod(shape[1:]) :] = 0
            self._laplacian_cache[height] = (L.tocsr(), shape)
        return self._laplacian_cache[height]

    def breadth_first_search(self, grid, condition=lambda x: x > 0, init=None):
        """Search the grid of space that meets a specific condition. """
        grid_flat = grid.reshape(-1)
        if init is None:
            init = range(np.prod(self.shape[1:]))
        elif str(init).lower() == "top":
            N = np.prod(self.shape)
            init = range(N - np.prod(self.shape[1:]), N)

        nodes = deque(init)
        reachable = np.ones(np.prod(self.shape), dtype=np.int8)
        while nodes:
            ix = nodes.popleft()
            if reachable[ix] != 1:
                continue  # if we've visited it already, skip it
            if not condition(grid_flat[ix]):
                reachable[ix] = 0  # mark unreachable, and a boundary
            else:
                reachable[ix] = -1  # is reachable!
                for ix_n in self.neighbors[ix]:  # add unvisited neighbors
                    if reachable[ix_n] == 1:  # is unvisited
                        nodes.append(ix_n)
        return reachable.reshape(self.shape)
