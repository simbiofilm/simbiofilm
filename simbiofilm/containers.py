"""
containers.py

Contains Classes holding central information for various container types.
"""
from itertools import count
import numpy as np
from .core import Container


class ParticleContainer(Container):
    """Contains full particle information. If location is int, old 'hybrid'.

    Parameters
    ----------

    name : str
        Unique name of the container, identical names will throw an error
        when added to the Simulation

    space : Space (see core.py for documentation)
        Space object for the simulation

    params : ParameterSet
        ParameterSet populated with parameters fulfilling the associated
        behaviors' requirements

    groups : list of str
        Groups to which the ParticleContainer belongs i.e. 'biomassContainers'.

    maxn : int, optional
        Initial maximum size of the data array. The array will double in
        size when needed

    """

    _unique_id = count(1)

    def __init__(self, name, groups, space, params, maxn=1024):
        """See ParticleContainer for init documentation."""
        datatypes = [("id", "u4")]
        datatypes.append(("parent", "u4"))
        datatypes.append(("mass", float))
        datatypes.append(("location", "u4"))
        for k, v in params.items():
            if k not in list(zip(*datatypes))[0]:
                datatypes.append((k, type(v)))

        super().__setattr__("_params", dict(datatypes))
        self._data = np.rec.array(np.zeros(int(maxn), dtype=datatypes))
        self._view_cache = {}
        self._id_to_index = {}
        self._count = 0
        self._current_id = 1

        self.name = name
        self.groups = groups
        self.space = space
        self.P = params

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)

        if isinstance(key, int):
            return self.with_id(key)

        return self.iter(key)

    def iter(self, arr):
        for ind in reversed(self._data[: self.n][arr]):
            yield ind

    def __getattr__(self, name):
        return getattr(self._data, name)[: self.n]
        if name in self._params:
            if name not in self._view_cache:
                self._view_cache[name] = getattr(self._data, name)[: self.n]
            return self._view_cache[name]
        msg = "'{}' has no attribute or parameter '{}'"
        msg = msg.format(type(self).__name__, name)
        raise AttributeError(msg)

    def __setattr__(self, name, value):
        if name in self._params:
            self._data[name][: self.n] = value
        else:
            super().__setattr__(name, value)

    def __iter__(self):
        """Provide a view of the record."""
        # TODO provide iterator that can take slices and can do both ind.remove and
        # container.remove(ind). setup, loop yield, teardown.
        for ind in reversed(self._data[: self.n]):
            yield ind

    def _clear_cache(self):
        self._view_cache = {}

    def _add_particle(self, params):
        if self._count == self._data.shape[0]:
            self._data = np.concatenate((self._data, self._data))
            self._data = np.rec.array(self._data)

        id = next(ParticleContainer._unique_id)
        self._current_id = id  # for saving
        ix = self._count
        self._id_to_index[id] = ix
        self._count += 1
        # TODO: rewrite to use a simple assignment
        for k, v in params.items():
            setattr(self._data[ix], k, v)
        self._data[ix].id = id
        self._clear_cache()
        return self._data[ix]

    def with_id(self, id):
        """Return the record of the particle with given id.

        Parameters
        ----------

        id : int
            Unique id of the particle of interest

        -------
        Returns : np.array : record of the particle

        """
        if id in self._id_to_index:
            return self._data[self._id_to_index[id]]
        msg = "'{}' has no particle with id '{}'"
        msg = msg.format(type(self).__name__, id)
        raise KeyError(msg)

    def at_location(self, location):
        """Return the records of the particle at specified location.

        Parameters
        ----

        location : int
            Location to find particles

        Returns : generator of np.arrays of particles at the location
        """
        if self.n == 0:
            return
        ixs = np.where(self.location == location)[0]
        for ix in ixs:
            yield self._data[ix]

    def add_individual(self, location, params=None):
        """Add an individual with default parameters at location.

        Parameters
        ----------

        location : int
            Location at which to add the indivdual

        params : ParameterSet
            Parameter set to assign indivdual, if None, uses defaults

        -------
        Returns: np.array : record of the added indivdual
        """
        if hasattr(location, "__iter__"):
            location = np.ravel_multi_index(location, self.space.shape)

        all_params = {}
        all_params.update(self.P.items())
        if params:
            all_params.update(params.items())
        all_params["location"] = location
        return self._add_particle(all_params)

    def add_multiple_individuals(self, n, location, params=None):
        """Add multiple individuals at specified location. See add_individual.

        Parameters
        ----------

        n : int
            Number of particles to add

        location : int
            Location at which to add the indviduals

        params : ParameterSet
            Parameter set to assign indivduals, if None, uses defaults

        -------
        Returns : list of np.arrays : record of the added individuals

        """
        # TODO: check if location is iterable sensibly
        return [self.add_individual(location, params) for _ in range(n)]

    def clone(self, parent, **params):
        """Clones parent exactly."""
        parameters = dict(zip(parent.dtype.names, parent))
        if params:
            parameters.update(params)
        return self.add_individual(parent.location, parameters)

    def remove(self, id):
        """Removes an individual.

        Parameters
        ----------

        id : int
            Unique id of particle to remove
        """
        try:
            individual = id
            id = individual.id
            if not individual == self.with_id(id):
                # TODO: better message
                msg = "Individual does not appear to be in this container."
                raise RuntimeError(msg)
        except AttributeError:
            individual = self.with_id(id)
        self._count -= 1
        old_ix = self._id_to_index[id]
        if old_ix < self._count:
            self._data[old_ix] = self._data[self._count]
        self._id_to_index = {id: ix for ix, id in enumerate(self.id)}
        self._clear_cache()

    def multi_remove(self, removal):
        """Removes mutliple individuals. Removes and replaces in bulk

        Parameters
        ----------

        remove : boolean array
            True if individual should be removed, false otherwise. Length must equal self.n

        """
        if len(removal) != self.n:
            msg = "Length of array for multi_remove must equal n of container."
            raise IndexError(msg)

        nalive = self._count - np.sum(removal)
        if np.sum(removal) == 0 or self._count == 0:
            return

        ixs = np.arange(self.n)
        alive_ix = ixs[nalive:][~removal[nalive:]]
        dead_ix = ixs[:nalive][removal[:nalive]]
        if alive_ix.size > 0:
            self._data[dead_ix] = self._data[alive_ix]

        self._count = nalive
        self._clear_cache()
        self._id_to_index = {id: ix for ix, id in enumerate(self.id)}

    def get_save_data(self):
        """Get savedata in a dict."""
        savedata = {}
        savedata[f"{self.name}_data"] = self._data[: self.n]
        savedata[f"{self.name}_n"] = self.n
        savedata[f"{self.name}_params"] = tuple(self._params.items())
        savedata[f"{self.name}_nextid"] = self._current_id
        return savedata

    def load(self, data):
        """Load data from dict."""
        #     name = name + "_"
        #     n = len(name)
        #     for key in list(dictionary.keys()):
        #         if key.startswith(name):
        #             newkey = key[n:]
        #             dictionary[newkey] = dictionary.pop(key)
        #     return dictionary
        # these_data = _strip_name_from_dict(c.name, these_data)
        # data
        datatypes = [(n, t) for n, t in data[f"{self.name}_params"]]
        self._params = dict(datatypes)
        self._count = int(data[f"{self.name}_n"])
        ParticleContainer._unique_id = count(int(data[f"{self.name}_nextid"]))

        if data[f"{self.name}_data"].shape[0] > 0:
            size = int(2 ** np.ceil(np.log2(data[f"{self.name}_data"].shape[0])))
            size = max((size, 1024))
            self._data = np.rec.array(np.zeros(size, dtype=datatypes))
            self._data[: self._count] = data[f"{self.name}_data"]
        else:
            self._data = np.rec.array(np.zeros(1024, dtype=datatypes))

        self._id_to_index = {id: ix for ix, id in enumerate(self.id)}
        self._clear_cache()

    def summarize(self):
        """Summarize container."""
        self.remainder = 1
        msg = "{0} N: {1}"
        msg = msg.format(self.name, self.n)
        return msg, {f"{self.name}_N": self.n}

    @property
    def n(self):
        """Return the number of particles."""
        return self._count

    @property
    def volume(self):
        """Relative grid volume, NOT absolute volume."""
        return (
            np.zeros(self.n)
            if "density" not in self.P
            else self.mass / (self.density * self.space.dV)
        )

    def phage_interacted(self, particle, phage):
        # Return [bool, bool] for whether or not [particle infected, phg removed]
        # [True, False] should not happen ever.
        msg = f"phage_interacted not implemented for {self}"
        raise NotImplementedError(msg)


class Bacteria(ParticleContainer):
    """Container for Bacteria particles. See ParticleContainer documentation."""

    def __init__(
        self,
        name,
        space,
        params,
        groups=["activeContainers", "biomassContainers"],
        extragroups=[],
        maxn=1024,
    ):
        """See Bacteria docstring for init documentation."""
        super().__init__(name, groups + extragroups, space, params, maxn)

    def phage_interacted(self, bacterium, phage):
        # see simbiofilm.behaviors.phage_interaction
        # Return true if we should remove the particle
        if not bacterium == self.with_id(bacterium.id):
            msg = "Individual does not appear to be in this container."
            raise RuntimeError(msg)
        return [not bacterium.resistant, bacterium.adsorbable]


class Matrix(ParticleContainer):
    """Container for Matrix particles. See ParticleContainer documentation."""

    def __init__(
        self,
        name,
        space,
        params,
        groups=["inertContainers", "matrixContainers", "biomassContainers"],
        sticky=False,
        maxn=1024,
    ):
        """See Bacteria docstring for init documentation."""
        super().__init__(name, groups, space, params, maxn)
        self.sticky = sticky

    def phage_interacted(self, bacterium, phage):
        return [False, self.sticky]


class InfectedBacteria(Bacteria):
    """Container for Infected bacteria for phage dynamics Contains information
    about previously clean bacteria and the infecting phage

    Parameters
    ----------

    name : str
        Unique name of the container

    space : Space (see core.py for documentation)
        Space object for the simulation

    bacteria_params : ParameterSet
        ParameterSet populated with parameters corresponding to the clean bacteria

    phage_params : ParameterSet
        ParameterSet populated with parameters corresponding to the infecting phage

    groups : list of str
        Groups to which the InfectedBacteria Container belongs
        defaults to 'inertContainers' and 'biomassContainers'

    maxn : int, optional
        Initial maximum size of the data array. The array will double in
        size when needed


    """

    def __init__(
        self,
        name,
        space,
        params,
        phage_params,
        groups=["inertContainers", "biomassContainers"],
        maxn=1e4,
    ):
        """See InfectedBacteria for init documentation."""
        infected_params = params.copy()
        for key in phage_params:
            newkey = "phage_{}".format(key)
            if newkey in infected_params:
                msg = "{} in infected and phage parameters, rename one."
                raise KeyError(msg.format(newkey))
            infected_params[newkey] = phage_params[key]
        infected_params["infectedby"] = -1
        infected_params["incubation_time"] = infected_params["phage_incubation_period"]
        super().__init__(name, space, infected_params, groups, maxn=maxn)
        self.phage_params = tuple(phage_params.keys())
        self.bacteria_params = tuple(k for k in params.keys() if k != "multi_infect")

    def infect(self, bacterium, phage):
        """Adds infected bacterium with parameters from previously clean
        bacteria and infecting phage
        """
        pname = "phage_{}"
        params = self.P.copy()
        for p in self.bacteria_params:
            params[p] = getattr(bacterium, p)
        for phg_p, p in ((pname.format(x), x) for x in self.phage_params):
            params[phg_p] = getattr(phage, p)
        params["infectedby"] = phage.id
        params["incubation_time"] = (
            phage.incubation_period * phage.remainder * np.random.normal(1, 0.1)
        )

        return self.add_individual(bacterium.location, params)

    def summarize(self):
        """Summarize infected bacteria."""
        msg = "{0} N: {1}"
        msg = msg.format(self.name, self.n)
        return msg, {f"{self.name}_N": self.n}

    def phage_interacted(self, bacterium, phage):
        # see simbiofilm.behaviors.phage_interaction
        # Return true if we should remove the particle
        return [False, bacterium.multi_infect]


class Phage(ParticleContainer):
    """Contains information for phage. See ParticleContainer documentation"""

    def __init__(
        self,
        name,
        space,
        params,
        groups=["phageContainers", "biomassContainers"],
        maxn=1e5,
    ):
        """See Phage for init documentation."""
        params = params.copy()
        params["remainder"] = float(1)
        super().__init__(name, groups, space, params, maxn)
        self.infected = False

    def add_individual(self, index, params=None):
        """Add an individual with default parameters at location."""
        if params is None:
            params = self.P
        super().add_individual(index, params)

    def get_save_data(self):
        savedata = super().get_save_data()
        savedata[f"{self.name}_infected"] = self.infected
        return savedata

    def load(self, data):
        self.infected = data["infected"]
        del data["infected"]
        super().load(data)

    def summarize(self):
        """Summarize phage."""
        self.remainder = 1
        msg = "{0} N: {1}"
        msg = msg.format(self.name, self.n)
        return msg, {f"{self.name}_N": self.n}


class Solute(Container):
    """Contains concentration and parameter information for solute.

    Parameters
    ----------

    name : str
        Unique name of the container

    space : Space (see core.py for documentation)
        Space object for the simulation

    params : ParameterSet
        ParameterSet populated with parameters fulfilling the associated behaviors' requirements

    groups : list of str
        Groups to which the ParticleContainer belongs. Default as 'soluteContainers'

    """

    def __init__(self, name, space, params, groups=["soluteContainers"], dtype=float):
        self.name = name
        self.space = space
        self.value = np.zeros(space.shape, dtype)
        self.P = params
        self.groups = groups

    def __getattr__(self, name):
        if name in self.P:
            return getattr(self.P, name)
        else:
            msg = "'{}' object has no attribute '{}'".format(type(self), name)
            raise AttributeError(msg)

    def get_save_data(self):
        return {f"{self.name}_value": self.value}

    def load(self, data):
        self.value = data[f"{self.name}_value"]

    def summarize(self):
        return None
