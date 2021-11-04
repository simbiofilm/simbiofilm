"""
events.py

Events to occur during the simulation.
"""
import numpy as np

from .core import to_grid, to_list


def inoculate_at(space, time, bacteria_container, n):
    """Adds n bacteria to the specified container, varying the mass of each slightly, at a specified time.
    Generally called on an empty container

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    time : float
        Time at which to innocluate

    bacteria_container : Bacteria (see containers.py for documentation)
        Container to innoclate. Uses default parameters for the container

    n : int
        Number of bacteria to add

    -------
    Returns: void, modifies bacteria_container
    """
    params = bacteria_container.P.copy()
    mass = params.mass
    for ix in np.random.randint(np.prod(space.shape[1:]), size=n):
        params["mass"] = np.random.normal(mass, mass * 0.2)
        bacteria_container.add_individual(ix, params)


def infect_at(space, time, phage_container, biomass_containers, N):
    """Initializes a phage infection at a specified time. Adds the phages across the top

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    time : float
        Time at which to begin the infection

    phage_container : Phage (see containers.py for documentation)
        Phage Container for phage of interest

    biomass_containers : ParticleContainer (see containers.py for documentation)
        Containers belonging to the biomassContainers group

    N : int
        How many individuals to infect from biomass_containers

    -------
    Returns: void, modifies the phage_container object

    """
    biomass = to_grid(biomass_containers, "mass")

    reachable = space.breadth_first_search(biomass, lambda x: x == 0, "top")
    indices = list(np.where(reachable == 0))
    indices[0] += 3
    indices = np.ravel_multi_index(indices, space.shape)
    for ix in np.random.choice(indices, N):
        phage_container.add_individual(ix)


def infect_point(space, time, count, phage_container, biomass_containers):
    """Initializes a phage infection at a specified location

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    time : float
        Time at which infection occurs

    count : int
        Number of phages to add

    phage_container : Phage (see containers.py for documentation)
        Phage container for phage of interest. phage_container.infected is set to true

    biomass_containers : ParticleContainer (see containers.py for documentation)
        Containers belonging to biomassContainers group, to be infected

    -------
    Returns: void, adds individuals to phage_container

    """
    if count == 0:
        return
    biomass = to_grid(biomass_containers, "mass")

    phage_container.infected = True

    if space.well_mixed:
        phage_container.add_multiple_individuals(count, 0)
        return

    reachable = space.breadth_first_search(biomass, lambda x: x == 0, "top")
    indices = list(np.where(reachable == 0))
    maxa = np.argmax(indices[0])

    ix = [ind[maxa] for ind in indices]
    phage_container.add_multiple_individuals(
        int(count), np.ravel_multi_index(ix, space.shape)
    )


def initialize_bulk_substrate(space, time, solute, minval=0.75):
    """Initializes bulk substrate, approximates downward

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    time : float
        Current time of simulation

    solute : Solute (see containers.py for documentation)
        Solute container for solute to be added

    minval : float
        Minimum value of solute at each grid point. Defaults to 0.75

    -------
    Returns: void, modifies the value attribute of solute

    """
    mgy = space.meshgrid[0]
    solute.value = np.clip(minval * (1 + mgy / np.max(mgy)), 0, 1)


def invasion(space, time, biomass_containers, invader, N, params_generator=None):
    """Initializes an invasion event by adding N inviduals to invader container

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    time : float
        Current time of simulation

    biomass_containers : ParticleContainer (see containers.py for documentation)
        Containers belonging to the biomassContainers group

    invader : Phage (see containers.py for documentation)
        Phage container of invading phages (empty)

    N : int
        Number of invaders

    params_generator
        Generator to add parameters to each invader, defaults to None

    -------
    Returns: void, modifies invader object

    """
    biomass = to_grid(biomass_containers, "mass")
    if not params_generator:
        params_generator = [None for _ in range(N)]
    reachable = space.breadth_first_search(biomass, lambda x: x == 0, "top")
    indices = np.ravel_multi_index(np.where(reachable == 0), space.shape)
    for ix, params in zip(np.random.choice(indices, N), params_generator):
        invader.add_individual(ix, params)
