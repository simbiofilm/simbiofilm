"""
end_conditions.py

Conditions for terminating the simulation
"""
import numpy as np

from .core import to_list, count


def empty_container(space, t, tcheck, containers):
    """After a specified time, checks if the containers of interest are empty

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    t : float
        Current simulation time

    tcheck : float
        Time after which to begin checking condition

    containers : Container (see core.py for documentation)
        Containers of interest

    -------
    Returns: True if after the specified time containers is empty, False otherwise

    """
    # TODO: end condition class
    return np.sum(count(space, containers)) == 0 and t > tcheck


def empty_after_infection(space, t, phage, biomass):
    """Checks if a container is empty after phage infection has occured

    Parameters
    ----------

    space : Space (see core.py for documentation)
        Space object for the simulation

    t : float
        Current simulation time

    phage : Phage (see containers.py for documentation)
        Infecting phage of interest

    biomass : ParticleContainer (see containers.py for documentation)
        Containers of interest

    -------
    Returns: True if biomass is empty and phage infection has occured, False otherwise

    """

    return empty_container(space, t, 0, biomass) if phage.infected else False
