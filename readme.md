# Quickstart

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. `mkdir PROJECTNAME; cd PROJECTNAME`
3. `conda create -c conda-forge -n PROJECTNAME numpy scipy numba fipy scipy pyamg libgcc pandas tqdm plotnine`
4. `conda activate PROJECTNAME`
5. `git clone git@github.com:nadellinsilico/simbiofilm.git`
6. `pip install -e simbiofilm`
7. `cp simbiofilm/scripts/plotlib.py .`
8. `python simbiofilm/examples/simple.py`

Creating the conda environment might take a while.

Copy the simple runscript and edit!

# The runscript

The simbiofilm runscript generally has two components: the default
configuration setup, and the simulation setup. The simulation setup, outside
of a few options, generally doesn't change within a single study, and should
remain relatively constant. The configuration is varies the parameters for the
study and any particular models you are testing.

The configuration is a dictionary with a definition to allow '.' access.
For example, `cfg.general.seed` is allowed, rather than only allowing
`cfg['general']['seed']`.

This dictionary gets potentially modified by the `-p` command line option, 
which has the format
`python runscript.py -p sectionname1:paramname1,sectionname2:paramname2 value1,value2`
For example,
`python runscript.py -p general:seed,species1:density 4321,210e3`

Note that the `sb.cfg_from_dict` will try its best to convert values to their
appropriate types, but sometimes will need help. In particular, don't try to use
lists/other iterables, and ints might need explicit conversion.


See the examples for the structure of the run scripts, but they have 3 major components:

1. The configuration, which is basically a dict wrapped by the `cfg_from_dict` function
2. The simulation setup, which creates the necessary simbiofilm simulation objects:
  a. space
  b. simulation
  c. containers (solutes and biomass/particulate containers)
  d. behaviors (interactions between containers)
3. A command line option parser, so that the parameter parsing above works correctly.

See examples/simple.py for a reduced example.

# Videos
