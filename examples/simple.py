import argparse
import os
import simbiofilm as sb

# There may be some warnings from libraries we use. Feel free to enable this
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def config():
    return sb.cfg_from_dict(
        {
            "general": {
                "output_frequency": 1,
                "nconnections": 1,
                "seed": 1235,
                "runtime": 20,
                "init_count": 150,
                "init_f": 0.5,
                "line_shove": False,
                "max_sim_time": 100,
                "3D": False,
                "connectivity": 0.05,
                "fixed_dt": 1 / 24 + 0.0000001,
                "targetone": True,
            },
            "space": {"dl": 3e-6, "shape": (75, 200), "well_mixed": False},
            "substrate": {"max": 6, "diffusivity": 2e-5, "K": 1.18, "h": 15e-6},
            "erosion": {"rate": 7.5e8},
            "species1": {
                "density": 200e3,
                "mass": 1e-12,
                "division_mass": 1.333e-12,
                "mu": 24.5,
                "adhesion": 1,
                "yield_s": 0.495,
            },
            "species2": {
                "density": 220e3,
                "mass": 1e-12,
                "division_mass": 1.333e-12,
                "mu": 24.5,
                "adhesion": 1,
                "yield_s": 0.495,
            },
        }
    )


def setup(cfg, outdir="tmp"):
    """Do the thing."""

    space = sb.Space(cfg.space)
    sim = sb.Simulation(space, cfg)

    substrate = sb.Solute("solute", space, cfg.substrate)

    activeSpecies = [
        sb.Bacteria("species1", space, cfg.species1, extragroups=['Susceptible']),
        sb.Bacteria("species2", space, cfg.species2, extragroups=['Producer']),
    ]

    sim.add_container(substrate, *activeSpecies)

    sb.inoculate_at(space, 0, activeSpecies[0], int(cfg.general.init_count * cfg.general.init_f))
    sb.inoculate_at(space, 0, activeSpecies[1], int(cfg.general.init_count * (1 - cfg.general.init_f)))

    sb.initialize_bulk_substrate(space, 0, substrate)

    reactions = [sb.MonodRateReaction(substrate, sp) for sp in activeSpecies]

    sim.add_behavior(
        sb.DiffusionReaction(substrate, reactions),
        sb.Biomass_growth(reactions),
        sb.Bacterial_division(),
        sb.Erode_individuals(cfg.erosion.rate),
        sb.Detach_biomass(),
        sb.Relax_biofilm(),
    )

    sim.initialize(f"{outdir}/run1", [], cfg.general.output_frequency)
    return sim


def main():
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", metavar=("names", "values"), nargs=2)
    parser.add_argument("-o", metavar="output_dir")

    args = parser.parse_args()

    cfg = config()
    if args.p:
        cfg = sb.parse_params(args.p[0], args.p[1], cfg)
    name = args.o if args.o else f"local_runs/{sys.argv[0][:-3]}"
    print(name)
    sim = setup(cfg, name)
    try:
        sim.iterate(cfg.general.runtime, dt_max=0.02)
    except KeyboardInterrupt as kerr:
        sim.finish()
        raise kerr
    sim.finish()


if __name__ == "__main__":
    main()
