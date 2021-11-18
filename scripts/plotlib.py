import pandas
from time import time
import numpy as np
import multiprocessing as mp
import re
from re import sub, match
import warnings
import argparse
import os
import csv
from contextlib import contextmanager

import matplotlib
import matplotlib.colors
from matplotlib.colors import cnames, hex2color, rgb_to_hsv, hsv_to_rgb, to_rgb
import matplotlib.animation as animation
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.widgets import Slider, Button, RadioButtons

from itertools import product
import itertools as itt

# matplotlib.use("agg")
# matplotlib deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

import plotnine
from plotnine import scales, geoms, labels, guides, guide_legend
from plotnine import ggplot, aes, facet_wrap, stats
from plotnine import element_text, theme
from plotnine.themes import theme_bw, theme_classic

from tempfile import mkstemp

def getcfg(file):
    if type(file) is str:
        with np.load(file) as f:
            configdata = f["config"]
    else:
        configdata = file

    cfg = dict()
    for name, value in configdata:
        section, item = name.split(":")
        if section not in cfg:
            cfg[section] = dict()
        cfg[section][item] = _convert(value)
    return cfg


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


def _generate_colors(names):
    colors = []
    cgen = {
        'species' : (n for n in ['red', 'blue', 'orange']),
        'matrix': (n for n in ['orange', 'peru', 'saddlebrown']),
        'infected': (n for n in ['green', 'lime', 'aquamarine']),
        'phage': (n for n in ['black', 'orange']),
        'solute': (n for n in ['#3c1978', 'orange']),
    }

    try:
        for sp in names:
            if re.match('phage', sp, flags=re.IGNORECASE):
                colors.append(next(cgen['phage']))
            elif re.match('infected', sp, flags=re.IGNORECASE):
                colors.append(next(cgen['infected']))
            elif re.match('matrix', sp, flags=re.IGNORECASE):
                colors.append(next(cgen['matrix']))
            elif re.match('solute', sp, flags=re.IGNORECASE):
                colors.append(next(cgen['solute']))
            elif re.match('substrate', sp, flags=re.IGNORECASE):
                colors.append(next(cgen['solute']))
            else:
                colors.append(next(cgen['species']))
    except StopIteration:
        msg = f'More species than colors. Check against _plot_frame: \n{names}'
        msg += '\n\nPairs:\n'
        msg += join([f'{sp}: {c}' for sp, c in itt.zip_longest(names, colors, fillvalue='Unassigned')], '\n')
        raise RuntimeError(msg)
    return colors


def _make_base(shape):
    from vapory_git import Box, Texture, Pigment, Finish, Normal
    box = Box(
        [-1, -1, -1],
        [0, shape[1] + 1, shape[2] + 1],
        Texture(
            Pigment("color", [0.5, 0.5, 0.5]),
            Finish("ambient", 0.6),
            Normal("agate", 0.25, "scale", 0.5),
        ),
    )
    return box


def _biomass_to_grid(file, species_names=None, solute_names=None, dat=None):
    with np.load(file, allow_pickle=True) as f:
        cfg = getcfg(f["config"])
        shape = tuple(map(int, cfg['space']['shape']))
        if dat:
            dat = {k: f[k] for k in dat}

        if species_names is None:
            species_names = [x[:-5] for x in list(f.keys()) if x.endswith("_data")]
        if solute_names is None:
            solute = {x[:-6]: f[x] for x in list(f.keys()) if x.endswith("_value")}
        else:
            solute = {x[:-6]: f[x] for x in solute_names}

        biomass = {x: np.zeros(shape) for x in species_names}

        for name, b in biomass.items():
            name = name + '_data'
            location = [ind["location"] for ind in f[name]]
            mass = 1 if "phage" in name else [ind["mass"] for ind in f[name]]
            np.add.at(b.reshape(-1), location, mass)


    return biomass, solute, cfg, dat


def _render_3d_frame(
    file,
    outdir,
    shape,
    max_biomass=6.5e-12,
    max_phage=1000,
    format="png",
    prefix="",
    species_names=None,
    container_colors=None,
):
    from vapory_git import Camera, Scene, Background, LightSource, Texture, Pigment, Finish, Box
    outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'

    with np.load(file) as f:
        if species_names is None:
            species_names = [x[:-5] for x in list(f.keys()) if x.endswith("_data")]

        biomass = {x: np.zeros(shape) for x in species_names}

        for name, b in biomass.items():
            name = name + '_data'
            location = [ind["location"] for ind in f[name]]
            mass = 1 if "phage" in name else [ind["mass"] for ind in f[name]]
            np.add.at(b.reshape(-1), location, mass)

    objects = [_make_base(shape), Background("color", [0.9, 0.9, 0.9])]

    colors = dict(zip(species_names, _generate_colors(species_names)))

    for ix in product(*map(range, shape)):
        total_biomass = sum([b[ix] for sp, b in biomass.items() if 'phage' not in sp])
        if total_biomass == 0:
            continue

        # get color
        color = np.zeros(3)  # rgb
        minv = 0.250
        phage_factor = 1
        for sp, b in biomass.items():
            if 'phage' not in sp:
                color += np.array(to_rgb(colors[sp])) * (b[ix] / total_biomass)
            else:
                if b[ix] > 0:
                    phage_factor *= 1 - (0.95 - minv) * np.log10(b[ix]) / np.log10(max_phage) + minv
        color = color * phage_factor ** 2 * 0.8
        color = color.tolist() + [1 - (total_biomass / max_biomass) ** 3]
        # get box corners
        spacing = 0.02
        ix1 = np.array(ix, dtype=float)
        ix2 = ix1 + 1 - spacing
        ix1 += spacing

        # make box
        box = Box(ix1, ix2, Texture(Pigment("color", "rgbf", color)), Finish("ambient", 1.0, "diffuse", 1.0))
        # , 'no_shadow')
        objects.append(box)

    camera = Camera(
        "location",
        # [shape[1], -shape[1] * 1 / 2, -shape[2] * 1 / 2],
        # [shape[1], -shape[1] * 1 / 2, shape[2] * 3 / 2],
        # [shape[1], shape[1] * 3 / 2, -shape[2] * 1 / 2],
        [shape[1], shape[1] * 3 / 2, shape[2] * 3 / 2],
        "sky",
        [1, 0, 0],
        "look_at",
        [0, shape[1] / 2, shape[2] / 2],
    )

    os.makedirs('temp_pov', exist_ok=True)
    tmpfname = mkstemp(".pov", dir='temp_pov')[1]
    Scene(camera, objects=objects).render(outname, tempfile=tmpfname, width=3000, height=3000)
    print(f"Rendered {outname}")


def _plot_frame(
    file,
    outdir,
    shape,
    crop=None,
    max_biomass=6.5e-12,
    max_phage=1000,
    format="png",
    prefix="",
    solute_adjustment=None,
    species_names=None,
    container_colors=None,
):

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # matplotlib deprecation warnings

    outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    biomass, solutes, cfg, dat = _biomass_to_grid(file, species_names, None, ['t'])
    t = float(dat['t'])
    gridcols = np.unravel_index(range(np.prod(shape)), shape)
    _make_df = lambda d, n: pandas.DataFrame({"count": d.reshape(-1), "x": gridcols[1], "y": gridcols[0], 'species': n})

    # Write biomass into dataframes
    dfs = {n: _make_df(d, n) for n, d in biomass.items()}
    biomass = pandas.concat([df for name, df in dfs.items() if 'phage' not in name], sort=False)
    phage = pandas.concat([df for name, df in dfs.items() if 'phage' in name], sort=False)

    # Calculate biomass alpha
    minv = 0.20
    minb = biomass[biomass["count"] > 0]["count"].min()
    biomass["alpha"] = ((0.95 - minv) * (biomass["count"] - minb)) / (max_biomass - minb) + minv

    if len(phage) > 0:
        # Calculate phage alpha
        minv = 0.250
        maxp = np.log10(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore log10(0) warnings
            phage["alpha"] = ((0.95 - minv) * (np.log10(phage["count"]))) / (maxp) + minv

    # Set up solute, merge data
    if solute_adjustment is not None:
        maxs = 0
        for name, value in solutes.items():
            value = np.log10(value * 1.0 + solute_adjustment)
            mins = value[value.nonzero()].min()
            maxs = value[value.nonzero()].max()
            value = ((0.95) * (value - mins)) / (maxs - mins)
            value = value[gridcols[0], gridcols[1]]
            solutes[name] = _make_df(value, name)
            solutes[name]['alpha'] = value
        data = pandas.concat([biomass, phage] + solutes.values(), sort=False)
    else:
        data = pandas.concat([biomass, phage], sort=False)

    data = data[data["count"] > 0]

    # set up colors
    species = list(species_names) if species_names else data.species.unique().tolist()
    if solute_adjustment:
        species = species + list(solutes.keys())

    if container_colors:
        colors = [container_colors[sp] for sp in container_colors]
    else:
        colors = _generate_colors(species)

    if crop is not None:
        data = data[(data.x >= crop[1][0]) & (data.x < crop[1][1])].copy()
        data["x"] -= crop[1][0]
        shape = (crop[0], crop[1][1] - crop[1][0])

    # set up colors
    dl = float(cfg['space']['dl']) * 1e6
    timestr = "{:.2f} days".format(t).rjust(10, " ")

    data["y"] += 1
    plot = ggplot()
    plot += geoms.geom_tile(
        aes(x="x", y="y", alpha="alpha", fill="factor(species)"), data=data
    )
    plot += scales.scale_alpha_continuous(range=[0, 1], limits=[0, 1], guide=False)
    plot += scales.scale_fill_manual(values=colors, limits=species, drop=False)
    plot += labels.ggtitle("t =" + timestr)
    plot += scales.scale_y_continuous(
        limits=[0, shape[0]],
        labels=lambda x: (dl * np.array(x).astype(int)).astype(str),
        expand=[0, 0],
    )
    plot += scales.scale_x_continuous(
        limits=[0, shape[1]],
        labels=lambda x: (dl * np.array(x).astype(int)).astype(str),
        expand=[0, 0],
    )
    plot += guides(fill=guide_legend(title=""))
    plot += labels.xlab("") + labels.ylab("")
    plot += theme_classic()
    plot += theme(plot_title=element_text(size=8))
    wid = 5.0
    hi = wid * shape[0] / shape[1]
    plot.save(
        outname, height=hi, width=wid, dpi=150, verbose=False, limitsize=False
    )


def _plot_frameorig(
    file,
    outdir,
    shape,
    crop=None,
    max_biomass=6.5e-12,
    max_phage=1000,
    format="png",
    prefix="",
    style=None,
    solute_adjustment=None,
    species_names=None,
    container_colors=None,
):

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # matplotlib deprecation warnings

    outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    biomass, solutes, cfg = _biomass_to_grid(file, species_names)
    gridcols = np.unravel_index(range(np.prod(shape)), shape)

    with np.load(file, allow_pickle=True) as f:
        pass
        # biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        # phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]
        # cfg = {k: v for k, v in f['config']}

        # # set up biomass
        # biomass = [np.rec.array(f[x]) for x in biomass_names]
        # biomass_totals = [np.zeros(np.prod(shape)) for x in biomass_names]
        # [np.add.at(tot, c.location, c.mass) for tot, c in zip(biomass_totals, biomass)]

        # _make_df = lambda d: pandas.DataFrame({"count": d, "x": gridcols[1], "y": gridcols[0]})

        # dfs = [_make_df(d) for d in biomass_totals]
        # for df, species in zip(dfs, biomass_names):
        #     df["species"] = re.sub("_data", "", species)
        # bio = pandas.concat(dfs)
        # minv = 0.20
        # minb = bio[bio["count"] > 0]["count"].min()
        # bio["alpha"] = ((0.95 - minv) * (bio["count"] - minb)) / (
        #     max_biomass - minb
        # ) + minv

        # # set up phage
        # phage = {x: np.rec.array(f[x]) for x in phage_names}
        # phage_totals = {x: np.zeros(np.prod(shape)) for x in phage_names}
        # for name in phage_names:
        #     np.add.at(phage_total[name], phage[name].location, 1)
        # [np.add.at(tot, c.location, c.mass) for tot, c in zip(phage_totals, phage)]

        # phage = [np.rec.array(f[x]) for x in phage_names]
        # # phage_total = {name:  for name in phage_names}
        # for name in phage_names:
        # phg = [_make_df(phage_total.copy()) for _ in phage_names]

        # minv = 0.250
        # if max_phage > 0:
        #     max_phage = np.log10(max_phage)
        # minp = np.log10(phg[phg["count"] > 0]["count"].min())
        # if max_phage <= minp:
        #     max_phage += min(0.01, max_phage * 1.1)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")  # ignore log10(0) warnings
        #     phg["alpha"] = ((0.95 - minv) * (np.log10(phg["count"]) - minp)) / (
        #         max_phage - minp
        #     ) + minv

        # phg["species"] = "phage"

        # if solute_adjustment is not None:
        #     sv = np.log10(f["solute_value"] * 1 + solute_adjustment)
        #     mins = sv[sv.nonzero()].min()
        #     maxs = sv[sv.nonzero()].max()
        #     sv = ((0.95) * (sv - mins)) / (maxs - mins)
        #     solute = pandas.DataFrame(
        #         {
        #             "y": gridcols[0],
        #             "x": gridcols[1],
        #             "alpha": sv[gridcols[0], gridcols[1]],
        #             "species": "substrate",
        #             "count": f["solute_value"][gridcols[0], gridcols[1]],
        #         }
        #     )

        #     data = pandas.concat([bio, phg, solute], sort=False)
        # else:
        #     data = pandas.concat([bio, phg], sort=False)

        # data = data[data["count"] > 0]

        # lims = list(species_names) + ['substrate'] if species_names else data.species.unique()
        # if container_colors:
        #     colors = [container_colors[sp] for sp in container_colors]
        # else:
        #     colors = _generate_colors(lims)

        # if crop is not None:
        #     data = data[(data.x >= crop[1][0]) & (data.x < crop[1][1])].copy()
        #     data["x"] -= crop[1][0]
        #     shape = (crop[0], crop[1][1] - crop[1][0])

        # dl = float(cfg['space:dl']) * 1e6
        # timestr = "{:.2f} days".format(f["t"]).rjust(10, " ")

        # data["y"] += 1
        # plot = ggplot()
        # plot += geoms.geom_tile(
        #     aes(x="x", y="y", alpha="alpha", fill="factor(species)"), data=data
        # )
        # plot += scales.scale_alpha_continuous(range=[0, 1], limits=[0, 1], guide=False)
        # plot += scales.scale_fill_manual(values=colors, limits=lims, drop=False)
        # plot += labels.ggtitle("t =" + timestr)
        # plot += scales.scale_y_continuous(
        #     limits=[0, shape[0]],
        #     labels=lambda x: (dl * np.array(x).astype(int)).astype(str),
        #     expand=[0, 0],
        # )
        # plot += scales.scale_x_continuous(
        #     limits=[0, shape[1]],
        #     labels=lambda x: (dl * np.array(x).astype(int)).astype(str),
        #     expand=[0, 0],
        # )
        # plot += guides(fill=guide_legend(title=""))
        # plot += labels.xlab("") + labels.ylab("")
        # plot += theme_classic()
        # plot += theme(plot_title=element_text(size=8))
        # wid = 5.0
        # hi = wid * shape[0] / shape[1]
        # plot.save(
        #     outname, height=hi, width=wid, dpi=150, verbose=False, limitsize=False
        # )


def plot_all_frames(
    datadir,
    outdir,
    crop=None,
    maxp=None,
    maxb=None,
    format="png",
    nprocesses=4,
    phage_time=False,
    solute_adjustment=None,
    **kwargs,
):
    print(outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    flist = [
        "{}/{}".format(datadir, x)
        for x in os.listdir(datadir)
        if ".npz" in x and not os.path.isfile(f"{outdir}/{x[:-4]}.png")
    ]
    if not flist:
        return
    cfg = getcfg(flist[0])
    shape = tuple(map(int, cfg['space']['shape']))

    species = None
    maxb = maxb or 6.5e-12
    maxp = maxp or 1000
    with mp.Pool(nprocesses) as p:
        species = p.starmap(_get_species, [(fname,) for fname in flist])
        first = tuple(species[0])
        species = tuple({name for each in species for name in each})
        species = first if set(first) == set(species) else species
        if (maxb is None and maxp is None) or phage_time:
            maxes = p.starmap(_get_max, [(fname, shape) for fname in flist])
            if maxb is None:
                maxb = max(x[0] for x in maxes)
            if maxp is None:
                maxp = int(max(x[1] for x in maxes))
            if phage_time:
                return

    # with warnings.catch_warnings(), mp.Pool(1) as p:
    with warnings.catch_warnings(), mp.Pool(nprocesses) as p:
        warnings.simplefilter("ignore")  # ignore log10(0) warnings
        if len(shape) == 2:
            args = (outdir, shape, crop, maxb, maxp, format, "")
            args = args + (solute_adjustment, species)
            p.starmap(_plot_frame, [(f,) + args for f in flist])

        elif len(shape) == 3:
            args = (outdir, shape, maxb, maxp, format, "")
            args = args + (species,)
            p.starmap(_render_3d_frame, [(f,) + args for f in flist])


def _get_max(fname, shape):
    maxb, maxp = 0, 0
    total = np.zeros(np.prod(shape))
    with np.load(fname) as f:
        time = f["t"]
        for name in [x for x in f.keys() if "_data" in x]:
            total[:] = 0
            if "phage" in name:
                np.add.at(total, f[name]["location"], 1)
                maxp = max([maxp, total.max()])
            else:
                np.add.at(total, f[name]["location"], f[name]["mass"])
                maxb = max([maxb, total.max()])
    return time, maxb, maxp


def _get_species(fname):
    with np.load(fname, allow_pickle=True) as f:
        names = [x[:-5] for x in f.keys() if "_data" in x]  # [:-5] removes "_data"
    return names


def _main():
    matplotlib.use("agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video")
    parser.add_argument("-f", "--format")
    parser.add_argument("--crop", nargs=3)
    parser.add_argument("--maxb")
    parser.add_argument("--maxp")
    parser.add_argument("--solute_adjustment")
    parser.add_argument("--interval")
    parser.add_argument("--name")
    parser.add_argument("--prefix")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--jobs")
    parser.add_argument("--old", action="store_true")
    parser.add_argument("-o", "--output")

    args = parser.parse_args()

    if args.video:

        if not args.format:
            raise RuntimeError("Video requires format argument (-f)")
        else:
            kwargs = {"format": args.format}

        if args.crop:
            kwargs["crop"] = [int(x) for x in args.crop]
        if args.maxb:
            kwargs["maxb"] = float(args.maxb)
        if args.maxp:
            kwargs["maxp"] = int(args.maxp)
        if args.solute_adjustment:
            kwargs["solute_adjustment"] = float(args.solute_adjustment)
        if args.interval:
            kwargs["interval"] = int(args.interval)
        if args.name:
            kwargs["name"] = args.name
        if args.prefix:
            kwargs["prefix"] = args.prefix
        if args.jobs:
            kwargs["nprocesses"] = int(args.jobs)
        if args.output:
            kwargs["outdir"] = args.output
        kwargs["progress_bar"] = args.progress
        kwargs["skip_made"] = args.skip

        plot_all_frames(args.video, **kwargs)

if __name__ == "__main__":
    _main()
