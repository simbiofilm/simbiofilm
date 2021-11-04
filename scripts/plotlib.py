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
from matplotlib.colors import cnames, hex2color, rgb_to_hsv, hsv_to_rgb
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


import simbiofilm as sb

from tempfile import mkstemp


@contextmanager
def ignore_copywarn():
    pandas.options.mode.chained_assignment = None
    yield
    pandas.options.mode.chained_assignment = "warn"


def _getcorners(ix, spacing=0.02):
    ix1 = np.array(ix, dtype=float)
    ix2 = ix1 + 1 - spacing
    ix1 += spacing
    return ix1, ix2


def _getcolor(s, r, i, p, maxp, maxb):
    if p > 0:
        minv = 0.250
        # phage_factor = ((0.95 - minv) * np.log10(p)) / np.log10(maxp) + minv
        # phage_factor = ((0.95 - minv) * p / maxp) + minv
        # phage_factor = 1 - (((0.95 - minv) * p / maxp) + minv)
        phage_factor = 1 - (0.95 - minv) * np.log10(p) / np.log10(maxp) + minv
    else:
        phage_factor = 1
    total_biomass = s + r + i
    # biomass_factor = 1 - total_biomass / maxb

    # red = (s / total_biomass * (1 - phage_factor)) ** 2
    # blu = (r / total_biomass * (1 - phage_factor)) ** 2
    # grn = (i / total_biomass * (1 - phage_factor)) ** 2

    red = 0.8 * (s / total_biomass * phage_factor) ** 2
    blu = 0.8 * (r / total_biomass * phage_factor) ** 2
    grn = 0.8 * (i / total_biomass * phage_factor) ** 2

    return red, grn, blu, 1 - (total_biomass / maxb) ** 3
    # return red, grn, blu, total_biomass / maxb


def _make_base(shape):
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


def _render_3d_frame(
    file, outdir, shape, max_biomass=6.5e-12, max_phage=1000, format="png", prefix=""
):
    outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    with np.load(file) as f:
        particle_names = [x for x in list(f.keys()) if x.endswith("data")]
        sus = np.zeros(shape)
        res = np.zeros(shape)
        inf = np.zeros(shape)
        phg = np.zeros(shape, dtype=int)
        biomass = {
            "Species1_data": sus,
            "Species2_data": res,
            "infected_data": inf,
            "phage_data": phg,
        }

        for name, dat in zip(particle_names, (f[x] for x in particle_names)):
            bio = biomass[name]
            for ind in dat:
                if "phage" in name:
                    bio[np.unravel_index(ind["location"], shape)] += 1
                else:
                    bio[np.unravel_index(ind["location"], shape)] += ind["mass"]

        objects = [_make_base(shape), Background("color", [0.9, 0.9, 0.9])]

        for ix in product(*map(range, shape)):
            s, r, i, p = sus[ix], res[ix], inf[ix], phg[ix]
            if s == 0 and r == 0 and i == 0:
                continue
            col = _getcolor(s, r, i, p, max_phage, max_biomass)
            ix1, ix2 = _getcorners(ix, 0.1)

            box = Box(
                ix,
                np.array(ix) + 1,
                Texture(Pigment("color", "rgbf", col)),
                Finish("ambient", 1.0, "diffuse", 1.0),
            )  # , 'no_shadow')
            objects.append(box)

            # box = Box(ix,
            #           np.array(ix) + 1, Texture(Pigment('color', col, 'filter', alpha)),
            #           Finish('ambient', 1.0, 'diffuse', 1.0))  # , 'no_shadow')
            # objects.append(box)

            # light = LightSource([shape[0], -shape[1] / 2, -shape[2] / 2], 'color', [1, 1, 1],
            #                     'spotlight', 'radius', shape[0], 'adaptive', 1, 'jitter',
            #                     'point_at', [0, shape[1] / 2, shape[2] / 2])
            # objects.append(light)

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
        tmpfname = mkstemp(".pov", dir="tmpf")[1]
        Scene(camera, objects=objects).render(
            outname, tempfile=tmpfname, width=3000, height=3000
        )
        print(f"Rendered {outname}")


_colors = {
    k: rgb_to_hsv(hex2color(v))[0]
    for k, v in cnames.items()
    if np.all(rgb_to_hsv(hex2color(v))[1:] == 1)
}


class Frame:
    # min and max are the scale
    minb = 1e-12 / 3
    maxb = 6.31e-12
    maxp = 0
    maxp = np.log10(1000)
    # minv and maxv determine the saturation.
    pminv = 0.3
    pmaxv = 0.99
    bminv = 0.2
    bmaxv = 0.95

    def __init__(self, containers, types, colors, dl, name, time):
        """
        Color options:
           aqua        blue      chartreuse cyan        darkorange
           deepskyblue fuchsia   gold       lime        magenta
           orange      orangered red        springgreen yellow
       Or a number [0, 1]
        """
        colors = [_colors[c] if c in _colors else float(c) for c in colors]
        self.types = types
        ctc = (containers, types, colors)
        self.phage = [(c, r) for c, t, r in zip(*ctc) if t == "phage"]
        self.solute = [(c, r) for c, t, r in zip(*ctc) if t == "solute"]
        self.biomass = [(c, r) for c, t, r in zip(*ctc) if t == "biomass"]
        self.shape = containers[0].shape
        self.name = name
        self.dl = dl
        self.time = time

    def render(self, solute=None):
        # HSV -> RGB
        # Biomass color -> Hue
        # Biomass value out of max -> saturation
        # phage value out of max -> value
        # Solute ??????
        frame = np.zeros((3,) + self.shape)  # NxMx3
        hue, saturation, value = frame
        for phage, _ in self.phage:  # color ignored
            value += phage
        value[value > 0] = (
            1
            - self.pminv
            - (self.pmaxv - self.pminv) * np.log10(value[value > 0]) / self.maxp
        )
        color = np.zeros_like(hue)
        biomass = np.zeros_like(hue)
        for i, (bio, col) in enumerate(self.biomass):
            hasbio = bio > 0
            mixed = (biomass > 0) & (bio > 0) & (np.abs(color - col) <= 0.5)
            single = (biomass == 0) & (bio > 0)
            circle = (biomass > 0) & (bio > 0) & (np.abs(color - col) > 0.5)
            linear = mixed | single
            biomass += bio
            color[circle] = (
                color[circle] - (1 - col) * bio[circle] / biomass[circle] + 1
            )
            color[linear] = color[linear] + col * bio[linear] / biomass[linear]
        hue[:] = color
        value[value == 0] = 1  # Default to white
        # hue[saturation > 0] /= saturation[saturation > 0]
        saturation[biomass > 0] = self.bminv + (self.bmaxv - self.bminv) * (
            biomass[biomass > 0] - self.minb
        ) / (self.maxb - self.minb)
        frame = np.moveaxis(frame, 0, -1)
        # TODO: getting negative values sometimes -- species mixing looks bad.
        return hsv_to_rgb(frame)


def biomass_colors(n, cb_safe=True):
    # if n <= 3:
    #     yield from ["#1b9e77", "#d95f02", "#7570b3"]
    # elif n == 4:
    yield "blue"
    yield "red"
    yield "orange"
    yield "magenta"
    yield "deepskyblue"
    # yield "magenta"
    # yield "springgreen"


def to_grid(shape, locations, values):
    ret = np.zeros(shape)
    np.add.at(ret.reshape(-1), locations, values)
    return ret


def _plot_frame_matplotlib(
    file,
    max_biomass=6.5e-12,
    max_phage=1000,
    format="png",
    prefix="",
    style=None,
    solute_adjustment=None,
):

    warnings.filterwarnings(
        "ignore", category=UserWarning
    )  # matplotlib deprecation warnings

    # outname = f'{outdir}/{prefix}{file.split("/")[-1][:-4]}.{format}'
    with np.load(file) as f:
        config = sb.parse_params(*f["config"].T)
        shape = [int(x) for x in config.space.shape]
        dl = config.space.dl * 1e6
        biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]
        if solute_adjustment:
            sv = np.log10(f["solute_value"] * 1 + solute_adjustment)

        # TODO: generic this.
        # TODO: get names and colors from config
        containers = []
        types = []
        hues = []
        col = biomass_colors(len(biomass_names))
        for k in f.keys():
            if k.lower().endswith("_value"):
                containers.append(f[k])
                types.append("solute")
                hues.append("magenta")
            if k.lower().endswith("_data"):
                if "phage" in k.lower():
                    types.append("phage")
                    containers.append(
                        to_grid(shape, f[k]["location"], np.ones_like(f[k]["location"]))
                    )
                    hues.append(0)
                if "species" in k.lower() or "infected" in k.lower():
                    containers.append(to_grid(shape, f[k]["location"], f[k]["mass"]))
                    types.append("biomass")
                    hues.append(next(col))

        fig, ax = plt.subplots(figsize=np.array(shape) / 2)
        frame = Frame(containers, types, hues)
        pltspace = (0, shape[1] * dl, 0, shape[0] * dl)
        ax.imshow(frame.render(), extent=pltspace, origin="lower", resample=True)
        fig.savefig("test.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
        # return fig, ax


def _make_frame(fname, crop=None, solute_adjustment=None):
    with np.load(fname, allow_pickle=True) as f:
        config = sb.parse_params(*f["config"].T)
        shape = [int(x) for x in config.space.shape]
        biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]
        if solute_adjustment:
            sv = np.log10(f["solute_value"] * 1 + solute_adjustment)

        # TODO: for legend, use Patch: https://bit.ly/2Mi64o5
        # TODO: Related, name containers by config. Also colors!

        # TODO: generic this.
        # TODO: get names and colors from config
        containers = []
        types = []
        hues = []
        col = biomass_colors(len(biomass_names))
        for k in f.keys():
            if k.lower().endswith("_value"):
                containers.append(f[k])
                types.append("solute")
                hues.append("magenta")
            if k.lower().endswith("_data"):
                if "phage" in k.lower():
                    types.append("phage")
                    containers.append(
                        to_grid(shape, f[k]["location"], np.ones_like(f[k]["location"]))
                    )
                    hues.append(0)
                # elif "matrix" in k.lower():
                else:
                    containers.append(to_grid(shape, f[k]["location"], f[k]["mass"]))
                    types.append("biomass")
                    hues.append(next(col))

        name = fname.split("/")[-1][:-4]  # removes npz
        return Frame(
            containers, types, hues, config.space.dl * 1e6, name, float(f["t"])
        )


@contextmanager
def make_video(
    datadir,
    format,
    crop=None,
    maxb=6.5e-12,
    maxp=1000,
    solute_adjustment=None,
    interval=20,
    name=None,
    outdir=".",
    prefix="",
    progress_bar=False,
    skip_made=False,
):
    flist = sorted(
        [
            "{}/{}".format(datadir, x)
            for x in os.listdir(datadir)
            if x.lower().endswith(".npz")
        ]
    )
    # flist = flist[360:361]
    name = name if name is not None else datadir.split("/")[-1]

    print("Making frames")
    frames = [_make_frame(f, crop, solute_adjustment) for f in flist]
    fs = frames[0]
    shape = np.array(fs.shape)[::-1]
    fig, ax = plt.subplots(figsize=shape / 25)  # TODO size args
    pltspace = (0, shape[0] * fs.dl, 0, shape[1] * fs.dl)
    images = []
    print("Plotting frames")
    if format in fig.canvas.get_supported_filetypes():
        os.makedirs(f"{outdir}/{name}_frames", exist_ok=True)
    else:
        os.makedirs(outdir, exist_ok=True)

    progress = tqdm80 if progress_bar else (lambda x: x)
    if format in fig.canvas.get_supported_filetypes():
        odir = f"{outdir}/{name}_frames"
        im = ax.imshow(np.zeros((50, 300, 3)), extent=pltspace, origin="lower")
        fig.savefig("tmp.png")
        for f in progress(frames):
            ofname = f"{odir}/{f.name}.{format}"
            if skip_made and os.path.isfile(ofname):
                continue
            print(f.name)
            im.set_data(f.render())
            timestr = "{:.2f} days".format(f.time).rjust(10, " ")
            ax.set_title("t = " + timestr)
            ax.draw_artist(ax.patch)
            ax.draw_artist(im)
            plt.savefig(ofname, dpi=150, bbox_inches="tight")

    else:
        for f in frames:
            im = ax.imshow(f.render(), extent=pltspace, origin="lower", animated=True)
            images.append([im])

        ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True)
        print("Making video")
        ani.save(f"{name}.{format}")

    plt.close()


def _plot_frame(
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
    with np.load(file, allow_pickle=True) as f:
        biomass_names = [x for x in f.keys() if "_data" in x and "phage" not in x]
        phage_names = [x for x in f.keys() if "_data" in x and "phage" in x]

        cfg = {k: v for k, v in f['config']}

        gridcols = np.unravel_index(range(np.prod(shape)), shape)

        # set up biomass
        biomass = [np.rec.array(f[x]) for x in biomass_names]
        biomass_totals = [np.zeros(np.prod(shape)) for x in biomass_names]
        [np.add.at(tot, c.location, c.mass) for tot, c in zip(biomass_totals, biomass)]

        def _make_df(data):
            return pandas.DataFrame({"count": data, "x": gridcols[1], "y": gridcols[0]})

        dfs = [_make_df(d) for d in biomass_totals]
        for df, species in zip(dfs, biomass_names):
            df["species"] = re.sub("_data", "", species)
        bio = pandas.concat(dfs)
        minv = 0.20
        minb = bio[bio["count"] > 0]["count"].min()
        bio["alpha"] = ((0.95 - minv) * (bio["count"] - minb)) / (
            max_biomass - minb
        ) + minv

        # phage
        # phage = np.rec.array(f[phage_names[0]])
        phage_total = np.zeros(np.prod(shape))
        # np.add.at(phage_total, phage.location, 1)
        phg = _make_df(phage_total)

        minv = 0.250
        if max_phage > 0:
            max_phage = np.log10(max_phage)
        minp = np.log10(phg[phg["count"] > 0]["count"].min())
        if max_phage <= minp:
            max_phage += min(0.01, max_phage * 1.1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore log10(0) warnings
            phg["alpha"] = ((0.95 - minv) * (np.log10(phg["count"]) - minp)) / (
                max_phage - minp
            ) + minv

        phg["species"] = "phage"

        # combine, and clear out empty cells. Otherwise, alpha is minv


        if solute_adjustment is not None:
            sv = np.log10(f["solute_value"] * 1 + solute_adjustment)
            mins = sv[sv.nonzero()].min()
            maxs = sv[sv.nonzero()].max()
            sv = ((0.95) * (sv - mins)) / (maxs - mins)
            solute = pandas.DataFrame(
                {
                    "y": gridcols[0],
                    "x": gridcols[1],
                    "alpha": sv[gridcols[0], gridcols[1]],
                    "species": "substrate",
                    "count": f["solute_value"][gridcols[0], gridcols[1]],
                }
            )

            data = pandas.concat([bio, phg, solute], sort=False)
        else:
            data = pandas.concat([bio, phg], sort=False)

        data = data[data["count"] > 0]
        lims = list(species_names) + ['substrate'] if species_names else data.species.unique()

        if container_colors:
            colors = [container_colors[sp] for sp in container_colors]
        else:
            colors = []
            cgen = {
                'species' : (n for n in ['red', 'blue', 'orange']),
                'matrix': (n for n in ['orange', 'peru', 'saddlebrown']),
                'infected': (n for n in ['green', 'lime', 'aquamarine']),
                'phage': (n for n in ['black', 'orange']),
                'solute': (n for n in ['#3c1978', 'orange']),
            }

            try:
                for sp in lims:
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
                msg = f'More species than colors. Check against _plot_frame: \n{lims}'
                msg += '\n\nPairs:\n'
                msg += join([f'{sp}: {c}' for sp, c in itt.zip_longest(lims, colors, fillvalue='Unassigned')], '\n')
                raise RuntimeError(msg)

            if crop is not None:
                data = data[(data.x >= crop[1][0]) & (data.x < crop[1][1])].copy()
                data["x"] -= crop[1][0]
                shape = (crop[0], crop[1][1] - crop[1][0])

        dl = float(cfg['space:dl']) * 1e6
        timestr = "{:.2f} days".format(f["t"]).rjust(10, " ")

        data["y"] += 1
        plot = ggplot()
        plot += geoms.geom_tile(
            aes(x="x", y="y", alpha="alpha", fill="factor(species)"), data=data
        )
        plot += scales.scale_alpha_continuous(range=[0, 1], limits=[0, 1], guide=False)
        plot += scales.scale_fill_manual(values=colors, limits=lims, drop=False)
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


def noframe(f, outdir):
    outname = "{}/{}.png".format(outdir, f[:-4])
    return not os.path.isfile(outname)


def plot_all_frames(
    datadir,
    outdir,
    crop=None,
    maxp=None,
    maxb=None,
    format="png",
    style=None,
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
        if ".npz" in x and noframe(x, outdir)
    ]
    if not flist:
        return
    cfg = sb.getcfg(flist[0])
    shape = tuple(map(int, cfg.space.shape))

    if len(shape) == 2:
        maxb, maxp = 6.5e-12, 1000
        # solute_adjustment = 0.05
        if nprocesses > 1:
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

            with warnings.catch_warnings(), mp.Pool(nprocesses) as p:
                warnings.simplefilter("ignore")  # ignore log10(0) warnings
                p.starmap(
                    _plot_frame,
                    [
                        (
                            f,
                            outdir,
                            shape,
                            crop,
                            maxb,
                            maxp,
                            format,
                            "",
                            style,
                            solute_adjustment,
                            species,
                        )
                        for f in flist
                    ],
                )
        else:
            if maxb is None and maxp is None:
                maxes = [_get_max(fname, shape) for fname in flist]
                if maxb is None:
                    maxb = max(x[0] for x in maxes)
                if maxp is None:
                    maxp = int(max(x[1] for x in maxes))
            for f in flist:
                print(f)
                _plot_frame(
                    f,
                    outdir,
                    shape,
                    crop,
                    maxb,
                    maxp,
                    format,
                    style=style,
                    solute_adjustment=solute_adjustment,
                )
                break  # FIXME REMOVE ME -- debugging only
    if len(shape) == 3:
        maxb, maxp = 6.5e-12, 1000
        if nprocesses > 1:
            with warnings.catch_warnings(), mp.Pool(nprocesses) as p:
                warnings.simplefilter("ignore")  # ignore log10(0) warnings
                p.starmap(
                    _render_3d_frame,
                    [(f, outdir, shape, maxb, maxp, format, "") for f in flist],
                )
        else:
            for f in flist:
                _render_3d_frame(f, outdir, shape, 6.5e-12, 1000, format, "")


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
        names = [x for x in f.keys() if "_data" in x and "phage" not in x]
    names = [re.sub("_data", "", n) for n in names]
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

        # print(kwargs)
        if args.old:
            plot_all_frames(args.video, **kwargs)
        else:
            make_video(args.video, **kwargs)


if __name__ == "__main__":
    _main()
