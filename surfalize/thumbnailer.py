#!/usr/bin/env python3

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from surfalize import Surface


def main():
    argv = sys.argv
    with open("/tmp/surfalize-thumb.log", "a") as f:
        f.write(f"started {argv}\n")

    if len(argv) != 4:
        return 1

    input_file = Path(argv[1])
    output_file = Path(argv[2])
    size = int(argv[3])

    surface = Surface.load(input_file)

    dpi = 72
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(surface.data, cmap="jet", interpolation="nearest")
    ax.axis("off")

    fig.savefig(output_file, format="png")
    plt.close(fig)

    return 0