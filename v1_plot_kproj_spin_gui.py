#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive KPROJ band-unfolding scatter plot

Files:
  band_kproj_1.dat -> spin up, red
  band_kproj_2.dat -> spin down, blue

Each data file should contain 3 columns:
  x   energy/eV   weight

Slider meaning:
  Show top N% weighted points. Moving the slider right shows more points.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons


def load_band(path: str) -> np.ndarray:
    """Load 3-column band data, ignoring comment lines beginning with #."""
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} does not look like a 3-column data file")
    return data[:, :3]


def select_by_top_percent(data: np.ndarray, percent: float) -> np.ndarray:
    """Return points with the largest weights, controlled by percent in [0, 100]."""
    percent = float(np.clip(percent, 0.1, 100.0))
    weights = data[:, 2]
    cutoff = np.percentile(weights, 100.0 - percent)
    return data[weights >= cutoff]


def weight_to_size(weights: np.ndarray, s_min: float = 4.0, s_max: float = 36.0) -> np.ndarray:
    """Map weights to marker sizes for better visibility."""
    if len(weights) == 0:
        return weights
    w_min, w_max = np.min(weights), np.max(weights)
    if np.isclose(w_min, w_max):
        return np.full_like(weights, (s_min + s_max) / 2.0, dtype=float)
    scaled = (weights - w_min) / (w_max - w_min)
    return s_min + scaled * (s_max - s_min)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive KPROJ spin band scatter plot")
    parser.add_argument("--up", default="band_kproj_1.dat", help="spin-up data file, default: band_kproj_1.dat")
    parser.add_argument("--down", default="band_kproj_2.dat", help="spin-down data file, default: band_kproj_2.dat")
    parser.add_argument("--start", type=float, default=15.0, help="initial top-weight percentage shown, default: 15")
    parser.add_argument("--emin", type=float, default=None, help="optional lower energy limit")
    parser.add_argument("--emax", type=float, default=None, help="optional upper energy limit")
    args = parser.parse_args()

    up = load_band(args.up)
    down = load_band(args.down)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, bottom=0.22, right=0.96, top=0.94)

    ax.set_title("KPROJ unfolded band structure")
    ax.set_xlabel("k-path / x")
    ax.set_ylabel("Energy (eV)")

    all_x = np.concatenate([up[:, 0], down[:, 0]])
    all_y = np.concatenate([up[:, 1], down[:, 1]])
    ax.set_xlim(np.min(all_x), np.max(all_x))
    ax.set_ylim(args.emin if args.emin is not None else np.min(all_y),
                args.emax if args.emax is not None else np.max(all_y))
    ax.grid(True, alpha=0.25)

    initial_percent = float(np.clip(args.start, 0.1, 100.0))
    up_show = select_by_top_percent(up, initial_percent)
    down_show = select_by_top_percent(down, initial_percent)

    sc_up = ax.scatter(
        up_show[:, 0], up_show[:, 1],
        s=weight_to_size(up_show[:, 2]),
        c="red", alpha=0.65, linewidths=0, label="up"
    )
    sc_down = ax.scatter(
        down_show[:, 0], down_show[:, 1],
        s=weight_to_size(down_show[:, 2]),
        c="blue", alpha=0.65, linewidths=0, label="down"
    )
    leg = ax.legend(loc="best")

    text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none")
    )

    slider_ax = fig.add_axes([0.18, 0.09, 0.68, 0.035])
    percent_slider = Slider(
        ax=slider_ax,
        label="显示最高权重点比例 (%)",
        valmin=0.1,
        valmax=100.0,
        valinit=initial_percent,
        valstep=0.1,
    )

    check_ax = fig.add_axes([0.02, 0.065, 0.10, 0.08])
    checks = CheckButtons(check_ax, ["up", "down"], [True, True])

    def redraw(percent: float) -> None:
        nonlocal up_show, down_show
        up_show = select_by_top_percent(up, percent)
        down_show = select_by_top_percent(down, percent)

        sc_up.set_offsets(up_show[:, :2])
        sc_up.set_sizes(weight_to_size(up_show[:, 2]))

        sc_down.set_offsets(down_show[:, :2])
        sc_down.set_sizes(weight_to_size(down_show[:, 2]))

        text.set_text(
            f"Top weight: {percent:.1f}%\n"
            f"up: {len(up_show)} / {len(up)} points\n"
            f"down: {len(down_show)} / {len(down)} points"
        )
        fig.canvas.draw_idle()

    def on_slider(val: float) -> None:
        redraw(val)

    def on_check(label: str) -> None:
        if label == "up":
            sc_up.set_visible(not sc_up.get_visible())
        elif label == "down":
            sc_down.set_visible(not sc_down.get_visible())
        fig.canvas.draw_idle()

    percent_slider.on_changed(on_slider)
    checks.on_clicked(on_check)
    redraw(initial_percent)

    plt.show()


if __name__ == "__main__":
    main()
