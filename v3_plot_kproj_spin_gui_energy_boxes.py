#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive KPROJ unfolded band scatter plot with separate UP/DOWN sliders and energy range boxes.

Input files:
  band_kproj_1.dat  -> spin up, red
  band_kproj_2.dat  -> spin down, blue

Each data file should contain at least 3 columns:
  x   energy/y   weight

Run:
  python plot_kproj_spin_gui_energy_boxes.py

Optional:
  python plot_kproj_spin_gui_energy_boxes.py --emin -2 --emax 2
  python plot_kproj_spin_gui_energy_boxes.py --up band_kproj_1.dat --down band_kproj_2.dat
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button, TextBox


def load_band_data(filename):
    """Load x, y, weight from a KPROJ-like 3-column data file."""
    data = np.loadtxt(filename, comments=["#", "!"])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"{filename} must have at least 3 columns: x, y, weight")

    x = data[:, 0]
    y = data[:, 1]
    w = data[:, 2]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    return x[mask], y[mask], w[mask]


def sort_by_weight_desc(x, y, w):
    """Sort points by spectral weight from high to low."""
    idx = np.argsort(w)[::-1]
    return x[idx], y[idx], w[idx]


def weight_to_size(w, min_size=2.0, max_size=35.0):
    """Map weights to scatter marker size."""
    w = np.asarray(w)
    if len(w) == 0:
        return w
    wmin, wmax = np.min(w), np.max(w)
    if np.isclose(wmin, wmax):
        return np.full_like(w, (min_size + max_size) / 2.0, dtype=float)
    wn = (w - wmin) / (wmax - wmin)
    return min_size + wn * (max_size - min_size)


def main():
    parser = argparse.ArgumentParser(description="Plot KPROJ unfolded spin bands with two sliders and energy min/max boxes.")
    parser.add_argument("--up", default="band_kproj_1.dat", help="UP data file, default: band_kproj_1.dat")
    parser.add_argument("--down", default="band_kproj_2.dat", help="DOWN data file, default: band_kproj_2.dat")
    parser.add_argument("--emin", type=float, default=-3.0, help="Initial minimum energy value, default: -3")
    parser.add_argument("--emax", type=float, default=3.0, help="Initial maximum energy value, default: 3")
    parser.add_argument("--alpha", type=float, default=0.65, help="Point transparency, default: 0.65")
    parser.add_argument("--dpi", type=int, default=300, help="DPI when saving PNG, default: 300")
    args = parser.parse_args()

    # Load and sort by weight, largest first.
    x_up, y_up, w_up = sort_by_weight_desc(*load_band_data(args.up))
    x_dn, y_dn, w_dn = sort_by_weight_desc(*load_band_data(args.down))

    n_up_total = len(x_up)
    n_dn_total = len(x_dn)

    # Initial display ratio.
    init_up_ratio = 30.0
    init_dn_ratio = 30.0

    def pick_top_percent(x, y, w, percent):
        n = int(round(len(x) * percent / 100.0))
        n = max(0, min(n, len(x)))
        return x[:n], y[:n], w[:n]

    xu, yu, wu = pick_top_percent(x_up, y_up, w_up, init_up_ratio)
    xd, yd, wd = pick_top_percent(x_dn, y_dn, w_dn, init_dn_ratio)

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.10, right=0.82, bottom=0.32, top=0.94)

    sc_up = ax.scatter(
        xu, yu,
        s=weight_to_size(wu),
        c="red",
        alpha=args.alpha,
        edgecolors="none",
        label="UP"
    )
    sc_dn = ax.scatter(
        xd, yd,
        s=weight_to_size(wd),
        c="blue",
        alpha=args.alpha,
        edgecolors="none",
        label="DOWN"
    )

    ax.set_xlabel("k-path / x")
    ax.set_ylabel("Energy / y")
    ax.set_title("KPROJ unfolded band structure")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")

    # Energy axis range, controlled by text boxes below.
    ax.set_ylim(args.emin, args.emax)

    # Slider axes.
    ax_up_slider = plt.axes([0.15, 0.19, 0.58, 0.03])
    ax_dn_slider = plt.axes([0.15, 0.14, 0.58, 0.03])

    up_slider = Slider(
        ax=ax_up_slider,
        label="UP points (%)",
        valmin=0.0,
        valmax=100.0,
        valinit=init_up_ratio,
        valstep=1.0,
    )
    dn_slider = Slider(
        ax=ax_dn_slider,
        label="DOWN points (%)",
        valmin=0.0,
        valmax=100.0,
        valinit=init_dn_ratio,
        valstep=1.0,
    )

    # Energy min/max input boxes. Press Enter in a box, or click Apply E range.
    ax_emin_box = plt.axes([0.15, 0.065, 0.16, 0.04])
    ax_emax_box = plt.axes([0.39, 0.065, 0.16, 0.04])
    emin_box = TextBox(ax_emin_box, "E min", initial=f"{args.emin:g}")
    emax_box = TextBox(ax_emax_box, "E max", initial=f"{args.emax:g}")

    ax_apply_e = plt.axes([0.60, 0.065, 0.13, 0.04])
    apply_e_button = Button(ax_apply_e, "Apply E")

    # Check buttons: show/hide spin channels.
    ax_check = plt.axes([0.84, 0.70, 0.12, 0.12])
    check = CheckButtons(ax_check, ["UP", "DOWN"], [True, True])

    # Save button.
    ax_save = plt.axes([0.84, 0.56, 0.12, 0.06])
    save_button = Button(ax_save, "Save PNG")

    status = ax.text(
        0.01, 0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none")
    )

    visible_up = True
    visible_dn = True

    def update_status(nu, nd):
        status.set_text(
            f"UP: {nu}/{n_up_total} points\nDOWN: {nd}/{n_dn_total} points"
        )

    def apply_energy_range():
        """Read E min/E max text boxes and apply them to the energy axis."""
        try:
            emin = float(emin_box.text)
            emax = float(emax_box.text)
        except ValueError:
            print("E min / E max must be numbers, for example -3 and 3.")
            return

        if emin >= emax:
            print("E min must be smaller than E max.")
            return

        ax.set_ylim(emin, emax)

    def update(_=None):
        nonlocal visible_up, visible_dn

        apply_energy_range()

        up_percent = up_slider.val
        dn_percent = dn_slider.val

        xu, yu, wu = pick_top_percent(x_up, y_up, w_up, up_percent)
        xd, yd, wd = pick_top_percent(x_dn, y_dn, w_dn, dn_percent)

        sc_up.set_offsets(np.column_stack([xu, yu]) if len(xu) else np.empty((0, 2)))
        sc_up.set_sizes(weight_to_size(wu))
        sc_up.set_visible(visible_up)

        sc_dn.set_offsets(np.column_stack([xd, yd]) if len(xd) else np.empty((0, 2)))
        sc_dn.set_sizes(weight_to_size(wd))
        sc_dn.set_visible(visible_dn)

        update_status(len(xu), len(xd))
        fig.canvas.draw_idle()

    def toggle(label):
        nonlocal visible_up, visible_dn
        if label == "UP":
            visible_up = not visible_up
        elif label == "DOWN":
            visible_dn = not visible_dn
        update()

    def save_png(_event):
        outfile = "kproj_spin_unfolded.png"
        fig.savefig(outfile, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {outfile}")

    up_slider.on_changed(update)
    dn_slider.on_changed(update)
    check.on_clicked(toggle)
    save_button.on_clicked(save_png)
    apply_e_button.on_clicked(update)
    emin_box.on_submit(update)
    emax_box.on_submit(update)

    update_status(len(xu), len(xd))
    plt.show()


if __name__ == "__main__":
    main()
