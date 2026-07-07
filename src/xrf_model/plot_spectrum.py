"""
Plotting helper for the simulated XRF detector spectrum.
"""
from __future__ import annotations

import numpy as np


def plot_and_save_spectrum(energy_kev, spectrum, df_fluo, out_path, conc_label: str,
                           geo_info: dict | None = None):
    """
    Plot the simulated spectrum on a semi-log scale and save to file.

    Parameters
    ----------
    energy_kev  : 1D array of energies [keV]
    spectrum    : 1D array of intensities [ph/s/keV]
    df_fluo     : DataFrame with columns Element, Line, E_line_keV, Intensity_ph_s
    out_path    : path-like — destination for the saved PNG
    conc_label  : string shown in the legend and title
    geo_info    : optional dict with keys src_dist_cm, src_angle_deg,
                  det_dist_cm, det_angle_deg
    """
    MIN_LABEL_ENERGY_KEV = 1.5
    INTENSITY_THRESHOLD  = 0.001
    E_PLOT_MIN, E_PLOT_MAX = 1.0, 20.0  # keV

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))

    spec_max = spectrum.max() if spectrum.max() > 0 else 1.0
    spectrum_norm = np.maximum(spectrum / spec_max, 1e-30)

    ax.semilogy(energy_kev, spectrum_norm,
                lw=0.8, color="steelblue", label=conc_label)

    ax.set_xlim(E_PLOT_MIN, E_PLOT_MAX)
    y_min, y_max = ax.get_ylim()

    visible = df_fluo[
        np.isfinite(df_fluo["E_line_keV"]) &
        (df_fluo["Intensity_ph_s"] > 0) &
        (df_fluo["E_line_keV"] >= MIN_LABEL_ENERGY_KEV) &
        (df_fluo["E_line_keV"] <= E_PLOT_MAX)
    ]
    if not visible.empty:
        i_max = visible["Intensity_ph_s"].max()
        visible = visible[visible["Intensity_ph_s"] >= INTENSITY_THRESHOLD * i_max]

    label_y_positions = [y_min * 4.0, y_min * 60.0]
    for i, (_, row) in enumerate(visible.iterrows()):
        E0 = row["E_line_keV"]
        ax.axvline(E0, color="lightgray", lw=0.4, alpha=0.6)
        ax.text(E0, label_y_positions[i % 2], f"{row['Element']} {row['Line']}",
                fontsize=6, va="bottom", rotation=90, color="#555555")

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Normalized intensity (a.u.)")

    if geo_info is not None:
        geo_str = (
            f"src: h={geo_info['src_dist_cm']:.2f} cm, {geo_info['src_angle_deg']:.1f}° from normal  |  "
            f"det: h={geo_info['det_dist_cm']:.2f} cm, {geo_info['det_angle_deg']:.1f}° from normal"
        )
        title = f"Simulated XRF spectrum — {conc_label}\n{geo_str}"
    else:
        title = f"Simulated XRF spectrum — {conc_label}"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="major", alpha=0.35, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"→ Saved spectrum plot to {out_path}")
