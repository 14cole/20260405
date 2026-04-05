from __future__ import annotations

import numpy as np


def render(self) -> None:
    self.last_plot_mode = "compare"
    datasets = self._selected_datasets()
    if len(datasets) != 2:
        self.status.showMessage("Compare: select exactly 2 datasets.")
        return

    az_values_sel = self._selected_values(self.list_az)
    if not az_values_sel:
        self.status.showMessage("Select one or more azimuths to plot.")
        return
    freq_values_sel = self._selected_values(self.list_freq)
    if not freq_values_sel:
        self.status.showMessage("Select one or more frequencies to plot.")
        return
    elev_values_sel = self._selected_values(self.list_elev)
    if not elev_values_sel:
        self.status.showMessage("Select one or more elevations to plot.")
        return
    pol_value_sel = self._single_selection_value(self.list_pol, "polarization")
    if pol_value_sel is None:
        return

    name_a, ds_a = datasets[0]
    name_b, ds_b = datasets[1]

    collected_a = self._collect_azimuth_series(
        ds_a, name_a, az_values_sel, elev_values_sel, freq_values_sel, pol_value_sel
    )
    collected_b = self._collect_azimuth_series(
        ds_b, name_b, az_values_sel, elev_values_sel, freq_values_sel, pol_value_sel
    )
    if collected_a is None:
        self.status.showMessage(f"Compare: '{name_a}' missing selected parameters.")
        return
    if collected_b is None:
        self.status.showMessage(f"Compare: '{name_b}' missing selected parameters.")
        return

    az_a, series_a = collected_a
    az_b, series_b = collected_b

    top_ax, res_ax = self._ensure_compare_axes()
    top_ax.clear()
    res_ax.clear()
    self._style_axes(top_ax)
    self._style_axes(res_ax)

    text_color = self._current_plot_text()
    grid_color = self._current_plot_grid()

    # ── overlay ───────────────────────────────────────────────────────────────
    color_a = "#4fc3f7"
    color_b = "#ff8a65"
    for rcs_disp, label in series_a:
        top_ax.plot(az_a, rcs_disp, color=color_a, linewidth=1.5, label=label)
    for rcs_disp, label in series_b:
        top_ax.plot(az_b, rcs_disp, color=color_b, linewidth=1.5,
                    linestyle="--", label=label)

    top_ax.set_ylabel(self._rcs_axis_label())
    self._update_legend_visibility()

    # ── residual (first series pair only) ────────────────────────────────────
    rcs_a0 = series_a[0][0]
    rcs_b0 = series_b[0][0]

    # Intersection of azimuth grids
    az_a_r = np.round(az_a, 8)
    az_b_r = np.round(az_b, 8)
    mask_a = np.isin(az_a_r, az_b_r)
    mask_b = np.isin(az_b_r, az_a_r)

    if mask_a.sum() < 2:
        res_ax.text(
            0.5, 0.5,
            "No common azimuth points for residual",
            transform=res_ax.transAxes, ha="center", va="center",
            color=text_color, fontsize=8,
        )
    else:
        az_common = az_a[mask_a]
        y_a = rcs_a0[mask_a]
        y_b = rcs_b0[mask_b]
        residual = y_a - y_b

        res_ax.axhline(0, color=grid_color, linewidth=0.8, linestyle="--")
        res_ax.plot(az_common, residual, color="#a5d6a7", linewidth=1.2,
                    label=f"{name_a} − {name_b}")
        res_ax.fill_between(az_common, residual, 0, alpha=0.15, color="#a5d6a7")
        res_ax.set_ylabel("Residual (dB)", fontsize=8)

        # ── statistics ───────────────────────────────────────────────────────
        finite = np.isfinite(residual)
        if finite.sum() > 1:
            res_fin = residual[finite]
            mean_err = float(np.mean(res_fin))
            rms_err  = float(np.sqrt(np.mean(res_fin ** 2)))
            max_err  = float(np.max(np.abs(res_fin)))

            fin_both = finite & np.isfinite(y_a) & np.isfinite(y_b)
            if fin_both.sum() > 1:
                corr = float(np.corrcoef(y_a[fin_both], y_b[fin_both])[0, 1])
                corr_str = f"   Corr: {corr:.4f}"
            else:
                corr_str = ""

            stats_text = (
                f"Mean: {mean_err:+.2f} dB   "
                f"RMS: {rms_err:.2f} dB   "
                f"Max|err|: {max_err:.2f} dB"
                + corr_str
            )
            top_ax.set_title(stats_text, fontsize=8, color=text_color, pad=4)

    res_ax.set_xlabel("Azimuth (deg)")
    self._apply_plot_limits()
    self.status.showMessage("Compare plot updated.")
