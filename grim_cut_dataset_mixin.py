from __future__ import annotations

import os

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QRadioButton,
    QTableWidgetItem,
    QVBoxLayout,
)

from grim_dataset import RcsGrid


class AxisCropDialog(QDialog):
    """Single dialog for axis crop: shows per-axis min/max spinboxes with live shape preview."""

    def __init__(
        self,
        dataset: RcsGrid,
        n_datasets: int = 1,
        presel_az=None,
        presel_el=None,
        presel_freq=None,
        presel_pol=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Axis Crop")
        self._dataset = dataset
        layout = QVBoxLayout(self)

        desc = QLabel(
            f"Cropping {n_datasets} dataset(s).  Adjust ranges for each axis — "
            "leave at full extent to keep all samples on that axis."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        grid = QGridLayout()
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(4, 1)

        def _make_spin(lo, hi, val):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(6)
            s.setValue(val)
            return s

        az_lo = float(dataset.azimuths.min())
        az_hi = float(dataset.azimuths.max())
        el_lo = float(dataset.elevations.min())
        el_hi = float(dataset.elevations.max())
        fr_lo = float(dataset.frequencies.min())
        fr_hi = float(dataset.frequencies.max())

        self._az_lo_full, self._az_hi_full = az_lo, az_hi
        self._el_lo_full, self._el_hi_full = el_lo, el_hi
        self._fr_lo_full, self._fr_hi_full = fr_lo, fr_hi

        self.spin_az_min = _make_spin(-1e9, 1e9, az_lo)
        self.spin_az_max = _make_spin(-1e9, 1e9, az_hi)
        self.spin_el_min = _make_spin(-1e9, 1e9, el_lo)
        self.spin_el_max = _make_spin(-1e9, 1e9, el_hi)
        self.spin_fr_min = _make_spin(-1e9, 1e9, fr_lo)
        self.spin_fr_max = _make_spin(-1e9, 1e9, fr_hi)

        def _info(n, lo, hi):
            return QLabel(f"  {n} samples  ({lo:.6g} – {hi:.6g})")

        grid.addWidget(QLabel("Azimuth"), 0, 0)
        grid.addWidget(QLabel("Min"), 0, 1)
        grid.addWidget(self.spin_az_min, 0, 2)
        grid.addWidget(QLabel("Max"), 0, 3)
        grid.addWidget(self.spin_az_max, 0, 4)
        grid.addWidget(_info(len(dataset.azimuths), az_lo, az_hi), 0, 5)

        grid.addWidget(QLabel("Elevation"), 1, 0)
        grid.addWidget(QLabel("Min"), 1, 1)
        grid.addWidget(self.spin_el_min, 1, 2)
        grid.addWidget(QLabel("Max"), 1, 3)
        grid.addWidget(self.spin_el_max, 1, 4)
        grid.addWidget(_info(len(dataset.elevations), el_lo, el_hi), 1, 5)

        grid.addWidget(QLabel("Frequency"), 2, 0)
        grid.addWidget(QLabel("Min"), 2, 1)
        grid.addWidget(self.spin_fr_min, 2, 2)
        grid.addWidget(QLabel("Max"), 2, 3)
        grid.addWidget(self.spin_fr_max, 2, 4)
        grid.addWidget(_info(len(dataset.frequencies), fr_lo, fr_hi), 2, 5)

        layout.addLayout(grid)

        pol_group = QGroupBox("Polarization  (check = keep)")
        pol_row = QHBoxLayout(pol_group)
        self._pol_checks: list[tuple[QCheckBox, object]] = []
        for pol in dataset.polarizations:
            chk = QCheckBox(str(pol))
            chk.setChecked(True)
            chk.toggled.connect(self._update_preview)
            pol_row.addWidget(chk)
            self._pol_checks.append((chk, pol))
        pol_row.addStretch(1)
        layout.addWidget(pol_group)

        self._lbl_preview = QLabel()
        layout.addWidget(self._lbl_preview)

        btn_reset = QPushButton("Reset to Full Range")
        btn_reset.clicked.connect(self._reset)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        bottom = QHBoxLayout()
        bottom.addWidget(btn_reset)
        bottom.addStretch(1)
        bottom.addWidget(btn_box)
        layout.addLayout(bottom)

        for spin in (
            self.spin_az_min, self.spin_az_max,
            self.spin_el_min, self.spin_el_max,
            self.spin_fr_min, self.spin_fr_max,
        ):
            spin.valueChanged.connect(self._update_preview)

        # Pre-fill from parameter list selections (if any)
        self._prefill_axis(self.spin_az_min, self.spin_az_max, presel_az)
        self._prefill_axis(self.spin_el_min, self.spin_el_max, presel_el)
        self._prefill_axis(self.spin_fr_min, self.spin_fr_max, presel_freq)
        if presel_pol is not None:
            presel_strs = {str(v) for v in presel_pol}
            for chk, pol in self._pol_checks:
                chk.setChecked(str(pol) in presel_strs)

        self._update_preview()

    @staticmethod
    def _prefill_axis(spin_min, spin_max, values) -> None:
        if not values:
            return
        arr = np.asarray(values, dtype=float)
        spin_min.setValue(float(arr.min()))
        spin_max.setValue(float(arr.max()))

    def _reset(self) -> None:
        self.spin_az_min.setValue(self._az_lo_full)
        self.spin_az_max.setValue(self._az_hi_full)
        self.spin_el_min.setValue(self._el_lo_full)
        self.spin_el_max.setValue(self._el_hi_full)
        self.spin_fr_min.setValue(self._fr_lo_full)
        self.spin_fr_max.setValue(self._fr_hi_full)
        for chk, _ in self._pol_checks:
            chk.setChecked(True)

    @staticmethod
    def _count_in_range(arr, lo, hi, tol: float = 1e-6) -> int:
        a = np.asarray(arr, dtype=float)
        return int(np.sum((a >= lo - tol) & (a <= hi + tol)))

    def _update_preview(self) -> None:
        ds = self._dataset
        n_az = self._count_in_range(ds.azimuths, self.spin_az_min.value(), self.spin_az_max.value())
        n_el = self._count_in_range(ds.elevations, self.spin_el_min.value(), self.spin_el_max.value())
        n_fr = self._count_in_range(ds.frequencies, self.spin_fr_min.value(), self.spin_fr_max.value())
        n_pol = sum(1 for chk, _ in self._pol_checks if chk.isChecked())
        orig = f"{len(ds.azimuths)} × {len(ds.elevations)} × {len(ds.frequencies)} × {len(ds.polarizations)}"
        result = f"{n_az} × {n_el} × {n_fr} × {n_pol}"
        self._lbl_preview.setText(
            f"Result (reference dataset):  {orig}  →  {result}  (az × el × freq × pol)"
        )

    def get_crop_params(self) -> dict:
        """Return kwargs suitable for RcsGrid.axis_crop()."""
        ds = self._dataset
        tol = 1e-6

        def _range_or_none(lo, hi, full_lo, full_hi):
            if lo <= full_lo + tol and hi >= full_hi - tol:
                return None
            return (lo, hi)

        az_range = _range_or_none(
            self.spin_az_min.value(), self.spin_az_max.value(),
            self._az_lo_full, self._az_hi_full,
        )
        el_range = _range_or_none(
            self.spin_el_min.value(), self.spin_el_max.value(),
            self._el_lo_full, self._el_hi_full,
        )
        fr_range = _range_or_none(
            self.spin_fr_min.value(), self.spin_fr_max.value(),
            self._fr_lo_full, self._fr_hi_full,
        )

        checked_pols = {str(pol) for chk, pol in self._pol_checks if chk.isChecked()}
        all_pols = {str(pol) for _, pol in self._pol_checks}
        if checked_pols >= all_pols or not checked_pols:
            pol_values = None
        else:
            pol_values = [pol for _, pol in self._pol_checks if str(pol) in checked_pols]

        return {
            "azimuth_range": az_range,
            "elevation_range": el_range,
            "frequency_range": fr_range,
            "polarizations": pol_values,
        }


class AlignDialog(QDialog):
    """Choose alignment mode when aligning datasets to a reference."""

    def __init__(self, ref_name: str, n_others: int, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Align Datasets")
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            f"Reference: <b>{ref_name}</b>  —  aligning {n_others} other dataset(s) to it."
        ))

        grp = QGroupBox("Alignment Mode")
        grp_layout = QVBoxLayout(grp)
        self._btn_group = QButtonGroup(self)
        self._radio_intersect = QRadioButton(
            "Intersect — keep only axis values present in both datasets (exact match, no interpolation)"
        )
        self._radio_interp = QRadioButton(
            "Interpolate — linearly interpolate to the reference axes (no extrapolation)"
        )
        self._radio_intersect.setChecked(True)
        self._btn_group.addButton(self._radio_intersect, 0)
        self._btn_group.addButton(self._radio_interp, 1)
        grp_layout.addWidget(self._radio_intersect)
        grp_layout.addWidget(self._radio_interp)
        layout.addWidget(grp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_mode(self) -> str:
        return "interp" if self._radio_interp.isChecked() else "intersect"


class ScaleDialog(QDialog):
    """Scale RCS values by a linear multiplier or a dB offset."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scale Dataset")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Apply a scaling factor to all RCS values:"))

        self._btn_group = QButtonGroup(self)
        self._radio_linear = QRadioButton("Linear multiplier:")
        self._radio_db = QRadioButton("dB offset (applied as power shift, e.g. +3 dBsm ≈ ×2.0):")
        self._radio_linear.setChecked(True)
        self._btn_group.addButton(self._radio_linear, 0)
        self._btn_group.addButton(self._radio_db, 1)

        self._spin_linear = QDoubleSpinBox()
        self._spin_linear.setRange(1e-12, 1e12)
        self._spin_linear.setDecimals(6)
        self._spin_linear.setValue(1.0)

        self._spin_db = QDoubleSpinBox()
        self._spin_db.setRange(-300.0, 300.0)
        self._spin_db.setDecimals(4)
        self._spin_db.setSingleStep(1.0)
        self._spin_db.setValue(0.0)
        self._spin_db.setEnabled(False)

        row1 = QHBoxLayout()
        row1.addWidget(self._radio_linear)
        row1.addWidget(self._spin_linear)
        row2 = QHBoxLayout()
        row2.addWidget(self._radio_db)
        row2.addWidget(self._spin_db)
        layout.addLayout(row1)
        layout.addLayout(row2)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._radio_linear.toggled.connect(self._update_enabled)

    def _update_enabled(self, linear_checked: bool) -> None:
        self._spin_linear.setEnabled(linear_checked)
        self._spin_db.setEnabled(not linear_checked)

    def get_factor(self) -> complex:
        """Return the complex linear multiplier to apply to RCS."""
        if self._radio_linear.isChecked():
            return complex(self._spin_linear.value())
        return complex(10.0 ** (self._spin_db.value() / 10.0))


class ResampleDialog(QDialog):
    """Resample dataset axes to a user-specified number of evenly-spaced points."""

    def __init__(self, dataset, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resample Dataset")
        self._dataset = dataset
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Interpolate each axis to a new number of evenly-spaced points "
            "(within the existing range — no extrapolation)."
        ))

        grid = QGridLayout()
        self._spins: dict[str, QDoubleSpinBox] = {}

        axes = [
            ("Azimuth", dataset.azimuths),
            ("Elevation", dataset.elevations),
            ("Frequency", dataset.frequencies),
        ]
        for row_idx, (label, arr) in enumerate(axes):
            n = len(arr)
            lo, hi = float(arr.min()), float(arr.max())
            spin = QDoubleSpinBox()
            spin.setRange(2 if n > 1 else 1, 8192)
            spin.setDecimals(0)
            spin.setValue(float(n))
            spin.setEnabled(n > 1)
            grid.addWidget(QLabel(f"{label}:"), row_idx, 0)
            grid.addWidget(spin, row_idx, 1)
            info = f"currently {n} pts,  {lo:.6g} – {hi:.6g}"
            if n == 1:
                info += "  (single-point axis, locked)"
            grid.addWidget(QLabel(info), row_idx, 2)
            self._spins[label] = spin

        layout.addLayout(grid)
        layout.addWidget(QLabel(
            "Polarization axis is unchanged (interpolation requires identical polarizations)."
        ))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_target_counts(self) -> tuple[int, int, int]:
        return (
            int(self._spins["Azimuth"].value()),
            int(self._spins["Elevation"].value()),
            int(self._spins["Frequency"].value()),
        )


class ExportCsvDialog(QDialog):
    """Options for exporting RCS data to a CSV/TSV file."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export to CSV")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        grid.addWidget(QLabel("Magnitude:"), 0, 0)
        self._combo_scale = QComboBox()
        self._combo_scale.addItem("Linear", "linear")
        self._combo_scale.addItem("dBsm", "dbsm")
        self._combo_scale.addItem("Both", "both")
        grid.addWidget(self._combo_scale, 0, 1)

        grid.addWidget(QLabel("Separator:"), 1, 0)
        self._combo_sep = QComboBox()
        self._combo_sep.addItem("Comma  (.csv)", ",")
        self._combo_sep.addItem("Tab  (.tsv)", "\t")
        grid.addWidget(self._combo_sep, 1, 1)

        layout.addLayout(grid)

        self._chk_phase = QCheckBox("Include phase column (degrees)")
        self._chk_phase.setChecked(False)
        layout.addWidget(self._chk_phase)

        layout.addWidget(QLabel(
            "Columns: azimuth, elevation, frequency, polarization, [magnitude], [phase].\n"
            "One row per sample — all combinations of selected axes."
        ))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_options(self) -> tuple[str, str, bool]:
        """Return (scale, separator, include_phase)."""
        return (
            self._combo_scale.currentData(),
            self._combo_sep.currentData(),
            self._chk_phase.isChecked(),
        )


class StatisticsDialog(QDialog):
    """Single dialog for statistics dataset: all options in one place."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Statistics Dataset")
        layout = QVBoxLayout(self)

        params_grid = QGridLayout()

        params_grid.addWidget(QLabel("Statistic:"), 0, 0)
        self.combo_stat = QComboBox()
        self.combo_stat.addItems(["mean", "median", "min", "max", "std", "percentile"])
        params_grid.addWidget(self.combo_stat, 0, 1)

        params_grid.addWidget(QLabel("Percentile:"), 0, 2)
        self.spin_pct = QDoubleSpinBox()
        self.spin_pct.setRange(0.0, 100.0)
        self.spin_pct.setDecimals(1)
        self.spin_pct.setSingleStep(5.0)
        self.spin_pct.setValue(50.0)
        self.spin_pct.setEnabled(False)
        self.spin_pct.setToolTip("Only used when Statistic = percentile")
        params_grid.addWidget(self.spin_pct, 0, 3)

        params_grid.addWidget(QLabel("Domain:"), 1, 0)
        self.combo_domain = QComboBox()
        self.combo_domain.addItem("Magnitude (linear)", "magnitude")
        self.combo_domain.addItem("dBsm", "dbsm")
        self.combo_domain.addItem("Complex", "complex")
        params_grid.addWidget(self.combo_domain, 1, 1, 1, 3)

        layout.addLayout(params_grid)

        axes_group = QGroupBox("Axes to Reduce")
        axes_row = QHBoxLayout(axes_group)
        self.chk_az = QCheckBox("Azimuth")
        self.chk_az.setChecked(True)
        self.chk_el = QCheckBox("Elevation")
        self.chk_el.setChecked(True)
        self.chk_freq = QCheckBox("Frequency")
        self.chk_freq.setChecked(True)
        self.chk_pol = QCheckBox("Polarization")
        self.chk_pol.setChecked(False)
        for chk in (self.chk_az, self.chk_el, self.chk_freq, self.chk_pol):
            axes_row.addWidget(chk)
        axes_row.addStretch(1)
        layout.addWidget(axes_group)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self.combo_stat.currentTextChanged.connect(
            lambda t: self.spin_pct.setEnabled(t == "percentile")
        )

    def get_params(self) -> tuple[str, float, str, list[str]]:
        """Return (statistic, percentile, domain, axes)."""
        statistic = self.combo_stat.currentText()
        percentile = self.spin_pct.value()
        domain = self.combo_domain.currentData()
        axes = [
            name
            for chk, name in (
                (self.chk_az, "azimuth"),
                (self.chk_el, "elevation"),
                (self.chk_freq, "frequency"),
                (self.chk_pol, "polarization"),
            )
            if chk.isChecked()
        ]
        return statistic, percentile, domain, axes


def _resample_grid(dataset: "RcsGrid", n_az: int, n_el: int, n_freq: int) -> "RcsGrid":
    """Interpolate a dataset's numeric axes to the given sample counts."""
    from scipy.interpolate import RegularGridInterpolator

    az = np.asarray(dataset.azimuths, dtype=float)
    el = np.asarray(dataset.elevations, dtype=float)
    fr = np.asarray(dataset.frequencies, dtype=float)
    new_az = np.linspace(az[0], az[-1], n_az) if len(az) > 1 else az.copy()
    new_el = np.linspace(el[0], el[-1], n_el) if len(el) > 1 else el.copy()
    new_fr = np.linspace(fr[0], fr[-1], n_freq) if len(fr) > 1 else fr.copy()

    rcs = dataset.rcs
    n_pol = rcs.shape[3]
    new_rcs = np.empty((len(new_az), len(new_el), len(new_fr), n_pol), dtype=rcs.dtype)

    coords = np.array(np.meshgrid(new_az, new_el, new_fr, indexing="ij")).reshape(3, -1).T
    for p in range(n_pol):
        vol = rcs[:, :, :, p]
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        re = RegularGridInterpolator((az, el, fr), vol.real, **kw)(coords)
        im = RegularGridInterpolator((az, el, fr), vol.imag, **kw)(coords)
        new_rcs[:, :, :, p] = (re + 1j * im).reshape(len(new_az), len(new_el), len(new_fr))

    return RcsGrid(
        new_az,
        new_el,
        new_fr,
        dataset.polarizations,
        new_rcs,
        rcs_power=dataset.rcs_to_linear(new_rcs),
        rcs_domain=dataset.rcs_domain,
    )


def _dataset_with_rcs(
    dataset: "RcsGrid",
    rcs,
    *,
    rcs_power=None,
    rcs_domain: str | None = None,
) -> "RcsGrid":
    return RcsGrid(
        dataset.azimuths,
        dataset.elevations,
        dataset.frequencies,
        dataset.polarizations,
        rcs,
        rcs_power=rcs_power,
        rcs_domain=(dataset.rcs_domain if rcs_domain is None else rcs_domain),
        units=dataset.units,
    )


def _write_dataset_csv(
    dataset: "RcsGrid",
    path: str,
    *,
    scale: str = "linear",
    sep: str = ",",
    include_phase: bool = False,
) -> None:
    """Write a flat az×el×freq×pol CSV from a dataset."""
    az = dataset.azimuths
    el = dataset.elevations
    fr = dataset.frequencies
    pol = dataset.polarizations
    rcs = dataset.rcs
    header = ["azimuth", "elevation", "frequency", "polarization"]
    if scale in ("linear", "both"):
        header.append("magnitude_linear")
    if scale in ("dbsm", "both"):
        header.append("magnitude_dbsm")
    if include_phase:
        header.append("phase_deg")

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(sep.join(header) + "\n")
        for ai, az_v in enumerate(az):
            for ei, el_v in enumerate(el):
                for fi, fr_v in enumerate(fr):
                    for pi, pol_v in enumerate(pol):
                        val = rcs[ai, ei, fi, pi]
                        mag = float(dataset.rcs_to_linear(val))
                        row = [str(az_v), str(el_v), str(fr_v), str(pol_v)]
                        if scale in ("linear", "both"):
                            row.append(f"{mag:.10g}")
                        if scale in ("dbsm", "both"):
                            row.append(f"{float(dataset.rcs_to_dbsm(val)):.6f}")
                        if include_phase:
                            row.append(f"{np.degrees(np.angle(val)):.6f}")
                        f.write(sep.join(row) + "\n")


class CalNormDialog(QDialog):
    """Prompt for the known theoretical RCS of the calibration reference target."""

    def __init__(self, ref_name: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibration Normalization")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Reference dataset: {ref_name}"))
        layout.addWidget(QLabel(
            "Enter the known theoretical RCS of the reference target.\n"
            "Each target is divided by the reference then scaled to absolute dBsm."
        ))
        grid = QGridLayout()
        grid.addWidget(QLabel("Known reference RCS (dBsm):"), 0, 0)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-200.0, 200.0)
        self.spin.setDecimals(4)
        self.spin.setValue(0.0)
        grid.addWidget(self.spin, 0, 1)
        layout.addLayout(grid)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_ref_rcs_dbsm(self) -> float:
        return self.spin.value()


class TimeGateDialog(QDialog):
    """Parameters for time-domain gating of frequency-domain RCS data."""

    def __init__(self, dataset: RcsGrid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Time Gate")
        c = 2.998e8
        freqs = np.asarray(dataset.frequencies, dtype=float)
        units = dataset.units or {}
        freq_unit = str(units.get("frequency", "")).lower()
        if freq_unit == "hz":
            freq_hz = freqs
        elif freq_unit == "mhz":
            freq_hz = freqs * 1e6
        else:
            freq_hz = freqs * 1e9
        n = len(freq_hz)
        if n > 1:
            bw = freq_hz[-1] - freq_hz[0]
            df = bw / (n - 1)
            range_res = c / (2.0 * bw) if bw > 0 else float("inf")
            range_per_bin = c / (2.0 * n * df) if df > 0 else float("inf")
            max_range = range_per_bin * (n // 2)
        else:
            range_res = range_per_bin = max_range = float("inf")

        layout = QVBoxLayout(self)
        info = QLabel(
            f"Frequency points: {n}\n"
            f"Range resolution: {range_res:.4g} m   |   bin width: {range_per_bin:.4g} m\n"
            f"Max physical range: {max_range:.4g} m"
        )
        layout.addWidget(info)
        grid = QGridLayout()
        grid.addWidget(QLabel("Gate start (m):"), 0, 0)
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 1e6)
        self.spin_start.setDecimals(3)
        self.spin_start.setValue(0.0)
        grid.addWidget(self.spin_start, 0, 1)
        grid.addWidget(QLabel("Gate end (m):"), 1, 0)
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(0.0, 1e6)
        self.spin_end.setDecimals(3)
        default_end = min(max_range * 0.5, 20.0) if max_range < float("inf") else 20.0
        self.spin_end.setValue(default_end)
        grid.addWidget(self.spin_end, 1, 1)
        grid.addWidget(QLabel("Window:"), 2, 0)
        self.combo = QComboBox()
        self.combo.addItems(["hanning", "hamming", "blackman", "blackmanharris", "boxcar"])
        grid.addWidget(self.combo, 2, 1)
        layout.addLayout(grid)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_params(self) -> tuple[float, float, str]:
        return self.spin_start.value(), self.spin_end.value(), self.combo.currentText()


class BwAvgDialog(QDialog):
    """Select a frequency sub-band to incoherently average over."""

    def __init__(self, dataset: RcsGrid, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Frequency Bandwidth Averaging")
        freqs = np.asarray(dataset.frequencies, dtype=float)
        f_lo, f_hi, n = float(freqs.min()), float(freqs.max()), len(freqs)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Available: {n} frequency points  ({f_lo:.6g} – {f_hi:.6g})\n"
            "Computes √(mean(|RCS(f)|²)) over the sub-band → single-frequency result."
        ))
        grid = QGridLayout()
        grid.addWidget(QLabel("Frequency min:"), 0, 0)
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(-1e9, 1e9)
        self.spin_min.setDecimals(6)
        self.spin_min.setValue(f_lo)
        grid.addWidget(self.spin_min, 0, 1)
        grid.addWidget(QLabel("Frequency max:"), 1, 0)
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(-1e9, 1e9)
        self.spin_max.setDecimals(6)
        self.spin_max.setValue(f_hi)
        grid.addWidget(self.spin_max, 1, 1)
        layout.addLayout(grid)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_freq_range(self) -> tuple[float, float]:
        return self.spin_min.value(), self.spin_max.value()


def _apply_time_gate(
    dataset: RcsGrid, gate_start_m: float, gate_end_m: float, window_type: str
) -> RcsGrid:
    from scipy.signal import get_window

    c = 2.998e8
    freqs = np.asarray(dataset.frequencies, dtype=float)
    units = dataset.units or {}
    freq_unit = str(units.get("frequency", "")).lower()
    if freq_unit == "hz":
        freq_hz = freqs
    elif freq_unit == "mhz":
        freq_hz = freqs * 1e6
    else:
        freq_hz = freqs * 1e9

    n = len(freq_hz)
    if n < 2:
        raise ValueError("Need at least 2 frequency points for time gating.")
    bw = freq_hz[-1] - freq_hz[0]
    if bw <= 0:
        raise ValueError("Frequencies must be strictly increasing.")
    df = bw / (n - 1)

    # Range axis: R[k] = c·k / (2·N·Δf)
    range_bins = (c / (2.0 * n * df)) * np.arange(n)

    # Build gate mask (only physical first n//2 bins are gated; rest zeroed)
    gate_mask = np.zeros(n, dtype=float)
    phys = np.arange(n // 2)
    gate_idx = phys[(range_bins[phys] >= gate_start_m) & (range_bins[phys] <= gate_end_m)]

    if gate_idx.size == 0:
        raise ValueError(
            f"No range bins in [{gate_start_m:.3f}, {gate_end_m:.3f}] m. "
            f"Max physical range ≈ {range_bins[n // 2 - 1]:.3f} m."
        )

    if window_type == "boxcar" or gate_idx.size == 1:
        gate_mask[gate_idx] = 1.0
    else:
        gate_mask[gate_idx] = get_window(window_type, gate_idx.size)

    rcs_time = np.fft.ifft(dataset.rcs, axis=2)
    rcs_gated = np.fft.fft(
        rcs_time * gate_mask[np.newaxis, np.newaxis, :, np.newaxis], axis=2
    )
    return RcsGrid(
        dataset.azimuths, dataset.elevations, dataset.frequencies,
        dataset.polarizations, rcs_gated,
        rcs_power=np.abs(rcs_gated) ** 2,
        rcs_domain="complex_amplitude",
        units=dataset.units,
    )


def _apply_bw_avg(dataset: RcsGrid, f_min: float, f_max: float) -> RcsGrid:
    freqs = np.asarray(dataset.frequencies, dtype=float)
    mask = (freqs >= f_min - 1e-9) & (freqs <= f_max + 1e-9)
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"No frequencies in range [{f_min:.6g}, {f_max:.6g}].")
    pwr_sub = dataset.rcs_power[:, :, indices, :]          # (az, el, n_sub, pol)
    mean_power = np.mean(pwr_sub, axis=2, keepdims=True)
    rcs_avg = np.sqrt(np.maximum(mean_power, 0.0)).astype(np.complex128)  # (az, el, 1, pol)
    center_freq = np.array([float(np.mean(freqs[indices]))])
    return RcsGrid(
        dataset.azimuths, dataset.elevations, center_freq,
        dataset.polarizations, rcs_avg, rcs_power=mean_power, rcs_domain="linear_rcs", units=dataset.units,
    )


class DatasetOpsMixin:
    def _handle_files_dropped(self, paths: list[str]) -> None:
        for path in paths:
            if not path.lower().endswith(".grim"):
                continue
            dataset = RcsGrid.load(path)
            file_name = os.path.basename(path)
            name = os.path.splitext(file_name)[0]
            self._add_dataset_row(dataset, name, path, file_name=file_name)

    def _add_dataset_row(self, dataset: RcsGrid, name: str, history: str, file_name: str | None = None) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, dataset)
        file_text = file_name or ""
        file_item = QTableWidgetItem(file_text)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        history_item = QTableWidgetItem(history)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, file_item)
        self.table.setItem(row, 2, history_item)

    def _on_dataset_selection_changed(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        self._update_dataset_selection_order([idx.row() for idx in selected])
        if not selected:
            self.active_dataset = None
            self._clear_param_lists()
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        dataset = item.data(Qt.UserRole) if item else None
        if not isinstance(dataset, RcsGrid):
            self.active_dataset = None
            self._clear_param_lists()
            return
        self.active_dataset = dataset
        self._populate_params(dataset)

    def _update_dataset_selection_order(self, selected_rows: list[int]) -> None:
        selected_set = set(selected_rows)
        previous_order = getattr(self, "_dataset_selection_order", [])
        order = [row for row in previous_order if row in selected_set]
        current_row = self.table.currentRow()

        for row in selected_rows:
            if row not in order:
                order.append(row)

        # Use the active row as the most-recent selection.
        if current_row in selected_set and current_row in order:
            order = [row for row in order if row != current_row] + [current_row]

        self._dataset_selection_order = order

    def _populate_params(self, dataset: RcsGrid) -> None:
        self._fill_list(self.list_pol, dataset.polarizations)
        self._fill_list(self.list_freq, dataset.frequencies)
        self._fill_list(self.list_elev, dataset.elevations)
        self._fill_list(self.list_az, dataset.azimuths)

    def _fill_list(self, widget: QListWidget, values, indices=None) -> None:
        widget.blockSignals(True)
        widget.clear()
        if indices is None:
            indices = range(len(values))
        for idx in indices:
            value = values[idx]
            item = QListWidgetItem(str(value))
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            item.setData(Qt.UserRole, value)
            item.setData(Qt.UserRole + 1, int(idx))
            widget.addItem(item)
        widget.blockSignals(False)

    def _clear_param_lists(self) -> None:
        for widget in (self.list_pol, self.list_freq, self.list_elev, self.list_az):
            widget.clear()

    def _on_param_item_changed(self, item: QListWidgetItem, axis_name: str, widget: QListWidget) -> None:
        if self.active_dataset is None:
            return
        axis_arr = self.active_dataset.get_axis(axis_name)
        idx = item.data(Qt.UserRole + 1)
        if idx is None:
            return
        if idx < 0 or idx >= len(axis_arr):
            return
        old_value = axis_arr[idx]
        new_text = item.text()
        if axis_name == "polarization":
            new_value = new_text
        else:
            try:
                new_value = float(new_text)
            except ValueError:
                widget.blockSignals(True)
                item.setText(str(old_value))
                widget.blockSignals(False)
                return
        axis_arr[idx] = new_value
        item.setData(Qt.UserRole, new_value)

    def _selected_indices(self, widget: QListWidget) -> set[int]:
        indices = set()
        for item in widget.selectedItems():
            idx = item.data(Qt.UserRole + 1)
            if idx is not None:
                indices.add(int(idx))
        return indices

    def _selected_values(self, widget: QListWidget) -> list:
        values = []
        for item in widget.selectedItems():
            values.append(item.data(Qt.UserRole))
        return values

    def _indices_for_values(self, axis_arr, values, tol=1e-6) -> list[int] | None:
        axis_arr = np.asarray(axis_arr)
        indices: list[int] = []
        is_numeric_axis = np.issubdtype(axis_arr.dtype, np.number)
        for value in values:
            if is_numeric_axis and isinstance(value, (int, float, np.floating, np.integer)):
                matches = np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
            else:
                matches = np.where(axis_arr == value)[0]
            if matches.size == 0:
                return None
            indices.append(int(matches[0]))
        return indices

    def _selected_datasets(self) -> list[tuple[str, RcsGrid]]:
        datasets: list[tuple[str, RcsGrid]] = []
        selected = self.table.selectionModel().selectedRows()
        for model_index in selected:
            row = model_index.row()
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if isinstance(dataset, RcsGrid):
                datasets.append((item.text(), dataset))
        if not datasets and isinstance(self.active_dataset, RcsGrid):
            datasets.append(("Dataset", self.active_dataset))
        return datasets

    def _selected_datasets_ordered(
        self,
        *,
        use_selection_order: bool = False,
        empty_message: str = "Select two or more datasets to combine.",
    ) -> list[tuple[str, RcsGrid]] | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage(empty_message)
            return None

        selected_rows = [idx.row() for idx in selected]
        if use_selection_order:
            ordered_rows = [
                row for row in getattr(self, "_dataset_selection_order", []) if row in selected_rows
            ]
            for row in selected_rows:
                if row not in ordered_rows:
                    ordered_rows.append(row)
            selected_rows = ordered_rows
        else:
            selected_rows = sorted(selected_rows)

        datasets: list[tuple[str, RcsGrid]] = []
        for row in selected_rows:
            item = self.table.item(row, 0)
            if item is None:
                return None
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return None
            datasets.append((item.text(), dataset))
        return datasets

    def _combine_datasets_add(
        self,
        op_label: str,
        op_symbol: str,
        func_add: str,
        func_add_many: str,
    ) -> None:
        datasets = self._selected_datasets_ordered()
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        names = [name for name, _ in datasets]
        base = datasets[0][1]
        try:
            if len(datasets) == 2:
                result = getattr(base, func_add)(datasets[1][1])
            else:
                others = [ds for _, ds in datasets[1:]]
                result = getattr(base, func_add_many)(*others)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"
        self._add_dataset_row(result, new_name, history, file_name="")
        self.status.showMessage(f"{op_label} created: {new_name}")

    def _combine_datasets_sub(self, op_label: str, op_symbol: str, func_sub: str) -> None:
        datasets = self._selected_datasets_ordered()
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to combine.")
            return
        names = [name for name, _ in datasets]
        result = datasets[0][1]
        try:
            for _, ds in datasets[1:]:
                result = getattr(result, func_sub)(ds)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f" {op_symbol} ".join(names)
        history = f"{op_label}: {new_name}"
        self._add_dataset_row(result, new_name, history, file_name="")
        self.status.showMessage(f"{op_label} created: {new_name}")

    def _coherent_add_selected(self) -> None:
        self._combine_datasets_add("Coherent +", "+", "coherent_add", "coherent_add_many")

    def _coherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Coherent -", "-", "coherent_subtract")

    def _incoherent_add_selected(self) -> None:
        self._combine_datasets_add("Incoherent +", "+", "incoherent_add", "incoherent_add_many")

    def _incoherent_sub_selected(self) -> None:
        self._combine_datasets_sub("Incoherent -", "-", "incoherent_subtract")

    def _join_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to join.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to join.")
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        try:
            merged = RcsGrid.join_many(*grids, tol=1e-6)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = " | ".join(names)
        history = f"Join (last selected wins overlap): {new_name}"
        self._add_dataset_row(merged, f"Join[{new_name}]", history, file_name="")
        self.status.showMessage(f"Join created. Overlap winner: {names[-1]}.")

    def _overlap_selected_datasets(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets for overlap.",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets for overlap.")
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        try:
            overlap_grids = RcsGrid.overlap_many(*grids, tol=1e-6)
            produced = 0
            for (name, _), overlap_grid in zip(datasets, overlap_grids):
                history = f"Overlap with [{', '.join(names)}]: {name}"
                self._add_dataset_row(overlap_grid, f"{name} [Overlap]", history, file_name="")
                produced += 1
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        if produced == 0:
            self.status.showMessage("No overlap outputs were created.")
            return
        self.status.showMessage(f"Overlap created {produced} dataset(s).")

    def _prompt_choice(self, title: str, label: str, choices: list[str], default_idx: int = 0) -> str | None:
        value, ok = QInputDialog.getItem(self, title, label, choices, default_idx, False)
        if not ok:
            return None
        return str(value)

    def _axis_crop_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to crop.",
        )
        if datasets is None:
            return

        ref = self.active_dataset if self.active_dataset is not None else datasets[0][1]
        dlg = AxisCropDialog(
            ref,
            n_datasets=len(datasets),
            presel_az=self._selected_values(self.list_az) or None,
            presel_el=self._selected_values(self.list_elev) or None,
            presel_freq=self._selected_values(self.list_freq) or None,
            presel_pol=self._selected_values(self.list_pol) or None,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        crop_params = dlg.get_crop_params()
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                cropped = dataset.axis_crop(**crop_params)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            self._add_dataset_row(cropped, f"{name} [Crop]", f"Axis Crop: {name}", file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Axis Crop created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Axis Crop created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Axis Crop created {produced} dataset(s).")

    def _statistics_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets for statistics.",
        )
        if datasets is None:
            return

        dlg = StatisticsDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        statistic, percentile, domain, axes = dlg.get_params()
        if not axes:
            self.status.showMessage("Select at least one axis for statistics reduction.")
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                stat_grid = dataset.statistics_dataset(
                    statistic=statistic,
                    axes=axes,
                    domain=domain,
                    percentile=percentile,
                    broadcast_reduced=True,
                )
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue

            if statistic == "percentile":
                stat_label = f"p{percentile:g}"
            else:
                stat_label = statistic
            history = f"Statistics ({stat_label}, {domain}, axes={axes}): {name}"
            self._add_dataset_row(stat_grid, f"{name} [{stat_label}]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Statistics created 0 datasets.")
            return
        if skipped:
            self.status.showMessage(
                f"Statistics created {produced} dataset(s). Skipped: {', '.join(skipped)}"
            )
            return
        self.status.showMessage(f"Statistics created {produced} dataset(s).")

    def _difference_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets for difference.",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("Select exactly 2 datasets for difference.")
            return

        mode = self._prompt_choice(
            "Difference",
            "Mode:",
            ["coherent", "incoherent", "db"],
            default_idx=0,
        )
        if mode is None:
            return

        names = [name for name, _ in datasets]
        grids = [grid for _, grid in datasets]
        try:
            result = grids[0].difference(grids[1], mode=mode)
        except (ValueError, TypeError) as exc:
            self.status.showMessage(str(exc))
            return

        new_name = f"Diff[{mode}] " + " - ".join(names)
        history = f"Difference ({mode}): {' - '.join(names)}"
        self._add_dataset_row(result, new_name, history, file_name="")
        self.status.showMessage(f"Difference created: {new_name}")

    def _delete_selected_datasets(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to delete.")
            return
        rows = sorted((idx.row() for idx in selected), reverse=True)
        for row in rows:
            self.table.removeRow(row)
        self.active_dataset = None
        self._clear_param_lists()
        self.status.showMessage(f"Deleted {len(rows)} dataset(s).")

    def _save_selected_datasets(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select one or more datasets to save.")
            return

        rows = sorted(idx.row() for idx in selected)

        if len(rows) == 1:
            # Single dataset — let the user pick the exact file path.
            row = rows[0]
            item = self.table.item(row, 0)
            if item is None:
                return
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                return
            name = item.text().strip() or "dataset"
            file_item = self.table.item(row, 1)
            prev_file = file_item.text() if file_item else ""
            prev_stem = os.path.splitext(prev_file)[0] if prev_file else ""
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Dataset", f"{name}.grim", "GRIM Files (*.grim)"
            )
            if not path:
                return
            saved_path = dataset.save(path)
            file_name = os.path.basename(saved_path)
            if file_item is None:
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            else:
                file_item.setText(file_name)
            history_item = self.table.item(row, 2)
            if history_item is None:
                history_item = QTableWidgetItem(saved_path)
                self.table.setItem(row, 2, history_item)
            else:
                history_item.setText(saved_path)
            new_stem = os.path.splitext(file_name)[0]
            if prev_stem and item.text().strip() == prev_stem:
                item.setText(new_stem)
            elif not item.text().strip():
                item.setText(new_stem)
            self.status.showMessage("Save completed.")
        else:
            # Multiple datasets — pick a folder once, save each using its table name.
            directory = QFileDialog.getExistingDirectory(self, "Save Selected Datasets")
            if not directory:
                return
            saved = 0
            for row in rows:
                item = self.table.item(row, 0)
                if item is None:
                    continue
                dataset = item.data(Qt.UserRole)
                if not isinstance(dataset, RcsGrid):
                    continue
                name = item.text().strip() or f"dataset_{row + 1}"
                path = os.path.join(directory, f"{name}.grim")
                saved_path = dataset.save(path)
                file_name = os.path.basename(saved_path)
                file_item = self.table.item(row, 1)
                if file_item is None:
                    file_item = QTableWidgetItem(file_name)
                    file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, 1, file_item)
                else:
                    file_item.setText(file_name)
                history_item = self.table.item(row, 2)
                if history_item is None:
                    history_item = QTableWidgetItem(saved_path)
                    self.table.setItem(row, 2, history_item)
                else:
                    history_item.setText(saved_path)
                saved += 1
            self.status.showMessage(f"Saved {saved} dataset(s) to {directory}.")

    def _save_all_datasets(self) -> None:
        if self.table.rowCount() == 0:
            self.status.showMessage("No datasets to save.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Save All Datasets")
        if not directory:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            dataset = item.data(Qt.UserRole)
            if not isinstance(dataset, RcsGrid):
                continue
            name = item.text().strip() or f"dataset_{row + 1}"
            filename = f"{name}.grim"
            path = os.path.join(directory, filename)
            saved_path = dataset.save(path)
            file_name = os.path.basename(saved_path)
            file_item = self.table.item(row, 1)
            if file_item is None:
                file_item = QTableWidgetItem(file_name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, file_item)
            else:
                file_item.setText(file_name)
            history_item = self.table.item(row, 2)
            if history_item is None:
                history_item = QTableWidgetItem(saved_path)
                self.table.setItem(row, 2, history_item)
            else:
                history_item.setText(saved_path)
        self.status.showMessage("Save all completed.")

    def _export_plot(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf)",
        )
        if not path:
            return
        root, ext = os.path.splitext(path)
        if not ext:
            if "PDF" in selected_filter:
                path = f"{path}.pdf"
            else:
                path = f"{path}.png"
        self.plot_figure.savefig(path, dpi=200, bbox_inches="tight")
        self.status.showMessage(f"Plot exported: {os.path.basename(path)}")

    def _on_plot_context_menu(self, pos) -> None:
        menu = QMenu(self)
        action_copy = menu.addAction("Copy Plot")
        pbp_menu = menu.addMenu("PBP Fill Mode")
        action_pbp_gray = pbp_menu.addAction("Gray")
        action_pbp_gray.setCheckable(True)
        action_pbp_gray.setChecked(self.pbp_fill_mode == "gray")
        action_pbp_rcs = pbp_menu.addAction("Heatmap (RCS Value)")
        action_pbp_rcs.setCheckable(True)
        action_pbp_rcs.setChecked(self.pbp_fill_mode == "heatmap_rcs")
        action_pbp_density = pbp_menu.addAction("Heatmap (Overlap Density)")
        action_pbp_density.setCheckable(True)
        action_pbp_density.setChecked(self.pbp_fill_mode == "heatmap_density")
        action = menu.exec(self.plot_canvas.mapToGlobal(pos))
        if action == action_copy:
            pixmap = self.plot_canvas.grab()
            QApplication.clipboard().setPixmap(pixmap)
            self.status.showMessage("Plot copied to clipboard.")
        elif action in (action_pbp_gray, action_pbp_rcs, action_pbp_density):
            if action == action_pbp_gray:
                self.pbp_fill_mode = "gray"
            elif action == action_pbp_rcs:
                self.pbp_fill_mode = "heatmap_rcs"
            else:
                self.pbp_fill_mode = "heatmap_density"
            if self.last_plot_mode == "azimuth_rect":
                self._plot_azimuth_rect()
            elif self.last_plot_mode == "azimuth_polar":
                self._plot_azimuth_polar()
            elif self.last_plot_mode == "frequency":
                self._plot_frequency()
            elif self.last_plot_mode == "isar_image":
                self._plot_isar_image()

    def _on_dataset_header_double_clicked(self, section: int) -> None:
        if section != 0:
            return
        self.table.selectAll()

    def _on_dataset_context_menu(self, pos) -> None:
        if not self.table.selectionModel().selectedRows():
            index = self.table.indexAt(pos)
            if index.isValid():
                self.table.selectRow(index.row())
            else:
                return
        menu = QMenu(self)
        action_save = menu.addAction("Save")
        action_delete = menu.addAction("Delete")
        menu.addSeparator()
        action_color = menu.addAction("Text Color…")
        action_reset_color = menu.addAction("Reset Text Color")
        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == action_save:
            self._save_selected_datasets()
        elif action == action_delete:
            self._delete_selected_datasets()
        elif action == action_color:
            self._set_dataset_text_color()
        elif action == action_reset_color:
            self._reset_dataset_text_color()

    def _set_dataset_text_color(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if not rows:
            return
        initial = self.table.item(rows[0], 0)
        initial_color = initial.foreground().color() if initial else QColor()
        color = QColorDialog.getColor(initial_color, self, "Choose Text Color")
        if not color.isValid():
            return
        brush = QBrush(color)
        for row in rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setForeground(brush)

    def _reset_dataset_text_color(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        for row in rows:
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is not None:
                    item.setForeground(QBrush())

    def _align_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select two or more datasets to align (first = reference).",
        )
        if datasets is None:
            return
        if len(datasets) < 2:
            self.status.showMessage("Select at least 2 datasets to align (first = reference).")
            return

        ref_name, ref_grid = datasets[0]
        others = datasets[1:]
        dlg = AlignDialog(ref_name, len(others), parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        mode = dlg.get_mode()
        produced = 0
        skipped: list[str] = []
        for name, dataset in others:
            try:
                aligned = dataset.align_to(ref_grid, mode=mode)
            except (ValueError, TypeError) as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Align ({mode}) to {ref_name}: {name}"
            self._add_dataset_row(aligned, f"{name} [Aligned]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Align created 0 datasets.")
            return
        msg = f"Align created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _scale_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to scale.",
        )
        if datasets is None:
            return

        dlg = ScaleDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        factor = dlg.get_factor()
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                power_scale = float(factor)
                if dataset.rcs_domain == "complex_amplitude":
                    amp_scale = np.sqrt(max(power_scale, 0.0))
                    result_rcs = dataset.rcs * amp_scale
                else:
                    result_rcs = dataset.rcs * power_scale
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power * power_scale,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Scale (×{factor:.6g}): {name}"
            self._add_dataset_row(result, f"{name} [Scaled]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Scale created 0 datasets.")
            return
        msg = f"Scale created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _offset_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to offset.",
        )
        if datasets is None:
            return

        value, ok = QInputDialog.getDouble(
            self, "Offset", "Offset (dB) — shifts all displayed values by this amount:",
            0.0, -300.0, 300.0, 4,
        )
        if not ok:
            return

        linear_scale = 10.0 ** (value / 10.0)
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                if dataset.rcs_domain == "complex_amplitude":
                    result_rcs = dataset.rcs * np.sqrt(linear_scale)
                else:
                    result_rcs = dataset.rcs * linear_scale
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power * linear_scale,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Offset ({value:+.6g}): {name}"
            self._add_dataset_row(result, f"{name} [Offset {value:+.6g}]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Offset created 0 datasets.")
            return
        msg = f"Offset created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _normalize_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to normalize.",
        )
        if datasets is None:
            return

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                peak = float(np.nanmax(dataset.rcs_power))
                if peak == 0.0:
                    skipped.append(f"{name} (all-zero RCS)")
                    continue
                if dataset.rcs_domain == "complex_amplitude":
                    result_rcs = dataset.rcs / np.sqrt(peak)
                else:
                    result_rcs = dataset.rcs / peak
                result = _dataset_with_rcs(
                    dataset,
                    result_rcs,
                    rcs_power=dataset.rcs_power / peak,
                    rcs_domain=dataset.rcs_domain,
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Normalize (peak={peak:.6g}): {name}"
            self._add_dataset_row(result, f"{name} [Norm]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Normalize created 0 datasets.")
            return
        msg = f"Normalize created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _phase_shift_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to phase-shift.",
        )
        if datasets is None:
            return

        phase_deg, ok = QInputDialog.getDouble(
            self, "Phase Shift", "Phase shift (degrees):", 0.0, -360.0, 360.0, 4
        )
        if not ok:
            return

        phasor = np.exp(1j * np.deg2rad(phase_deg))
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _dataset_with_rcs(
                    dataset,
                    dataset.rcs * phasor,
                    rcs_power=dataset.rcs_power,
                    rcs_domain="complex_amplitude",
                )
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Phase Shift ({phase_deg:+.4g} deg): {name}"
            self._add_dataset_row(result, f"{name} [Phase {phase_deg:+.4g}°]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Phase Shift created 0 datasets.")
            return
        msg = f"Phase Shift created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _resample_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to resample.",
        )
        if datasets is None:
            return

        ref = self.active_dataset if self.active_dataset is not None else datasets[0][1]
        dlg = ResampleDialog(ref, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        n_az, n_el, n_freq = dlg.get_target_counts()
        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _resample_grid(dataset, n_az, n_el, n_freq)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            history = f"Resample ({n_az}×{n_el}×{n_freq}): {name}"
            self._add_dataset_row(result, f"{name} [Resampled]", history, file_name="")
            produced += 1

        if produced == 0:
            self.status.showMessage("Resample created 0 datasets.")
            return
        msg = f"Resample created {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _rename_selected(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.status.showMessage("Select a dataset to rename.")
            return
        if len(selected) > 1:
            self.status.showMessage("Select exactly one dataset to rename.")
            return
        row = selected[0].row()
        item = self.table.item(row, 0)
        if item is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Dataset", "New name:", text=item.text()
        )
        if not ok or not new_name.strip():
            return
        item.setText(new_name.strip())
        self.status.showMessage(f"Renamed to: {new_name.strip()}")

    def _duplicate_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to duplicate.",
        )
        if datasets is None:
            return

        for name, dataset in datasets:
            dup = RcsGrid(
                dataset.azimuths.copy(),
                dataset.elevations.copy(),
                dataset.frequencies.copy(),
                list(dataset.polarizations),
                dataset.rcs.copy(),
                rcs_power=dataset.rcs_power.copy(),
                rcs_domain=dataset.rcs_domain,
            )
            self._add_dataset_row(dup, f"{name} [Copy]", f"Duplicate of: {name}", file_name="")
        self.status.showMessage(f"Duplicated {len(datasets)} dataset(s).")

    def _export_csv_selected(self) -> None:
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to export.",
        )
        if datasets is None:
            return

        dlg = ExportCsvDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        scale, sep, include_phase = dlg.get_options()
        ext = ".tsv" if sep == "\t" else ".csv"
        produced = 0
        for name, dataset in datasets:
            safe_name = name.replace("/", "_").replace("\\", "_")
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Export {name}",
                f"{safe_name}{ext}",
                "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)",
            )
            if not path:
                continue
            _write_dataset_csv(dataset, path, scale=scale, sep=sep, include_phase=include_phase)
            produced += 1

        if produced:
            self.status.showMessage(f"Exported {produced} dataset(s) to CSV.")
        else:
            self.status.showMessage("Export cancelled.")

    def _reselect_indices(self, widget: QListWidget, indices: set[int]) -> None:
        if not indices:
            return
        widget.blockSignals(True)
        for row in range(widget.count()):
            item = widget.item(row)
            idx = item.data(Qt.UserRole + 1)
            if idx in indices:
                item.setSelected(True)
        widget.blockSignals(False)

    # ── RCS-specific processing ───────────────────────────────────────────────

    def _coherent_div_selected(self) -> None:
        """Divide numerator dataset by denominator (complex, element-wise)."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select exactly 2 datasets (numerator first, then denominator).",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("Coherent ÷: select exactly 2 datasets.")
            return
        name_a, ds_a = datasets[0]
        name_b, ds_b = datasets[1]

        if ds_a.rcs.shape != ds_b.rcs.shape:
            self.status.showMessage(
                f"Coherent ÷: shape mismatch {ds_a.rcs.shape} vs {ds_b.rcs.shape}."
            )
            return

        denom = ds_b.rcs.copy()
        denom[denom == 0] = 1e-30 + 0j
        result_rcs = ds_a.rcs / denom
        result = RcsGrid(
            ds_a.azimuths, ds_a.elevations, ds_a.frequencies,
            ds_a.polarizations, result_rcs,
            rcs_power=np.abs(result_rcs) ** 2,
            rcs_domain="complex_amplitude",
            units=ds_a.units,
        )
        out_name = f"{name_a} ÷ {name_b}"
        self._add_dataset_row(result, out_name, f"Coherent ÷: {name_a} / {name_b}", file_name="")
        self.status.showMessage(f"Coherent ÷ produced: {out_name}")

    def _bg_subtract_selected(self) -> None:
        """Subtract background (second dataset) from target (first dataset)."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select exactly 2 datasets (target first, background second).",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("BG Subtract: select exactly 2 datasets.")
            return
        name_tgt, ds_tgt = datasets[0]
        name_bg, ds_bg = datasets[1]

        if ds_tgt.rcs.shape != ds_bg.rcs.shape:
            self.status.showMessage(
                f"BG Subtract: shape mismatch {ds_tgt.rcs.shape} vs {ds_bg.rcs.shape}."
            )
            return

        result_rcs = ds_tgt.rcs - ds_bg.rcs
        result = RcsGrid(
            ds_tgt.azimuths, ds_tgt.elevations, ds_tgt.frequencies,
            ds_tgt.polarizations, result_rcs,
            rcs_power=np.abs(result_rcs) ** 2,
            rcs_domain="complex_amplitude",
            units=ds_tgt.units,
        )
        out_name = f"{name_tgt} - BG"
        self._add_dataset_row(result, out_name, f"BG Subtract: {name_tgt} - {name_bg}", file_name="")
        self.status.showMessage(f"BG Subtract produced: {out_name}")

    def _cal_norm_selected(self) -> None:
        """Calibration normalization: (target / reference) * 10^(ref_rcs_dbsm/10)."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select exactly 2 datasets (target first, calibration reference second).",
        )
        if datasets is None:
            return
        if len(datasets) != 2:
            self.status.showMessage("Cal Norm: select exactly 2 datasets.")
            return
        name_tgt, ds_tgt = datasets[0]
        name_ref, ds_ref = datasets[1]

        if ds_tgt.rcs.shape != ds_ref.rcs.shape:
            self.status.showMessage(
                f"Cal Norm: shape mismatch {ds_tgt.rcs.shape} vs {ds_ref.rcs.shape}."
            )
            return

        dlg = CalNormDialog(name_ref, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        ref_rcs_dbsm = dlg.get_ref_rcs_dbsm()

        ref_linear = 10.0 ** (ref_rcs_dbsm / 10.0)
        denom = ds_ref.rcs.copy()
        denom[denom == 0] = 1e-30 + 0j
        result_rcs = (ds_tgt.rcs / denom) * np.sqrt(ref_linear)
        result = RcsGrid(
            ds_tgt.azimuths, ds_tgt.elevations, ds_tgt.frequencies,
            ds_tgt.polarizations, result_rcs,
            rcs_power=np.abs(result_rcs) ** 2,
            rcs_domain="complex_amplitude",
            units=ds_tgt.units,
        )
        out_name = f"{name_tgt} [CalNorm {ref_rcs_dbsm:+.2f}dBsm]"
        self._add_dataset_row(result, out_name, f"Cal Norm: {name_tgt} / {name_ref} @ {ref_rcs_dbsm:+.2f}dBsm", file_name="")
        self.status.showMessage(f"Cal Norm produced: {out_name}")

    def _time_gate_selected(self) -> None:
        """Apply time-domain gating via IFFT → window → FFT."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets to time gate.",
        )
        if datasets is None:
            return

        # Use first selected dataset to configure the dialog
        _, ds_first = datasets[0]
        dlg = TimeGateDialog(ds_first, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        gate_start, gate_end, window_type = dlg.get_params()

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _apply_time_gate(dataset, gate_start, gate_end, window_type)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            out_name = f"{name} [Gate {gate_start:.1f}-{gate_end:.1f}m {window_type}]"
            self._add_dataset_row(result, out_name,
                f"Time Gate: {name}  {gate_start:.3f}–{gate_end:.3f}m  {window_type}", file_name="")
            produced += 1

        msg = f"Time Gate produced {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)

    def _bw_avg_selected(self) -> None:
        """Incoherent frequency bandwidth averaging: √(mean(|RCS(f)|²))."""
        datasets = self._selected_datasets_ordered(
            use_selection_order=True,
            empty_message="Select one or more datasets for BW averaging.",
        )
        if datasets is None:
            return

        _, ds_first = datasets[0]
        dlg = BwAvgDialog(ds_first, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        f_min, f_max = dlg.get_freq_range()

        produced = 0
        skipped: list[str] = []
        for name, dataset in datasets:
            try:
                result = _apply_bw_avg(dataset, f_min, f_max)
            except Exception as exc:
                skipped.append(f"{name} ({exc})")
                continue
            out_name = f"{name} [BwAvg {f_min:.6g}–{f_max:.6g}]"
            self._add_dataset_row(result, out_name,
                f"BW Avg: {name}  {f_min:.6g}–{f_max:.6g}", file_name="")
            produced += 1

        msg = f"BW Avg produced {produced} dataset(s)."
        if skipped:
            msg += f" Skipped: {', '.join(skipped)}"
        self.status.showMessage(msg)
