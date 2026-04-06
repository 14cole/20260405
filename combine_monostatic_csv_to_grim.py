#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# USER SETTINGS (hardcoded mode, no CLI)
# Edit these values, then run:
#   python3 combine_monostatic_csv_to_grim.py
# -----------------------------------------------------------------------------
INPUT_DIRECTORY = Path("/Users/emery/Documents/Plot_Cut")
OUTPUT_GRIM_FILE = Path("/Users/emery/Documents/Plot_Cut/combined_monostatic.grim")
TARGET_FILENAME = "monostatic.csv"
OUTPUT_FREQUENCY_UNIT = "ghz"  # "ghz" or "hz"


def _canon(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _find_basic_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    canon_to_original = {_canon(name): name for name in fieldnames}
    for candidate in candidates:
        col = canon_to_original.get(_canon(candidate))
        if col is not None:
            return col
    return None


def _find_pair_column(
    fieldnames: list[str],
    prefix: str,
    first: str,
    second: str,
    suffix_hint: str,
) -> str | None:
    prefix = _canon(prefix)
    first = _canon(first)
    second = _canon(second)
    suffix_hint = _canon(suffix_hint)

    best_name = None
    best_score = None
    for original in fieldnames:
        key = _canon(original)
        if prefix not in key:
            continue
        i1 = key.find(first)
        i2 = key.find(second)
        if i1 < 0 or i2 < 0:
            continue
        if first != second and i1 >= i2:
            continue
        score = 0
        if suffix_hint and suffix_hint in key:
            score += 10
        score -= len(key)
        if best_score is None or score > best_score:
            best_score = score
            best_name = original
    return best_name


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    return val


def _dbsm_to_linear(dbsm: float) -> float:
    return 10.0 ** (dbsm / 10.0)


@dataclass(frozen=True)
class ChannelSpec:
    pol_label: str
    first: str
    second: str


CHANNELS = [
    ChannelSpec("THETA_THETA", "theta", "theta"),
    ChannelSpec("PHI_THETA", "phi", "theta"),
    ChannelSpec("THETA_PHI", "theta", "phi"),
    ChannelSpec("PHI_PHI", "phi", "phi"),
]


def _discover_monostatic_csv_files(root: Path, target_name: str) -> list[Path]:
    target = target_name.lower()
    matches: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower() == target:
                matches.append(Path(dirpath) / filename)
    matches.sort()
    return matches


def _ensure_grim_ext(path: Path) -> Path:
    if path.suffix.lower() == ".grim":
        return path
    return path.with_suffix(".grim")


def _convert_frequency(freq_hz: float, output_unit: str) -> float:
    if output_unit == "ghz":
        return freq_hz * 1e-9
    return freq_hz


def _load_samples(
    csv_paths: list[Path],
    output_freq_unit: str,
) -> tuple[list[tuple[float, float, float, str, complex]], dict[str, int]]:
    samples: list[tuple[float, float, float, str, complex]] = []
    stats = {
        "rows_total": 0,
        "rows_skipped_base": 0,
        "channels_missing_phase": 0,
        "channels_skipped": 0,
        "channels_loaded": 0,
    }

    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{csv_path}: missing header row.")
            fieldnames = list(reader.fieldnames)

            freq_col = _find_basic_column(
                fieldnames,
                [
                    "frequency(hz)",
                    "frequency_hz",
                    "frequency hz",
                    "freq(hz)",
                    "freq_hz",
                ],
            )
            theta_col = _find_basic_column(
                fieldnames,
                ["theta(deg)", "theta_deg", "theta"],
            )
            phi_col = _find_basic_column(
                fieldnames,
                ["phi(deg)", "phi_deg", "phi"],
            )

            if freq_col is None or theta_col is None or phi_col is None:
                raise ValueError(
                    f"{csv_path}: required base columns not found. "
                    f"Need frequency(hz), theta(deg), phi(deg)."
                )

            channel_cols: dict[str, tuple[str, str]] = {}
            for spec in CHANNELS:
                rcs_col = _find_pair_column(
                    fieldnames,
                    prefix="rcs",
                    first=spec.first,
                    second=spec.second,
                    suffix_hint="dbsm",
                )
                phase_col = _find_pair_column(
                    fieldnames,
                    prefix="phase",
                    first=spec.first,
                    second=spec.second,
                    suffix_hint="deg",
                )
                if rcs_col is not None:
                    channel_cols[spec.pol_label] = (rcs_col, phase_col or "")

            if not channel_cols:
                raise ValueError(
                    f"{csv_path}: no polarization channels found "
                    f"(expected RCS/phase theta-phi combinations)."
                )

            for row in reader:
                stats["rows_total"] += 1
                freq_hz = _parse_float(row.get(freq_col))
                theta_deg = _parse_float(row.get(theta_col))
                phi_deg = _parse_float(row.get(phi_col))
                if freq_hz is None or theta_deg is None or phi_deg is None:
                    stats["rows_skipped_base"] += 1
                    continue

                freq_out = _convert_frequency(freq_hz, output_freq_unit)

                for pol_label, (rcs_col, phase_col) in channel_cols.items():
                    rcs_dbsm = _parse_float(row.get(rcs_col))
                    if rcs_dbsm is None:
                        stats["channels_skipped"] += 1
                        continue
                    phase_deg = _parse_float(row.get(phase_col)) if phase_col else None
                    if phase_deg is None:
                        phase_deg = 0.0
                        stats["channels_missing_phase"] += 1

                    lin = _dbsm_to_linear(rcs_dbsm)
                    amp_mag = math.sqrt(max(lin, 0.0))
                    amp = amp_mag * complex(
                        math.cos(math.radians(phase_deg)),
                        math.sin(math.radians(phase_deg)),
                    )
                    samples.append((theta_deg, phi_deg, freq_out, pol_label, amp))
                    stats["channels_loaded"] += 1

    return samples, stats


def _build_grid(
    samples: list[tuple[float, float, float, str, complex]],
    source_root: Path,
    csv_paths: list[Path],
    frequency_unit: str,
) -> dict[str, object]:
    if not samples:
        raise ValueError("No valid samples were loaded from CSV files.")

    azimuths = np.asarray(sorted({s[0] for s in samples}), dtype=float)
    elevations = np.asarray(sorted({s[1] for s in samples}), dtype=float)
    frequencies = np.asarray(sorted({s[2] for s in samples}), dtype=float)
    polarization_order = [c.pol_label for c in CHANNELS]
    present_pols = [p for p in polarization_order if any(s[3] == p for s in samples)]
    polarizations = np.asarray(present_pols, dtype=str)

    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    sum_rcs = np.zeros(shape, dtype=np.complex128)
    hit_count = np.zeros(shape, dtype=np.int32)

    az_idx = {v: i for i, v in enumerate(azimuths)}
    el_idx = {v: i for i, v in enumerate(elevations)}
    fq_idx = {v: i for i, v in enumerate(frequencies)}
    pol_idx = {v: i for i, v in enumerate(polarizations)}

    for theta, phi, freq, pol, amp in samples:
        idx = (az_idx[theta], el_idx[phi], fq_idx[freq], pol_idx[pol])
        sum_rcs[idx] += amp
        hit_count[idx] += 1

    rcs = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    populated = hit_count > 0
    rcs[populated] = sum_rcs[populated]
    rcs_power = np.full(shape, np.nan, dtype=float)
    rcs_power[populated] = np.abs(rcs[populated]) ** 2

    units = {
        "azimuth": "deg",
        "elevation": "deg",
        "frequency": "GHz" if frequency_unit == "ghz" else "Hz",
    }

    history = (
        f"Combined {len(csv_paths)} monostatic.csv files from {source_root}. "
        "Input channels converted from dBsm+deg to complex amplitude and coherently "
        "summed when axis bins overlapped."
    )

    return {
        "azimuths": azimuths,
        "elevations": elevations,
        "frequencies": frequencies,
        "polarizations": polarizations,
        "rcs": rcs,
        "rcs_amp": rcs,
        "rcs_real": np.real(rcs).astype(float),
        "rcs_imag": np.imag(rcs).astype(float),
        "rcs_power": rcs_power,
        "rcs_domain": "complex_amplitude",
        "power_domain": "linear_rcs",
        "source_path": str(source_root),
        "history": history,
        "units": json.dumps(units),
    }


def _save_grim(payload: dict[str, object], output_path: Path) -> Path:
    output_path = _ensure_grim_ext(output_path)
    with output_path.open("wb") as f:
        np.savez(
            f,
            azimuths=payload["azimuths"],
            elevations=payload["elevations"],
            frequencies=payload["frequencies"],
            polarizations=payload["polarizations"],
            rcs=payload["rcs"],
            rcs_amp=payload["rcs_amp"],
            rcs_real=payload["rcs_real"],
            rcs_imag=payload["rcs_imag"],
            rcs_power=payload["rcs_power"],
            rcs_domain=payload["rcs_domain"],
            power_domain=payload["power_domain"],
            source_path=payload["source_path"],
            history=payload["history"],
            units=payload["units"],
        )
    return output_path


def main() -> int:
    root = INPUT_DIRECTORY.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input directory does not exist: {root}")

    freq_unit = str(OUTPUT_FREQUENCY_UNIT).strip().lower()
    if freq_unit not in {"ghz", "hz"}:
        raise SystemExit(
            f"Invalid OUTPUT_FREQUENCY_UNIT={OUTPUT_FREQUENCY_UNIT!r}. Use 'ghz' or 'hz'."
        )

    csv_paths = _discover_monostatic_csv_files(root, TARGET_FILENAME)
    if not csv_paths:
        raise SystemExit(f"No files named '{TARGET_FILENAME}' were found under: {root}")

    samples, stats = _load_samples(csv_paths, output_freq_unit=freq_unit)
    payload = _build_grid(samples, source_root=root, csv_paths=csv_paths, frequency_unit=freq_unit)

    out_path = OUTPUT_GRIM_FILE.expanduser().resolve()
    out_path = _save_grim(payload, out_path)

    print(f"Found CSV files: {len(csv_paths)}")
    print(f"Rows read: {stats['rows_total']}")
    print(f"Rows skipped (missing frequency/theta/phi): {stats['rows_skipped_base']}")
    print(f"Channel samples loaded: {stats['channels_loaded']}")
    print(f"Channel samples skipped (missing RCS value): {stats['channels_skipped']}")
    print(f"Channel samples missing phase (phase set to 0 deg): {stats['channels_missing_phase']}")
    print(f"Output written: {out_path}")
    print(
        "Axes sizes: "
        f"az={len(payload['azimuths'])}, "
        f"el={len(payload['elevations'])}, "
        f"freq={len(payload['frequencies'])}, "
        f"pol={len(payload['polarizations'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
