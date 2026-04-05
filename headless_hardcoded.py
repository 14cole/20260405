from __future__ import annotations

import json
from typing import Any, Dict, List

from headless_solver import _parse_sweep, run_headless
from solver_benchmarks import run_pec_circle_benchmark_suite


# Edit this block for your run configuration.
CONFIG: Dict[str, Any] = {
    "geometry_path": "square.geo",
    "output_path": "hardcoded_run.grim",
    "units": "inches",  # inches or meters
    "polarization": "TM",  # TM/TE (also accepts HH/VV mappings downstream)
    "workers": 1,  # >1 enables multiprocessing across frequencies
    "quiet": False,
    "history": "solver=headless-hardcoded",
    "csv_output_path": "hardcoded_run.csv",  # set to None or "" to disable CSV
    "json_summary_path": "hardcoded_run_summary.json",  # set to None or "" to disable JSON summary
    "run_benchmarks": False,
    "benchmark_json_path": "hardcoded_benchmarks.json",
    "benchmark": {
        "radius_m": 0.5,
        "frequency_ghz": 1.0,
        "elevations_step_deg": 5.0,
        "mesh_levels": [6, 12, 24],
        "pols": ["TM", "TE"],
    },
    "frequency_mode": "list",  # "list" or "sweep"
    "frequency_list_ghz": [1.0, 2.0, 3.0],
    "frequency_sweep_ghz": {"start": 1.0, "stop": 10.0, "step": 0.5},
    "elevation_mode": "list",  # "list" or "sweep"
    "elevation_list_deg": [0.0, 45.0, 90.0, 135.0, 180.0],
    "elevation_sweep_deg": {"start": 0.0, "stop": 180.0, "step": 2.0},
}


def _build_freqs(cfg: Dict[str, Any]) -> List[float]:
    mode = str(cfg.get("frequency_mode", "list")).strip().lower()
    if mode == "sweep":
        sweep = dict(cfg.get("frequency_sweep_ghz", {}) or {})
        return _parse_sweep(
            float(sweep.get("start", 1.0)),
            float(sweep.get("stop", 10.0)),
            float(sweep.get("step", 1.0)),
            "Frequencies",
        )
    values = cfg.get("frequency_list_ghz", []) or []
    return [float(v) for v in values]


def _build_elevs(cfg: Dict[str, Any]) -> List[float]:
    mode = str(cfg.get("elevation_mode", "list")).strip().lower()
    if mode == "sweep":
        sweep = dict(cfg.get("elevation_sweep_deg", {}) or {})
        return _parse_sweep(
            float(sweep.get("start", 0.0)),
            float(sweep.get("stop", 180.0)),
            float(sweep.get("step", 1.0)),
            "Elevations",
        )
    values = cfg.get("elevation_list_deg", []) or []
    return [float(v) for v in values]


def main() -> int:
    cfg = dict(CONFIG)
    benchmark_cfg = dict(cfg.get("benchmark", {}) or {})

    if bool(cfg.get("run_benchmarks", False)):
        report = run_pec_circle_benchmark_suite(
            radius_m=float(benchmark_cfg.get("radius_m", 0.5)),
            frequency_ghz=float(benchmark_cfg.get("frequency_ghz", 1.0)),
            elevations_step_deg=float(benchmark_cfg.get("elevations_step_deg", 5.0)),
            mesh_levels=[int(v) for v in benchmark_cfg.get("mesh_levels", [6, 12, 24])],
            pols=[str(v).upper() for v in benchmark_cfg.get("pols", ["TM", "TE"])],
        )
        print(json.dumps({"benchmarks": report}, indent=2))
        benchmark_path = str(cfg.get("benchmark_json_path", "")).strip()
        if benchmark_path:
            with open(benchmark_path, "w") as f:
                json.dump(report, f, indent=2)

    freqs = _build_freqs(cfg)
    elevs = _build_elevs(cfg)

    payload = run_headless(
        geometry_path=str(cfg["geometry_path"]),
        output_path=str(cfg["output_path"]),
        frequencies_ghz=freqs,
        elevations_deg=elevs,
        units=str(cfg.get("units", "inches")),
        polarization=str(cfg.get("polarization", "TE")),
        workers=int(cfg.get("workers", 1)),
        csv_output_path=(str(cfg.get("csv_output_path", "")).strip() or None),
        history=str(cfg.get("history", "")),
        quiet=bool(cfg.get("quiet", False)),
    )

    summary = {
        "geometry_path": payload["geometry_path"],
        "workers": payload["workers"],
        "sample_count": len(payload["result"].get("samples", [])),
        "grim_files": payload["grim_files"],
        "csv_file": payload["csv_file"],
        "metadata": payload["result"].get("metadata", {}),
    }
    print(json.dumps(summary, indent=2))

    summary_path = str(cfg.get("json_summary_path", "")).strip()
    if summary_path:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
