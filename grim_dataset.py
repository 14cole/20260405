import json
import numpy as np


class RcsGrid:
    """Container for gridded RCS data with axis metadata and helpers."""

    def __init__(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs,
        rcs_imag=None,
        rcs_power=None,
        rcs_domain: str | None = None,
        source_path: str | None = None,
        history: str | None = None,
        units: dict | None = None,
    ):
        """Build a grid from axis arrays and complex RCS samples.

        Use when loading data from files or constructing an in-memory grid.

        Args:
            azimuths: 1D sequence of azimuth values (deg).
            elevations: 1D sequence of elevation values (deg).
            frequencies: 1D sequence of frequency values (GHz or Hz).
            polarizations: 1D sequence of polarization labels.
            rcs: RCS complex field samples shaped (az, el, f, pol). Can be real or complex.
            rcs_imag: Optional imaginary field samples with same shape as rcs.
            rcs_power: Optional linear-power samples with same shape as rcs.
            rcs_domain: Optional domain tag: "complex_amplitude" or "linear_rcs".
            source_path: Optional source path for provenance.
            history: Optional history string.
            units: Optional units dict (e.g., {"azimuth": "deg", "frequency": "GHz"}).

        Raises:
            ValueError: if shapes do not match the expected grid.
        """

        self.azimuths = np.asarray(azimuths)
        self.elevations = np.asarray(elevations)
        self.frequencies = np.asarray(frequencies)
        self.polarizations = np.asarray(polarizations)

        expected = (len(self.azimuths), len(self.elevations), len(self.frequencies), len(self.polarizations))
        rcs_arr = np.asarray(rcs)
        if rcs_imag is not None:
            rcs_imag_arr = np.asarray(rcs_imag)
            if rcs_arr.shape != expected or rcs_imag_arr.shape != expected:
                raise ValueError(
                    f"rcs/rcs_imag shapes {rcs_arr.shape}/{rcs_imag_arr.shape} != {expected}"
                )
            rcs_arr = rcs_arr + 1j * rcs_imag_arr
        elif rcs_arr.shape == expected + (2,):
            rcs_arr = rcs_arr[..., 0] + 1j * rcs_arr[..., 1]

        if rcs_arr.shape != expected:
            raise ValueError(f"rcs shape {rcs_arr.shape} != {expected}")

        if not np.iscomplexobj(rcs_arr):
            rcs_arr = rcs_arr.astype(np.complex128)

        self.rcs = rcs_arr
        domain = str(rcs_domain or "").strip().lower()
        if domain not in {"complex_amplitude", "linear_rcs"}:
            domain = ""

        if rcs_power is not None:
            power_arr = np.asarray(rcs_power, dtype=float)
            if power_arr.shape != expected:
                raise ValueError(f"rcs_power shape {power_arr.shape} != {expected}")
        else:
            power_arr = None

        if domain == "complex_amplitude":
            inferred_power = np.abs(self.rcs) ** 2
            power_arr = inferred_power if power_arr is None else power_arr
        elif domain == "linear_rcs":
            inferred_power = np.real(self.rcs)
            power_arr = inferred_power if power_arr is None else power_arr
        else:
            imag = np.asarray(self.rcs.imag, dtype=float)
            imag_max = float(np.nanmax(np.abs(imag))) if imag.size else 0.0
            if imag_max > 1e-10:
                domain = "complex_amplitude"
                inferred_power = np.abs(self.rcs) ** 2
            else:
                domain = "linear_rcs"
                inferred_power = np.real(self.rcs)
            power_arr = inferred_power if power_arr is None else power_arr

        power_arr = np.asarray(power_arr, dtype=float)
        finite = np.isfinite(power_arr)
        power_clean = np.zeros_like(power_arr, dtype=float)
        power_clean[finite] = np.maximum(power_arr[finite], 0.0)
        power_clean[~finite] = np.nan
        self.rcs_power = power_clean
        self.rcs_domain = domain
        self.power_domain = "linear_rcs"
        self.source_path = source_path
        self.history = history
        self.units = units or {}

    def __len__(self):
        """Return total number of complex samples in the grid."""
        return self.rcs.size

    def get(self, az_idx, el_idx, f_idx, p_idx):
        """Fetch a single sample by axis indices.

        Args:
            az_idx: Azimuth index.
            el_idx: Elevation index.
            f_idx: Frequency index.
            p_idx: Polarization index.

        Returns:
            dict with axis values and complex RCS sample.
        """
        return {
            "azimuth": self.azimuths[az_idx],
            "elevation": self.elevations[el_idx],
            "frequency": self.frequencies[f_idx],
            "polarization": self.polarizations[p_idx],
            "rcs": self.rcs[az_idx, el_idx, f_idx, p_idx],
        }

    def get_axis(self, name):
        """Return a single axis array by name.

        Use when you need a specific axis without unpacking all axes.

        Args:
            name: One of "azimuth", "elevation", "frequency", "polarization".

        Returns:
            Numpy array for the requested axis.
        """
        if name == "azimuth":
            return self.azimuths
        if name == "elevation":
            return self.elevations
        if name == "frequency":
            return self.frequencies
        if name == "polarization":
            return self.polarizations
        raise ValueError(f"unknown axis name: {name}")

    def get_axes(self):
        """Return all axis arrays in a dict."""
        return {
            "azimuths": self.azimuths,
            "elevations": self.elevations,
            "frequencies": self.frequencies,
            "polarizations": self.polarizations,
        }

    def _assert_compatible(self, other):
        """Validate another grid for element-wise operations.

        Use before coherent/incoherent add/subtract operations.

        Args:
            other: Another RcsGrid instance.

        Raises:
            TypeError: if other is not an RcsGrid.
            ValueError: if axes or shapes differ.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        if self.rcs.shape != other.rcs.shape:
            raise ValueError(f"rcs shape {other.rcs.shape} != {self.rcs.shape}")
        if not np.array_equal(self.azimuths, other.azimuths):
            raise ValueError("azimuth axis mismatch")
        if not np.array_equal(self.elevations, other.elevations):
            raise ValueError("elevation axis mismatch")
        if not np.array_equal(self.frequencies, other.frequencies):
            raise ValueError("frequency axis mismatch")
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch")

    def coherent_add(self, other):
        """Coherently add two grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with rcs = self.rcs + other.rcs.
        """
        self._assert_compatible(other)
        rcs_out = self.rcs + other.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_power=np.abs(rcs_out) ** 2,
            rcs_domain="complex_amplitude",
        )

    def coherent_add_many(self, *grids):
        """Coherently add multiple grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with rcs = self.rcs + sum(grid.rcs).
        """
        if not grids:
            return self
        total = np.array(self.rcs, copy=True)
        for grid in grids:
            self._assert_compatible(grid)
            total = total + grid.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            total,
            rcs_power=np.abs(total) ** 2,
            rcs_domain="complex_amplitude",
        )

    def coherent_subtract(self, other):
        """Coherently subtract two grids (complex difference).

        Use when phases are aligned and you want field-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with rcs = self.rcs - other.rcs.
        """
        self._assert_compatible(other)
        rcs_out = self.rcs - other.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_power=np.abs(rcs_out) ** 2,
            rcs_domain="complex_amplitude",
        )

    def incoherent_add(self, other):
        """Incoherently add two grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.rcs_power + other.rcs_power.
        """
        self._assert_compatible(other)
        power_sum = self.rcs_power + other.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            self._amplitude_from_power(power_sum),
            rcs_power=power_sum,
            rcs_domain="linear_rcs",
        )

    def incoherent_add_many(self, *grids):
        """Incoherently add multiple grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with linear power = self.rcs_power + sum(grid.rcs_power).
        """
        if not grids:
            return self
        total = np.array(self.rcs_power, copy=True)
        for grid in grids:
            self._assert_compatible(grid)
            total = total + grid.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            self._amplitude_from_power(total),
            rcs_power=total,
            rcs_domain="linear_rcs",
        )

    def incoherent_subtract(self, other):
        """Incoherently subtract two grids (magnitude difference).

        Use when phases are unrelated and you want power-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = max(self.rcs_power - other.rcs_power, 0).
        """
        self._assert_compatible(other)
        power_diff = self.rcs_power - other.rcs_power
        power_diff = np.maximum(power_diff, 0.0)
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            self._amplitude_from_power(power_diff),
            rcs_power=power_diff,
            rcs_domain="linear_rcs",
        )

    def align_to(self, other, mode="exact"):
        """Align this grid to another grid's axes.

        Modes:
            exact: require identical axes (returns self on success).
            intersect: keep only axis values present in both grids.
            interp: interpolate numeric axes to match other (no extrapolation).

        Args:
            other: Another RcsGrid instance.
            mode: "exact", "intersect", or "interp".

        Returns:
            New RcsGrid aligned to other's axes.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")

        if mode == "exact":
            self._assert_compatible(other)
            return self
        if mode not in ("intersect", "interp"):
            raise ValueError("mode must be 'exact', 'intersect', or 'interp'")

        if mode == "intersect":
            def _match_axis(axis_self, axis_other, tol=1e-6):
                axis_self = np.asarray(axis_self)
                axis_other = np.asarray(axis_other)
                is_numeric = np.issubdtype(axis_self.dtype, np.number)
                keep_other = []
                indices_self = []
                for value in axis_other:
                    if is_numeric:
                        matches = np.where(np.isclose(axis_self, value, atol=tol, rtol=0.0))[0]
                    else:
                        matches = np.where(axis_self == value)[0]
                    if matches.size > 0:
                        keep_other.append(value)
                        indices_self.append(int(matches[0]))
                if not indices_self:
                    raise ValueError("no overlapping axis values for intersect")
                return np.asarray(keep_other), indices_self

            az_new, az_idx = _match_axis(self.azimuths, other.azimuths)
            el_new, el_idx = _match_axis(self.elevations, other.elevations)
            f_new, f_idx = _match_axis(self.frequencies, other.frequencies)
            pol_new, pol_idx = _match_axis(self.polarizations, other.polarizations, tol=0.0)
            rcs_new = self.rcs[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            pwr_new = self.rcs_power[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            return self._new_grid(
                az_new,
                el_new,
                f_new,
                pol_new,
                rcs_new,
                rcs_power=pwr_new,
                rcs_domain=self.rcs_domain,
            )

        # interp mode
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch for interp")

        def _check_sorted(axis, name):
            axis = np.asarray(axis)
            if axis.size < 2:
                return
            if not np.all(np.diff(axis) > 0):
                raise ValueError(f"{name} axis must be strictly increasing for interp")

        _check_sorted(self.azimuths, "azimuth")
        _check_sorted(self.elevations, "elevation")
        _check_sorted(self.frequencies, "frequency")
        _check_sorted(other.azimuths, "azimuth")
        _check_sorted(other.elevations, "elevation")
        _check_sorted(other.frequencies, "frequency")

        def _interp_axis(data, x_old, x_new, axis):
            x_old = np.asarray(x_old, dtype=float)
            x_new = np.asarray(x_new, dtype=float)
            if x_new.min() < x_old.min() or x_new.max() > x_old.max():
                raise ValueError("interp would require extrapolation")
            moved = np.moveaxis(data, axis, 0)
            flat = moved.reshape(moved.shape[0], -1)
            real = np.empty((x_new.size, flat.shape[1]), dtype=float)
            imag = np.empty((x_new.size, flat.shape[1]), dtype=float)
            for i in range(flat.shape[1]):
                real[:, i] = np.interp(x_new, x_old, flat[:, i].real)
                imag[:, i] = np.interp(x_new, x_old, flat[:, i].imag)
            combined = real + 1j * imag
            out = combined.reshape((x_new.size,) + moved.shape[1:])
            return np.moveaxis(out, 0, axis)

        rcs_interp = _interp_axis(self.rcs, self.azimuths, other.azimuths, axis=0)
        rcs_interp = _interp_axis(rcs_interp, self.elevations, other.elevations, axis=1)
        rcs_interp = _interp_axis(rcs_interp, self.frequencies, other.frequencies, axis=2)
        return self._new_grid(
            other.azimuths,
            other.elevations,
            other.frequencies,
            other.polarizations,
            rcs_interp,
            rcs_domain=self.rcs_domain,
        )

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _axis_value_match(axis_arr, value, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        if np.issubdtype(axis_arr.dtype, np.number) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
        return np.where(axis_arr == value)[0]

    @staticmethod
    def _indices_for_axis_values(axis_arr, values, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        indices = []
        for value in values:
            matches = RcsGrid._axis_value_match(axis_arr, value, tol=tol)
            if matches.size == 0:
                return None
            idx = int(matches[0])
            if idx not in indices:
                indices.append(idx)
        return indices

    @staticmethod
    def _axis_union(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        union_values = []
        first_dtype = np.asarray(axis_arrays[0]).dtype
        numeric_axis = np.issubdtype(first_dtype, np.number)
        for axis_arr in axis_arrays:
            for value in np.asarray(axis_arr):
                plain = value.item() if isinstance(value, np.generic) else value
                if numeric_axis:
                    exists = any(
                        np.isclose(float(existing), float(plain), atol=tol, rtol=0.0)
                        for existing in union_values
                    )
                else:
                    exists = any(existing == plain for existing in union_values)
                if not exists:
                    union_values.append(plain)
        if numeric_axis:
            union_values.sort(key=float)
        return np.asarray(union_values)

    @staticmethod
    def _axis_intersection(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        common = [
            value.item() if isinstance(value, np.generic) else value
            for value in np.asarray(axis_arrays[0])
        ]
        for axis_arr in axis_arrays[1:]:
            common = [
                value
                for value in common
                if RcsGrid._axis_value_match(axis_arr, value, tol=tol).size > 0
            ]
            if not common:
                break
        return np.asarray(common)

    @classmethod
    def _ensure_grids(cls, grids):
        checked = []
        for grid in grids:
            if not isinstance(grid, cls):
                raise TypeError("all inputs must be RcsGrid instances")
            checked.append(grid)
        if not checked:
            raise ValueError("at least one grid is required")
        return checked

    def _new_grid(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs,
        *,
        rcs_power=None,
        rcs_domain=None,
        history=None,
    ):
        return RcsGrid(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs,
            rcs_power=rcs_power,
            rcs_domain=(self.rcs_domain if rcs_domain is None else rcs_domain),
            source_path=self.source_path,
            history=history if history is not None else self.history,
            units=dict(self.units),
        )

    def _power_from_values(self, rcs_value):
        values_raw = np.asarray(rcs_value)
        if np.iscomplexobj(values_raw):
            values = np.asarray(values_raw, dtype=np.complex128)
            if self.rcs_domain == "complex_amplitude":
                power = np.abs(values) ** 2
            else:
                power = np.real(values)
        else:
            power = np.asarray(values_raw, dtype=float)
        power = np.asarray(power, dtype=float)
        finite = np.isfinite(power)
        out = np.zeros_like(power, dtype=float)
        out[finite] = np.maximum(power[finite], 0.0)
        out[~finite] = np.nan
        return out

    def _amplitude_from_power(self, power_value):
        power = np.asarray(power_value, dtype=float)
        finite = np.isfinite(power)
        clipped = np.zeros_like(power, dtype=float)
        clipped[finite] = np.maximum(power[finite], 0.0)
        clipped[~finite] = np.nan
        amp = np.sqrt(clipped).astype(np.complex128)
        return amp

    def rcs_to_linear(self, rcs_value):
        """Convert complex field or real-power values to linear power."""
        return self._power_from_values(rcs_value)

    def linear_to_dbsm(self, linear_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        return 10.0 * np.log10(linear)

    def axis_crop(
        self,
        *,
        azimuths=None,
        elevations=None,
        frequencies=None,
        polarizations=None,
        azimuth_range=None,
        elevation_range=None,
        frequency_range=None,
        azimuth_min=None,
        azimuth_max=None,
        elevation_min=None,
        elevation_max=None,
        frequency_min=None,
        frequency_max=None,
        tol=1e-6,
    ):
        """Return a grid cropped by explicit axis values and/or numeric ranges."""

        def _resolve_range(raw_range, vmin, vmax):
            if raw_range is not None:
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError("axis range must be a 2-item [min, max] sequence")
                return raw_range[0], raw_range[1]
            if vmin is None and vmax is None:
                return None
            return vmin, vmax

        azimuth_range = _resolve_range(azimuth_range, azimuth_min, azimuth_max)
        elevation_range = _resolve_range(elevation_range, elevation_min, elevation_max)
        frequency_range = _resolve_range(frequency_range, frequency_min, frequency_max)

        def _axis_indices(axis_arr, axis_values, axis_range, axis_name, axis_tol):
            all_indices = list(range(len(axis_arr)))
            values = self._as_list(axis_values)
            if values is not None:
                selected = self._indices_for_axis_values(axis_arr, values, tol=axis_tol)
                if selected is None:
                    raise ValueError(f"{axis_name} contains value(s) not present in dataset")
                indices = selected
            else:
                indices = all_indices

            if axis_range is not None:
                lo, hi = axis_range
                if lo is not None:
                    lo = float(lo)
                if hi is not None:
                    hi = float(hi)
                if lo is not None and hi is not None and lo > hi:
                    lo, hi = hi, lo

                axis_num = np.asarray(axis_arr, dtype=float)
                range_mask = np.ones(axis_num.shape[0], dtype=bool)
                if lo is not None:
                    range_mask &= axis_num >= (lo - axis_tol)
                if hi is not None:
                    range_mask &= axis_num <= (hi + axis_tol)
                range_idx = set(np.where(range_mask)[0].tolist())
                indices = [idx for idx in indices if idx in range_idx]

            if not indices:
                raise ValueError(f"{axis_name} crop produced no samples")
            return indices

        az_idx = _axis_indices(self.azimuths, azimuths, azimuth_range, "azimuth", tol)
        el_idx = _axis_indices(self.elevations, elevations, elevation_range, "elevation", tol)
        f_idx = _axis_indices(self.frequencies, frequencies, frequency_range, "frequency", tol)
        p_idx = _axis_indices(self.polarizations, polarizations, None, "polarization", 0.0)

        return self._new_grid(
            self.azimuths[az_idx],
            self.elevations[el_idx],
            self.frequencies[f_idx],
            self.polarizations[p_idx],
            self.rcs[np.ix_(az_idx, el_idx, f_idx, p_idx)],
        )

    @classmethod
    def join_many(cls, *grids, tol=1e-6):
        """Join datasets on union axes; later grids overwrite overlaps."""
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            grid = grids[0]
            return grid._new_grid(
                np.array(grid.azimuths, copy=True),
                np.array(grid.elevations, copy=True),
                np.array(grid.frequencies, copy=True),
                np.array(grid.polarizations, copy=True),
                np.array(grid.rcs, copy=True),
                rcs_power=np.array(grid.rcs_power, copy=True),
                rcs_domain=grid.rcs_domain,
            )

        az_union = cls._axis_union([grid.azimuths for grid in grids], tol=tol)
        el_union = cls._axis_union([grid.elevations for grid in grids], tol=tol)
        f_union = cls._axis_union([grid.frequencies for grid in grids], tol=tol)
        p_union = cls._axis_union([grid.polarizations for grid in grids], tol=0.0)

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        joined_rcs = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
        joined_power = np.full(shape, np.nan, dtype=float)

        for grid in grids:
            az_idx = cls._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
            el_idx = cls._indices_for_axis_values(el_union, grid.elevations, tol=tol)
            f_idx = cls._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
            p_idx = cls._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during join")
            joined_rcs[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs
            joined_power[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_power

        last = grids[-1]
        return cls(
            az_union,
            el_union,
            f_union,
            p_union,
            joined_rcs,
            rcs_power=joined_power,
            rcs_domain=last.rcs_domain,
            source_path=last.source_path,
            history=last.history,
            units=dict(last.units),
        )

    @classmethod
    def overlap_many(cls, *grids, tol=1e-6):
        """Return one cropped dataset per input, all on common overlap axes."""
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            return [grids[0]]

        az_common = cls._axis_intersection([grid.azimuths for grid in grids], tol=tol)
        el_common = cls._axis_intersection([grid.elevations for grid in grids], tol=tol)
        f_common = cls._axis_intersection([grid.frequencies for grid in grids], tol=tol)
        p_common = cls._axis_intersection([grid.polarizations for grid in grids], tol=0.0)

        if (
            az_common.size == 0
            or el_common.size == 0
            or f_common.size == 0
            or p_common.size == 0
        ):
            raise ValueError("no overlap across one or more axes")

        overlap_grids = []
        for grid in grids:
            az_idx = cls._indices_for_axis_values(grid.azimuths, az_common, tol=tol)
            el_idx = cls._indices_for_axis_values(grid.elevations, el_common, tol=tol)
            f_idx = cls._indices_for_axis_values(grid.frequencies, f_common, tol=tol)
            p_idx = cls._indices_for_axis_values(grid.polarizations, p_common, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during overlap")

            overlap_grids.append(
                cls(
                    az_common,
                    el_common,
                    f_common,
                    p_common,
                    grid.rcs[np.ix_(az_idx, el_idx, f_idx, p_idx)],
                    rcs_power=grid.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)],
                    rcs_domain=grid.rcs_domain,
                    source_path=grid.source_path,
                    history=grid.history,
                    units=dict(grid.units),
                )
            )

        return overlap_grids

    def difference(self, other, mode="coherent"):
        """Difference between two compatible datasets."""
        if mode == "coherent":
            return self.coherent_subtract(other)
        if mode == "incoherent":
            return self.incoherent_subtract(other)
        if mode in ("db", "dbsm"):
            self._assert_compatible(other)
            diff = self.rcs_to_dbsm(self.rcs) - self.rcs_to_dbsm(other.rcs)
            return self._new_grid(
                np.array(self.azimuths, copy=True),
                np.array(self.elevations, copy=True),
                np.array(self.frequencies, copy=True),
                np.array(self.polarizations, copy=True),
                diff,
                rcs_domain="linear_rcs",
            )
        raise ValueError("mode must be 'coherent', 'incoherent', or 'db'")

    def statistics_dataset(
        self,
        statistic="mean",
        axes=("azimuth", "elevation", "frequency"),
        *,
        domain="magnitude",
        percentile=50.0,
        broadcast_reduced=False,
    ):
        """Compute a statistic over selected axes and return a dataset."""
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2, "polarization": 3}
        axis_alias = {
            "azimuths": "azimuth",
            "elevations": "elevation",
            "frequencies": "frequency",
            "polarizations": "polarization",
            "az": "azimuth",
            "el": "elevation",
            "freq": "frequency",
            "pol": "polarization",
        }

        axes_list = self._as_list(axes)
        if axes_list is None:
            raise ValueError("axes must include at least one axis")
        reduce_axes = []
        for axis_name in axes_list:
            key = str(axis_name).strip().lower()
            key = axis_alias.get(key, key)
            if key not in axis_map:
                raise ValueError(f"unknown axis: {axis_name}")
            idx = axis_map[key]
            if idx not in reduce_axes:
                reduce_axes.append(idx)
        if not reduce_axes:
            raise ValueError("axes must include at least one axis")
        reduce_axes = tuple(sorted(reduce_axes))

        if domain == "complex":
            values = self.rcs
        elif domain == "magnitude":
            values = self.rcs_power
        elif domain in ("db", "dbsm"):
            values = self.linear_to_dbsm(self.rcs_power)
        else:
            raise ValueError("domain must be 'complex', 'magnitude', or 'dbsm'")

        stat_key = str(statistic).strip().lower()
        if stat_key.startswith("p") and stat_key[1:].replace(".", "", 1).isdigit():
            percentile = float(stat_key[1:])
            stat_key = "percentile"

        if domain == "complex" and stat_key == "percentile":
            raise ValueError("percentile on complex values is not supported; use magnitude or dbsm domain")

        if stat_key == "mean":
            reduced = np.nanmean(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "median":
            reduced = np.nanmedian(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "min":
            reduced = np.nanmin(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "max":
            reduced = np.nanmax(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "std":
            reduced = np.nanstd(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "percentile":
            reduced = np.nanpercentile(values, float(percentile), axis=reduce_axes, keepdims=True)
        else:
            raise ValueError(
                "statistic must be mean, median, min, max, std, percentile, or pXX (for percentile XX)"
            )

        axis_values = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        if broadcast_reduced:
            # Repeat the reduced result across each reduced axis so the output
            # keeps original axis lengths for downstream plotting.
            reduced = np.broadcast_to(reduced, values.shape).copy()
        else:
            for axis_idx in reduce_axes:
                original = axis_values[axis_idx]
                if axis_idx == 3:
                    axis_values[axis_idx] = np.asarray(["ALL"])
                else:
                    numeric = np.asarray(original, dtype=float)
                    rep = float(np.nanmean(numeric)) if numeric.size else 0.0
                    axis_values[axis_idx] = np.asarray([rep], dtype=float)

        if domain == "complex":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                reduced,
                rcs_domain="complex_amplitude",
            )
        if domain == "magnitude":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                self._amplitude_from_power(reduced),
                rcs_power=np.asarray(reduced, dtype=float),
                rcs_domain="linear_rcs",
            )
        # db domain: keep values for plotting/export convenience.
        return self._new_grid(
            axis_values[0],
            axis_values[1],
            axis_values[2],
            axis_values[3],
            np.asarray(reduced, dtype=np.complex128),
            rcs_domain="linear_rcs",
        )

    def _index_for_value(self, axis, value, tol=0.0):
        """Find the first index of a value on an axis.

        Args:
            axis: 1D array to search.
            value: Value to find.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Integer index of the first match.

        Raises:
            ValueError: if no match is found.
        """
        axis_arr = np.asarray(axis)
        if tol > 0.0:
            matches = np.where(np.isclose(axis_arr, value, atol=tol, rtol=0.0))[0]
        else:
            matches = np.where(axis_arr == value)[0]
        if matches.size == 0:
            raise ValueError(f"value {value} not found on axis")
        return int(matches[0])

    def get_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0):
        """Fetch a single sample by axis values.

        Use when you have physical axis values rather than indices.

        Args:
            azimuth: Azimuth value.
            elevation: Elevation value.
            frequency: Frequency value.
            polarization: Polarization label.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Complex RCS sample.
        """
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
        return self.rcs[az_idx, el_idx, f_idx, p_idx]

    def rcs_to_dbsm(self, rcs_value, eps=1e-12):
        """Convert linear RCS to dBsm.

        Args:
            rcs_value: Complex or real RCS value(s).
            eps: Floor to avoid log(0).

        Returns:
            dBsm value(s) as float or ndarray.
        """
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbsm(linear, eps=eps)

    def get_dbsm(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBsm."""
        return self.rcs_to_dbsm(self.rcs[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbsm_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBsm."""
        return self.rcs_to_dbsm(
            self.get_by_value(azimuth, elevation, frequency, polarization, tol=tol),
            eps=eps,
        )

    def save(self, path):
        """Save the grid to a .grim (npz) file.

        Args:
            path: Output path, with or without .grim.

        Returns:
            The actual path written (always ends with .grim).
        """
        if not path.endswith(".grim"):
            path = f"{path}.grim"
        with open(path, "wb") as f:
            units_payload = json.dumps(self.units) if self.units else ""
            np.savez(
                f,
                azimuths=self.azimuths,
                elevations=self.elevations,
                frequencies=self.frequencies,
                polarizations=self.polarizations,
                rcs=self.rcs,
                rcs_amp=self.rcs,
                rcs_real=np.real(self.rcs).astype(float),
                rcs_imag=np.imag(self.rcs).astype(float),
                rcs_power=self.rcs_power,
                rcs_domain=self.rcs_domain,
                power_domain=self.power_domain,
                source_path=self.source_path if self.source_path is not None else "",
                history=self.history if self.history is not None else "",
                units=units_payload,
            )
        return path

    @classmethod
    def load(cls, path, mmap_mode: str | None = None):
        """Load a grid from a .grim (npz) file.

        Args:
            path: Input path, with or without .grim.
            mmap_mode: Optional numpy mmap mode (e.g., "r") for lazy loading.

        Returns:
            RcsGrid instance loaded from disk.
        """
        if not path.endswith(".grim"):
            path = f"{path}.grim"
        with open(path, "rb") as f:
            data = np.load(f, mmap_mode=mmap_mode, allow_pickle=False)

            units = {}
            if "units" in data:
                raw_units = data["units"]
                if isinstance(raw_units, np.ndarray):
                    raw_units = raw_units.item()
                if isinstance(raw_units, bytes):
                    raw_units = raw_units.decode("utf-8")
                if isinstance(raw_units, str) and raw_units:
                    try:
                        units = json.loads(raw_units)
                    except json.JSONDecodeError:
                        units = {}
                elif isinstance(raw_units, dict):
                    units = raw_units

            source_path_raw = data["source_path"].item() if "source_path" in data else None
            source_path = source_path_raw if source_path_raw else None
            history_raw = data["history"].item() if "history" in data else None
            history = history_raw if history_raw else None
            rcs_domain = ""
            if "rcs_domain" in data:
                raw_domain = data["rcs_domain"]
                if isinstance(raw_domain, np.ndarray):
                    raw_domain = raw_domain.item()
                if isinstance(raw_domain, bytes):
                    raw_domain = raw_domain.decode("utf-8")
                if isinstance(raw_domain, str):
                    rcs_domain = raw_domain.strip().lower()
            if "rcs_real" in data and "rcs_imag" in data:
                amp = np.asarray(data["rcs_real"], dtype=np.complex128) + 1j * np.asarray(
                    data["rcs_imag"], dtype=np.complex128
                )
                pwr = data["rcs_power"] if "rcs_power" in data else (
                    np.real(np.asarray(data["rcs"], dtype=np.complex128)) if "rcs" in data else None
                )
                if "rcs_power" not in data and pwr is not None:
                    phase = np.angle(amp)
                    amp = np.sqrt(np.maximum(np.asarray(pwr, dtype=float), 0.0)) * np.exp(1j * phase)
                return cls(
                    data["azimuths"],
                    data["elevations"],
                    data["frequencies"],
                    data["polarizations"],
                    amp,
                    rcs_power=pwr,
                    rcs_domain=(rcs_domain or "complex_amplitude"),
                    source_path=source_path,
                    history=history,
                    units=units,
                )
            if "rcs_amp" in data:
                amp = np.asarray(data["rcs_amp"], dtype=np.complex128)
                pwr = data["rcs_power"] if "rcs_power" in data else (
                    np.real(np.asarray(data["rcs"], dtype=np.complex128)) if "rcs" in data else None
                )
                if "rcs_power" not in data and pwr is not None:
                    phase = np.angle(amp)
                    amp = np.sqrt(np.maximum(np.asarray(pwr, dtype=float), 0.0)) * np.exp(1j * phase)
                return cls(
                    data["azimuths"],
                    data["elevations"],
                    data["frequencies"],
                    data["polarizations"],
                    amp,
                    rcs_power=pwr,
                    rcs_domain=(rcs_domain or "complex_amplitude"),
                    source_path=source_path,
                    history=history,
                    units=units,
                )
            if rcs_domain == "complex_amplitude":
                return cls(
                    data["azimuths"],
                    data["elevations"],
                    data["frequencies"],
                    data["polarizations"],
                    data["rcs"],
                    rcs_power=(data["rcs_power"] if "rcs_power" in data else None),
                    rcs_domain="complex_amplitude",
                    source_path=source_path,
                    history=history,
                    units=units,
                )
            return cls(
                data["azimuths"],
                data["elevations"],
                data["frequencies"],
                data["polarizations"],
                data["rcs"],
                rcs_power=(data["rcs_power"] if "rcs_power" in data else None),
                rcs_domain=(rcs_domain or None),
                source_path=source_path,
                history=history,
                units=units,
            )
    
