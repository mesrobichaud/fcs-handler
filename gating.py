"""
Parse gating definitions from SpectroFlo WTML template files.

Provides gate classes (RectGate, EllipseGate, IntervalGate, NotGate) and the
logicle-based coordinate transform that SpectroFlo uses internally.

Coordinate conversion
---------------------
SpectroFlo stores gate vertices in *normalised display coordinates* [0, 1].
Both "Log" and "Biexponential" scales use a logicle transform with
``T = channel_max`` (2^22 = 4194304) and ``M = log10(T)`` (≈ 6.623 decades).
The ``_XRValue`` / ``_YRValue`` per gate defines the negative data range and
determines the linearisation width ``W``:

    W = (M - log10(T / |XRValue|)) / 2

When ``XRValue >= 0`` (no negative range), W = 0.01 (effectively pure log).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from flowutils.transforms import logicle, logicle_inverse


# ---------------------------------------------------------------------------
# Axis transform
# ---------------------------------------------------------------------------

CHANNEL_MAX = 4_194_304  # 2^22, Cytek Aurora default $PnR


BIEX_M = 4.5  # Decades for Biexponential scale


@dataclass
class SpectroFloTransform:
    """Logicle transform matching SpectroFlo's display coordinates.

    Log scale uses M = log10(channel_max) ≈ 6.623 decades.
    Biexponential scale uses M = 4.5 decades (standard logicle default).
    """
    channel_max: float = CHANNEL_MAX
    r_value: float = 0.0
    A: float = 0.0
    M: float = None  # Set by make_transform based on scale type

    def __post_init__(self):
        if self.M is None:
            self.M = np.log10(self.channel_max)

    @property
    def T(self) -> float:
        return self.channel_max

    @property
    def W(self) -> float:
        if self.r_value >= 0:
            return 0.01
        return (self.M - np.log10(self.channel_max / abs(self.r_value))) / 2

    def data_to_norm(self, values: np.ndarray) -> np.ndarray:
        """Raw data -> normalised [0, 1] via logicle."""
        v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        return logicle(v, [0], t=self.T, m=self.M, w=self.W, a=self.A).ravel()

    def norm_to_data(self, norm: np.ndarray) -> np.ndarray:
        """Normalised [0, 1] -> raw data via inverse logicle."""
        n = np.asarray(norm, dtype=np.float64).reshape(-1, 1)
        return logicle_inverse(n, [0], t=self.T, m=self.M, w=self.W, a=self.A).ravel()


def make_transform(scale: str, r_value: float, **kwargs) -> SpectroFloTransform:
    if scale == 'Biexponential':
        return SpectroFloTransform(r_value=r_value, M=BIEX_M, **kwargs)
    else:  # Log
        return SpectroFloTransform(r_value=r_value, **kwargs)


@dataclass
class QuantileTransform:
    """Map raw values from one instrument to normalised [0, 1] coordinates
    using quantile-based interpolation against a reference instrument.

    Build from paired reference data: for each channel, compute percentiles
    on both instruments. The interpolation maps source raw values → reference
    normalised coordinates so that the same gates apply.
    """
    interp_fn: interp1d = field(repr=False)

    @classmethod
    def build(
        cls,
        ref_values: np.ndarray,
        src_values: np.ndarray,
        ref_transform: SpectroFloTransform,
        n_quantiles: int = 500,
    ) -> QuantileTransform:
        """Build a quantile mapping from paired reference arrays.

        Parameters
        ----------
        ref_values : array
            Raw channel values from the reference instrument (e.g. Aurora).
        src_values : array
            Raw channel values from the source instrument (e.g. FACSCalibur).
        ref_transform : SpectroFloTransform
            The logicle transform used to normalise the reference data.
        n_quantiles : int
            Number of percentile points to use.
        """
        percentiles = np.linspace(0, 100, n_quantiles)
        ref_pct = np.percentile(ref_values, percentiles)
        src_pct = np.percentile(src_values, percentiles)

        # Map source percentile values → reference normalised coordinates
        ref_norm = ref_transform.data_to_norm(ref_pct)

        fn = interp1d(
            src_pct, ref_norm,
            kind='linear',
            bounds_error=False,
            fill_value=(ref_norm[0], ref_norm[-1]),
        )
        return cls(interp_fn=fn)

    def data_to_norm(self, values: np.ndarray) -> np.ndarray:
        """Source raw data → reference normalised [0, 1]."""
        return self.interp_fn(np.asarray(values, dtype=np.float64))

    def norm_to_data(self, norm: np.ndarray) -> np.ndarray:
        """Inverse: reference normalised → source raw data (approximate)."""
        # Build inverse from the stored interpolation
        x = self.interp_fn.x
        y = self.interp_fn.y
        inv_fn = interp1d(
            y, x, kind='linear',
            bounds_error=False,
            fill_value=(x[0], x[-1]),
        )
        return inv_fn(np.asarray(norm, dtype=np.float64))


# ---------------------------------------------------------------------------
# Channel name resolution
# ---------------------------------------------------------------------------

SIGNAL_TYPE_SUFFIX = {
    '65': '-A',   # Area
    '72': '-H',   # Height
}


def _resolve_channel(fluorochrome: str, signal_type: str) -> str:
    """Map WTML fluorochrome + signal type to FCS channel name."""
    suffix = SIGNAL_TYPE_SUFFIX.get(signal_type, '-A')
    return f"{fluorochrome}{suffix}"


# ---------------------------------------------------------------------------
# Gate classes
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    """Base gate – stores metadata but cannot apply itself."""
    name: str
    gate_id: str
    gate_type: str
    channels: list[str]
    scales: list[str]
    r_values: list[float]
    parent: str | None = None
    children: list[Gate] = field(default_factory=list)

    def _get_transform(self, axis: int, **transform_kwargs):
        """Return the transform for the given axis index.

        If ``_qtransforms`` has been set (by ``GateSet.from_mapped``), use the
        quantile transform for the channel.  Otherwise fall back to the
        standard logicle-based ``make_transform``.
        """
        qt = getattr(self, '_qtransforms', None)
        if qt and self.channels[axis] in qt:
            return qt[self.channels[axis]]
        return make_transform(self.scales[axis], self.r_values[axis], **transform_kwargs)

    def apply(self, df: pd.DataFrame, **transform_kwargs) -> np.ndarray:
        raise NotImplementedError(f"apply() not implemented for {self.gate_type}")


@dataclass
class RectGate(Gate):
    """Rotated rectangle gate defined by center, width/height, and angle."""
    center: tuple[float, float] = (0, 0)
    width_height: tuple[float, float] = (0, 0)
    angle_deg: float = 0.0

    def apply(self, df: pd.DataFrame, **transform_kwargs) -> np.ndarray:
        tx = self._get_transform(0, **transform_kwargs)
        ty = self._get_transform(1, **transform_kwargs)

        nx = tx.data_to_norm(df[self.channels[0]].values)
        ny = ty.data_to_norm(df[self.channels[1]].values)

        cx, cy = self.center
        hw = self.width_height[0] / 2
        hh = self.width_height[1] / 2
        theta = np.radians(self.angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        dx = nx - cx
        dy = ny - cy
        rx =  cos_t * dx + sin_t * dy
        ry = -sin_t * dx + cos_t * dy

        return (np.abs(rx) <= hw) & (np.abs(ry) <= hh)


@dataclass
class EllipseGate(Gate):
    """Ellipse gate defined by center, width/height, and rotation angle."""
    center: tuple[float, float] = (0, 0)
    width_height: tuple[float, float] = (0, 0)
    angle_deg: float = 0.0

    def apply(self, df: pd.DataFrame, **transform_kwargs) -> np.ndarray:
        tx = self._get_transform(0, **transform_kwargs)
        ty = self._get_transform(1, **transform_kwargs)

        nx = tx.data_to_norm(df[self.channels[0]].values)
        ny = ty.data_to_norm(df[self.channels[1]].values)

        cx, cy = self.center
        a = self.width_height[0] / 2
        b = self.width_height[1] / 2
        theta = np.radians(self.angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        dx = nx - cx
        dy = ny - cy
        rx =  cos_t * dx + sin_t * dy
        ry = -sin_t * dx + cos_t * dy

        return (rx / a) ** 2 + (ry / b) ** 2 <= 1.0


@dataclass
class IntervalGate(Gate):
    """1-D threshold / interval gate (two x-boundary points)."""
    lo: float = 0.0
    hi: float = 1.0

    def apply(self, df: pd.DataFrame, **transform_kwargs) -> np.ndarray:
        tx = self._get_transform(0, **transform_kwargs)
        lo_data = tx.norm_to_data(np.array([self.lo]))[0]
        hi_data = tx.norm_to_data(np.array([self.hi]))[0]
        values = df[self.channels[0]].values
        return (values >= lo_data) & (values <= hi_data)


@dataclass
class NotGate(Gate):
    """Boolean NOT of a referenced gate."""
    ref_gate: Gate | None = None

    def apply(self, df: pd.DataFrame, **transform_kwargs) -> np.ndarray:
        if self.ref_gate is None:
            raise ValueError(f"NotGate '{self.name}' has no reference gate")
        return ~self.ref_gate.apply(df, **transform_kwargs)


# ---------------------------------------------------------------------------
# WTML parser
# ---------------------------------------------------------------------------

def _ns_strip(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag


def _find_child(elem, local_name: str):
    for child in elem:
        if _ns_strip(child.tag) == local_name:
            return child
    return None


def _get_text(elem, local_name: str, default=''):
    child = _find_child(elem, local_name)
    if child is not None and child.text:
        return child.text
    return default


def _get_float(elem, local_name: str, default=0.0):
    txt = _get_text(elem, local_name)
    try:
        return float(txt)
    except (ValueError, TypeError):
        return default


def _parse_point(point_elem) -> tuple[float, float]:
    return (_get_float(point_elem, '_x'), _get_float(point_elem, '_y'))


def _parse_size(size_elem) -> tuple[float, float]:
    return (_get_float(size_elem, '_width'), _get_float(size_elem, '_height'))


def _parse_gate_elem(gate_elem, parent_name: str | None = None) -> Gate | None:
    """Parse a single GateDesc XML element into a Gate object."""
    gate_type_attr = gate_elem.attrib.get(
        '{http://www.w3.org/2001/XMLSchema-instance}type', ''
    )
    name = _get_text(gate_elem, '_Name')
    gate_id = _get_text(gate_elem, '_Id')

    if not name:
        return None

    channels = []
    params_elem = _find_child(gate_elem, '_GateParameters')
    if params_elem is not None:
        for p in params_elem:
            fluor = _get_text(p, '_Fluorochrome')
            sig = _get_text(p, '_SignalType', '65')
            if fluor:
                channels.append(_resolve_channel(fluor, sig))

    scales = []
    scales_elem = _find_child(gate_elem, '_ParameterScales')
    if scales_elem is not None:
        for s in scales_elem:
            scales.append(s.text or 'Biexponential')

    xr = _get_float(gate_elem, '_XRValue')
    yr = _get_float(gate_elem, '_YRValue')
    r_values = [xr, yr] if len(channels) == 2 else [xr]

    kwargs = dict(
        name=name, gate_id=gate_id, gate_type=gate_type_attr,
        channels=channels, scales=scales, r_values=r_values,
        parent=parent_name,
    )

    if 'NotGateDesc' in gate_type_attr:
        gate = NotGate(**kwargs)

    elif 'IntervalGateDesc' in gate_type_attr:
        vertices_elem = _find_child(gate_elem, '_Vertices')
        lo = hi = 0.5
        if vertices_elem is not None:
            points = [_parse_point(p) for p in vertices_elem]
            xs = [pt[0] for pt in points]
            if xs:
                lo, hi = min(xs), max(xs)
        gate = IntervalGate(lo=lo, hi=hi, **kwargs)

    elif 'EllipseGateDesc' in gate_type_attr:
        center_elem = _find_child(gate_elem, '_Center')
        wh_elem = _find_child(gate_elem, '_WidthHeight')
        center = _parse_point(center_elem) if center_elem is not None else (0.5, 0.5)
        wh = _parse_size(wh_elem) if wh_elem is not None else (0.1, 0.1)
        angle = _get_float(gate_elem, '_RotatingAngle')
        gate = EllipseGate(center=center, width_height=wh, angle_deg=angle, **kwargs)

    elif 'RectGateDesc' in gate_type_attr:
        center_elem = _find_child(gate_elem, '_Center')
        wh_elem = _find_child(gate_elem, '_WidthHeight')
        center = _parse_point(center_elem) if center_elem is not None else (0.5, 0.5)
        wh = _parse_size(wh_elem) if wh_elem is not None else (0.1, 0.1)
        angle = _get_float(gate_elem, '_RotatingAngle')
        gate = RectGate(center=center, width_height=wh, angle_deg=angle, **kwargs)

    else:
        return None

    sub_elem = _find_child(gate_elem, '_SubGates')
    if sub_elem is not None:
        for child_gate_elem in sub_elem:
            if _ns_strip(child_gate_elem.tag) == 'GateDesc':
                child_gate = _parse_gate_elem(child_gate_elem, parent_name=name)
                if child_gate is not None:
                    gate.children.append(child_gate)

    return gate


def parse_wtml(path: str | Path) -> dict[str, Gate]:
    """Parse a SpectroFlo WTML template into a flat dict of gate name -> Gate."""
    tree = ET.parse(str(path))
    root = tree.getroot()

    gates: dict[str, Gate] = {}

    def _collect(elem):
        tag = _ns_strip(elem.tag)
        if tag == 'GateDesc':
            gate = _parse_gate_elem(elem)
            if gate is not None:
                gates[gate.name] = gate
                for child in gate.children:
                    gates[child.name] = child
            return
        for child in elem:
            _collect(child)

    _collect(root)

    for g in gates.values():
        if isinstance(g, NotGate) and g.name.startswith('NOT(') and g.name.endswith(')'):
            ref_name = g.name[4:-1]
            if ref_name in gates:
                g.ref_gate = gates[ref_name]

    return gates
