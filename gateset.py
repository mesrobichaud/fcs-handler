"""
GateSet: high-level interface for applying and visualising SpectroFlo gates.

Example::

    from experiment_handler.gateset import GateSet

    gs = GateSet.from_wtml('processed_data/MR_PLT.WTML')
    gs.view(adata)
    adata_plt = gs.apply(adata, 'PLT')
    gs.plot(adata, 'PLT')
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.transforms import Affine2D

from .gating import (
    Gate,
    EllipseGate,
    IntervalGate,
    NotGate,
    RectGate,
    QuantileTransform,
    SpectroFloTransform,
    make_transform,
    parse_wtml,
)


def _adata_to_df(adata) -> pd.DataFrame:
    return pd.DataFrame(adata.X, columns=list(adata.var_names))


class GateSet:
    """A collection of parsed gates from a SpectroFlo WTML template.

    Example::

        gs = GateSet.from_wtml('MR_PLT.WTML')
        gs.view(adata)
        adata_plt = gs.apply(adata, 'PLT')
        gs.plot(adata, 'PLT')
    """

    def __init__(self, gates: dict[str, Gate]) -> None:
        self._gates = gates

    @classmethod
    def from_wtml(cls, path: str | Path) -> GateSet:
        """Parse a SpectroFlo WTML template file."""
        return cls(parse_wtml(path))

    @classmethod
    def from_mapped(
        cls,
        source_gs: GateSet,
        channel_map: dict[str, str],
        ref_adata,
        src_adata,
        gate_names: list[str],
        n_quantiles: int = 500,
        gate_ignore: list[str] | None = None,
    ) -> GateSet:
        """Create a GateSet that applies Aurora gates to a different instrument.

        Uses quantile-based normalisation: for each channel pair, percentiles
        from reference data on both instruments are matched so that the same
        gate boundaries (in normalised [0, 1] space) produce equivalent results.

        Parameters
        ----------
        source_gs : GateSet
            Original GateSet (e.g. parsed from a WTML file for Cytek Aurora).
        channel_map : dict
            Mapping from Aurora channel names to source instrument channel names.
            E.g. ``{'FSC-H': 'FSC-H', 'SSC-B-H': 'SSC-H', 'B2-A': 'FL1-H'}``.
        ref_adata : AnnData
            Reference data from the Aurora (used to compute reference percentiles).
        src_adata : AnnData
            Reference data from the source instrument (e.g. FACSCalibur).
        gate_names : list[str]
            Which gates to include (with their parent hierarchy).
        n_quantiles : int
            Number of percentile points for the quantile mapping.
        gate_ignore : list[str], optional
            Gate names to exclude. If an ignored gate appears in the parent
            hierarchy, it is skipped and the child becomes a root gate.

        Returns
        -------
        GateSet
            New GateSet whose gates use quantile transforms and remapped channels.
        """
        ignore = set(gate_ignore or [])
        ref_df = _adata_to_df(ref_adata)
        src_df = _adata_to_df(src_adata)

        # Collect all gates needed (including parents), skipping ignored gates
        needed = set()
        for name in gate_names:
            if name in ignore:
                continue
            needed.add(name)
            g = source_gs[name]
            while g.parent and g.parent in source_gs.gates:
                if g.parent in ignore:
                    break
                needed.add(g.parent)
                g = source_gs[g.parent]

        # Build quantile transforms per Aurora channel
        qtransforms: dict[str, QuantileTransform] = {}
        for gate_name in needed:
            gate = source_gs[gate_name]
            if isinstance(gate, NotGate):
                continue
            for i, aurora_ch in enumerate(gate.channels):
                if aurora_ch in qtransforms:
                    continue
                if aurora_ch not in channel_map:
                    raise ValueError(
                        f"Aurora channel '{aurora_ch}' not in channel_map"
                    )
                src_ch = channel_map[aurora_ch]
                ref_transform = make_transform(
                    gate.scales[i], gate.r_values[i],
                )
                qtransforms[aurora_ch] = QuantileTransform.build(
                    ref_values=ref_df[aurora_ch].values,
                    src_values=src_df[src_ch].values,
                    ref_transform=ref_transform,
                    n_quantiles=n_quantiles,
                )

        # Deep-copy needed gates, remap channels, store quantile transforms
        new_gates: dict[str, Gate] = {}
        for name in needed:
            gate = copy.deepcopy(source_gs[name])

            # Clear parent if it was ignored or not included
            if gate.parent and gate.parent not in needed:
                gate.parent = None

            if isinstance(gate, NotGate):
                # Re-link ref_gate to the new copy
                if gate.ref_gate and gate.ref_gate.name in needed:
                    gate.ref_gate = None  # will re-link below
                new_gates[name] = gate
                continue

            mapped_channels = []
            for ch in gate.channels:
                mapped_channels.append(channel_map[ch])
            gate.channels = mapped_channels

            # Store quantile transforms on the gate for apply/plot
            gate._qtransforms = {
                channel_map[aurora_ch]: qtransforms[aurora_ch]
                for aurora_ch in source_gs[name].channels
            }

            new_gates[name] = gate

        # Re-link NotGate references
        for g in new_gates.values():
            if isinstance(g, NotGate) and g.ref_gate is None:
                ref_name = g.name[4:-1] if g.name.startswith('NOT(') else None
                if ref_name and ref_name in new_gates:
                    g.ref_gate = new_gates[ref_name]

        # Re-link children
        for g in new_gates.values():
            g.children = [
                new_gates[c.name] for c in g.children
                if c.name in new_gates
            ]

        gs = cls(new_gates)
        gs._qtransforms = qtransforms
        gs._channel_map = channel_map
        return gs

    @property
    def gates(self) -> dict[str, Gate]:
        return self._gates

    @property
    def names(self) -> list[str]:
        return list(self._gates.keys())

    def __getitem__(self, name: str) -> Gate:
        return self._gates[name]

    def __repr__(self) -> str:
        return f"GateSet({list(self._gates.keys())})"

    # -----------------------------------------------------------------
    # save / load
    # -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the GateSet to a pickle file (preserves all transforms)."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> GateSet:
        """Load a GateSet from a pickle file."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    # -----------------------------------------------------------------
    # modify
    # -----------------------------------------------------------------

    def move(self, gate_name: str, dx: float = 0, dy: float = 0) -> None:
        """Shift a gate's center by (dx, dy) in normalised coordinates.

        Parameters
        ----------
        gate_name : str
            Name of the gate to move.
        dx, dy : float
            Offset in normalised [0, 1] display coordinates.
        """
        gate = self._gates[gate_name]
        if not hasattr(gate, 'center'):
            raise ValueError(
                f"Gate '{gate_name}' ({gate.__class__.__name__}) has no center"
            )
        cx, cy = gate.center
        gate.center = (cx + dx, cy + dy)

    def resize(self, gate_name: str, scale_x: float = 1, scale_y: float = 1) -> None:
        """Scale a gate's width/height by the given factors.

        Parameters
        ----------
        gate_name : str
            Name of the gate to resize.
        scale_x, scale_y : float
            Multiplicative factors for width and height.
        """
        gate = self._gates[gate_name]
        if not hasattr(gate, 'width_height'):
            raise ValueError(
                f"Gate '{gate_name}' ({gate.__class__.__name__}) has no width_height"
            )
        w, h = gate.width_height
        gate.width_height = (w * scale_x, h * scale_y)

    def reshape(
        self,
        gate_name: str,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """Set new axis-aligned bounds for a rectangular gate.

        The gate's center, width, and height are recomputed from the bounds.
        The rotation angle is reset to 0.

        Parameters
        ----------
        gate_name : str
            Name of the gate to reshape.
        x_min, x_max : float
            New x-axis bounds in normalised [0, 1] coordinates.
        y_min, y_max : float
            New y-axis bounds in normalised [0, 1] coordinates.
        """
        gate = self._gates[gate_name]
        if not hasattr(gate, 'center') or not hasattr(gate, 'width_height'):
            raise ValueError(
                f"Gate '{gate_name}' ({gate.__class__.__name__}) "
                "does not support reshape"
            )
        gate.center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        gate.width_height = (x_max - x_min, y_max - y_min)
        gate.angle_deg = 0.0

    def add_gate(
        self,
        name: str,
        gate_type: str,
        channels: list[str],
        scales: list[str],
        r_values: list[float],
        parent: str | None = None,
        *,
        center: tuple[float, float] | None = None,
        width_height: tuple[float, float] | None = None,
        angle_deg: float = 0.0,
        lo: float = 0.0,
        hi: float = 1.0,
        create_not: bool = False,
    ) -> None:
        """Add a new gate to the GateSet.

        Parameters
        ----------
        name : str
            Gate name (must be unique).
        gate_type : 'rect', 'ellipse', or 'interval'
            Type of gate to create.
        channels : list[str]
            Channel names (2 for rect/ellipse, 1 for interval).
        scales : list[str]
            Scale type per channel ('Log' or 'Biexponential').
        r_values : list[float]
            R-value per channel (controls logicle W parameter).
        parent : str, optional
            Name of an existing parent gate.
        center : tuple[float, float]
            Center in normalised [0, 1] coordinates (rect/ellipse).
        width_height : tuple[float, float]
            Width and height in normalised coordinates (rect/ellipse).
        angle_deg : float
            Rotation angle in degrees CCW (rect/ellipse).
        lo, hi : float
            Lower/upper bounds in normalised coordinates (interval).
        create_not : bool
            If True, also create a ``NOT(name)`` gate that inverts this gate.
        """
        if name in self._gates:
            raise ValueError(f"Gate '{name}' already exists")
        if parent and parent not in self._gates:
            raise KeyError(f"Parent gate '{parent}' not found")

        gate_type_lower = gate_type.lower()
        kwargs = dict(
            name=name,
            gate_id='',
            gate_type=gate_type_lower,
            channels=channels,
            scales=scales,
            r_values=r_values,
            parent=parent,
        )

        if gate_type_lower == 'rect':
            gate = RectGate(
                center=center or (0.5, 0.5),
                width_height=width_height or (0.1, 0.1),
                angle_deg=angle_deg,
                **kwargs,
            )
        elif gate_type_lower == 'ellipse':
            gate = EllipseGate(
                center=center or (0.5, 0.5),
                width_height=width_height or (0.1, 0.1),
                angle_deg=angle_deg,
                **kwargs,
            )
        elif gate_type_lower == 'interval':
            gate = IntervalGate(lo=lo, hi=hi, **kwargs)
        else:
            raise ValueError(
                f"Unknown gate_type '{gate_type}'. Use 'rect', 'ellipse', or 'interval'."
            )

        self._gates[name] = gate
        if parent:
            self._gates[parent].children.append(gate)

        if create_not:
            not_name = f'NOT({name})'
            not_gate = NotGate(
                name=not_name,
                gate_id='',
                gate_type='not',
                channels=[],
                scales=[],
                r_values=[],
                parent=parent,
                ref_gate=gate,
            )
            self._gates[not_name] = not_gate

    # -----------------------------------------------------------------
    # count
    # -----------------------------------------------------------------

    def count(
        self,
        collection,
        gate_names: list[str] | None = None,
        obs_cols: list[str] | None = None,
        **transform_kwargs,
    ) -> pd.DataFrame:
        """Count events per gate across an FCSCollection.

        Parameters
        ----------
        collection : FCSCollection
            Collection of FCS series.
        gate_names : list[str], optional
            Gates to count. Defaults to all gates in the GateSet.
        obs_cols : list[str], optional
            Obs columns to group by. Each unique combination of values
            becomes one row. If None, one row per series.

        Returns
        -------
        pd.DataFrame
            Columns are uns metadata, obs grouping columns, 'total',
            then '{gate}_count' and '{gate}_pct' for each gate.
        """
        if gate_names is None:
            gate_names = self.names

        rows = []
        for ad in collection._series:
            meta = dict(ad.uns.get('metadata', {}))

            if obs_cols:
                grouped = ad.obs.groupby(obs_cols, observed=True)
                groups = [(key, ad[idx.index]) for key, idx in grouped]
            else:
                groups = [(None, ad)]

            for key, sub_ad in groups:
                row = dict(meta)
                if key is not None:
                    if isinstance(key, tuple):
                        for col, val in zip(obs_cols, key):
                            row[col] = val
                    else:
                        row[obs_cols[0]] = key

                df = _adata_to_df(sub_ad)
                total = len(df)
                row['total'] = total

                for gn in gate_names:
                    try:
                        mask = self._apply_hierarchy(df, gn, **transform_kwargs)
                        n = int(mask.sum())
                        row[f'{gn}_count'] = n
                        row[f'{gn}_pct'] = 100 * n / total if total > 0 else 0.0
                    except Exception as e:
                        row[f'{gn}_count'] = None
                        row[f'{gn}_pct'] = None
                rows.append(row)

        return pd.DataFrame(rows)

    def median(
        self,
        collection,
        gate_name: str,
        channels: list[str] | None = None,
        obs_cols: list[str] | None = None,
        **transform_kwargs,
    ) -> pd.DataFrame:
        """Compute median channel values within a gate for each group.

        Parameters
        ----------
        collection : FCSCollection
            Collection of FCS series.
        gate_name : str
            Gate to apply before computing medians.
        channels : list[str], optional
            Channels to compute medians for. Defaults to all channels.
        obs_cols : list[str], optional
            Obs columns to group by. Each unique combination becomes
            one row. If None, one row per series.

        Returns
        -------
        pd.DataFrame
            Columns are uns metadata, obs grouping columns, 'n_events',
            then one column per channel with the median value.
        """
        rows = []
        for ad in collection._series:
            meta = dict(ad.uns.get('metadata', {}))

            if obs_cols:
                grouped = ad.obs.groupby(obs_cols, observed=True)
                groups = [(key, ad[idx.index]) for key, idx in grouped]
            else:
                groups = [(None, ad)]

            for key, sub_ad in groups:
                row = dict(meta)
                if key is not None:
                    if isinstance(key, tuple):
                        for col, val in zip(obs_cols, key):
                            row[col] = val
                    else:
                        row[obs_cols[0]] = key

                df = _adata_to_df(sub_ad)
                mask = self._apply_hierarchy(df, gate_name, **transform_kwargs)
                gated = df.loc[mask]
                row['n_events'] = int(mask.sum())

                cols = channels if channels else list(df.columns)
                for ch in cols:
                    row[ch] = gated[ch].median() if len(gated) > 0 else None

                rows.append(row)

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # apply
    # -----------------------------------------------------------------

    def apply(
        self,
        adata,
        gate_name: str,
        as_mask: bool = False,
        **transform_kwargs,
    ):
        """Apply a gate (with parent hierarchy) to an AnnData object.

        Parameters
        ----------
        adata : AnnData
            AnnData whose ``var`` index holds FCS channel names.
        gate_name : str
            Name of the gate to apply.
        as_mask : bool
            If True, return a boolean array instead of a filtered AnnData view.

        Returns
        -------
        AnnData or np.ndarray
        """
        df = _adata_to_df(adata)
        mask = self._apply_hierarchy(df, gate_name, **transform_kwargs)
        if as_mask:
            return mask
        return adata[mask]

    def _apply_hierarchy(
        self, df: pd.DataFrame, gate_name: str, **transform_kwargs
    ) -> np.ndarray:
        gate = self._gates[gate_name]
        mask = gate.apply(df, **transform_kwargs)
        if gate.parent and gate.parent in self._gates:
            parent_mask = self._apply_hierarchy(
                df, gate.parent, **transform_kwargs
            )
            mask = mask & parent_mask
        return mask

    # -----------------------------------------------------------------
    # view
    # -----------------------------------------------------------------

    def view(self, adata=None, **transform_kwargs) -> None:
        """Print the gating hierarchy as an indented tree.

        Parameters
        ----------
        adata : AnnData, optional
            If provided, event counts and percentages are shown for each gate.
        """
        df = None
        total = None
        if adata is not None:
            df = _adata_to_df(adata)
            total = len(df)

        root_names = [
            name for name, g in self._gates.items()
            if g.parent is None
        ]

        def _format_gate(g: Gate) -> str:
            gtype = g.__class__.__name__
            if isinstance(g, NotGate):
                return f"{g.name} [NOT]"
            elif len(g.channels) == 2:
                return (
                    f"{g.name} ({gtype}: {g.channels[0]} vs {g.channels[1]})"
                )
            elif len(g.channels) == 1:
                return f"{g.name} ({gtype}: {g.channels[0]})"
            return f"{g.name} ({gtype})"

        def _count_str(gate_name: str) -> str:
            if df is None:
                return ""
            try:
                mask = self._apply_hierarchy(
                    df, gate_name, **transform_kwargs
                )
                n = mask.sum()
                pct = 100 * n / total
                return f" — {n:,} ({pct:.1f}%)"
            except Exception as e:
                return f" — ERROR: {e}"

        def _print_subtree(name: str, prefix: str, is_last: bool):
            g = self._gates[name]
            connector = "└── " if is_last else "├── "
            print(prefix + connector + _format_gate(g) + _count_str(name))

            child_names = [
                c.name for c in g.children if c.name in self._gates
            ]
            not_children = [
                ng.name
                for ng in self._gates.values()
                if isinstance(ng, NotGate)
                and ng.ref_gate is g
                and ng.name not in child_names
            ]
            all_children = child_names + not_children

            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child_name in enumerate(all_children):
                _print_subtree(
                    child_name, new_prefix, i == len(all_children) - 1
                )

        header = "All Events"
        if total is not None:
            header += f" ({total:,})"
        print(header)

        display_roots = [
            name
            for name in root_names
            if not (
                isinstance(self._gates[name], NotGate)
                and self._gates[name].ref_gate is not None
            )
        ]
        for i, name in enumerate(display_roots):
            _print_subtree(name, "", i == len(display_roots) - 1)

    # -----------------------------------------------------------------
    # plot
    # -----------------------------------------------------------------

    def plot(
        self,
        adata,
        gate_name: str,
        ax=None,
        gate_color: str = "red",
        data_kwargs: dict | None = None,
        show_parent: bool = True,
        **transform_kwargs,
    ):
        """Plot data with the gate shape overlaid.

        2-D gates (Rect, Ellipse) produce a scatter plot in normalised
        coordinates.  1-D gates (Interval) produce a histogram with vertical
        threshold lines.  NotGates plot the referenced gate with a dashed
        boundary.

        Parameters
        ----------
        adata : AnnData
            Data to plot.
        gate_name : str
            Name of the gate to overlay.
        ax : matplotlib Axes, optional
            Axes to draw on.  Created if None.
        gate_color : str
            Color for the gate boundary.
        data_kwargs : dict, optional
            Extra kwargs forwarded to the scatter / histogram call.
        show_parent : bool
            If True and the gate has a parent, only show events passing the
            parent gate (i.e. the input population).
        """
        gate = self._gates[gate_name]

        # Resolve NotGate
        plot_gate = gate
        is_not = False
        if isinstance(gate, NotGate) and gate.ref_gate is not None:
            plot_gate = gate.ref_gate
            is_not = True

        df = _adata_to_df(adata)

        parent_mask = np.ones(len(df), dtype=bool)
        if (
            show_parent
            and plot_gate.parent
            and plot_gate.parent in self._gates
        ):
            parent_mask = self._apply_hierarchy(
                df, plot_gate.parent, **transform_kwargs
            )

        if data_kwargs is None:
            data_kwargs = {}

        if len(plot_gate.channels) == 1:
            return self._plot_interval(
                df, plot_gate, parent_mask, ax, gate_color,
                is_not, data_kwargs, **transform_kwargs,
            )
        elif len(plot_gate.channels) == 2:
            return self._plot_2d(
                df, plot_gate, parent_mask, ax, gate_color,
                is_not, data_kwargs, **transform_kwargs,
            )
        else:
            raise ValueError(
                f"Cannot plot gate '{plot_gate.name}': "
                f"expected 1 or 2 channels, got {len(plot_gate.channels)}"
            )

    def plot_all(
        self,
        adata,
        gate_names: list[str] | None = None,
        gate_color: str = "red",
        data_kwargs: dict | None = None,
        ncols: int = 4,
        figsize_per_ax: tuple[float, float] = (4, 3.5),
        **transform_kwargs,
    ):
        """Plot all gates in a grid.

        Parameters
        ----------
        adata : AnnData
            Data to plot.
        gate_names : list[str], optional
            Gates to plot. Defaults to all gates (excluding NotGates).
        gate_color : str
            Color for gate boundaries.
        data_kwargs : dict, optional
            Extra kwargs forwarded to scatter / histogram calls.
        ncols : int
            Number of columns in the grid.
        figsize_per_ax : tuple
            (width, height) per subplot in inches.
        """
        if gate_names is None:
            gate_names = [
                n for n, g in self._gates.items()
                if not isinstance(g, NotGate)
            ]

        n = len(gate_names)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        )
        axes_flat = np.asarray(axes).flatten()

        for i, gn in enumerate(gate_names):
            self.plot(
                adata, gn, ax=axes_flat[i],
                gate_color=gate_color, data_kwargs=data_kwargs,
                **transform_kwargs,
            )

        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        return fig, axes

    def _plot_2d(
        self, df, gate, parent_mask, ax, gate_color,
        is_not, data_kwargs, **transform_kwargs,
    ):
        tx = gate._get_transform(0, **transform_kwargs)
        ty = gate._get_transform(1, **transform_kwargs)

        x_all = tx.data_to_norm(df[gate.channels[0]].values)
        y_all = ty.data_to_norm(df[gate.channels[1]].values)
        x = x_all[parent_mask]
        y = y_all[parent_mask]

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 3))

        scatter_defaults = dict(s=1, alpha=0.3, c="steelblue", rasterized=True)
        scatter_defaults.update(data_kwargs)
        ax.scatter(x, y, **scatter_defaults)

        cx, cy = gate.center
        w, h = gate.width_height
        linestyle = "--" if is_not else "-"

        if isinstance(gate, EllipseGate):
            patch = Ellipse(
                (cx, cy), w, h,
                angle=gate.angle_deg,
                fill=False, edgecolor=gate_color,
                linewidth=1.5, linestyle=linestyle,
            )
        else:
            patch = Rectangle(
                (cx - w / 2, cy - h / 2), w, h,
                fill=False, edgecolor=gate_color,
                linewidth=1.5, linestyle=linestyle,
            )
            t = (
                Affine2D().rotate_deg_around(cx, cy, gate.angle_deg)
                + ax.transData
            )
            patch.set_transform(t)

        ax.add_patch(patch)

        label = f"NOT({gate.name})" if is_not else gate.name
        ax.set_xlabel(gate.channels[0])
        ax.set_ylabel(gate.channels[1])
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def _plot_interval(
        self, df, gate, parent_mask, ax, gate_color,
        is_not, data_kwargs, **transform_kwargs,
    ):
        tx = gate._get_transform(0, **transform_kwargs)

        x_all = tx.data_to_norm(df[gate.channels[0]].values)
        x = x_all[parent_mask]

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 3))

        hist_defaults = dict(bins=200, color="steelblue", alpha=0.7)
        hist_defaults.update(data_kwargs)
        ax.hist(x, **hist_defaults)

        linestyle = "--" if is_not else "-"
        ax.axvline(gate.lo, color=gate_color, linewidth=1.5, linestyle=linestyle)
        ax.axvline(gate.hi, color=gate_color, linewidth=1.5, linestyle=linestyle)

        label = f"NOT({gate.name})" if is_not else gate.name
        ax.set_xlabel(gate.channels[0])
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.set_xlim(0, 1)
        return ax
