"""
FCS file import utilities for batch loading into AnnData objects.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import FlowCal
import anndata
import numpy as np
import pandas as pd
from anndata import AnnData


def _import_fcs(path: str | Path) -> tuple[np.ndarray, list[str], float | None]:
    """Load a single FCS file and return its data, channels, and duration.

    Args:
        path: Path to the FCS file.

    Returns:
        Tuple of (data array [n_cells × n_channels], channel name list,
        acquisition duration in seconds or None).
    """
    fcs = FlowCal.transform.to_rfi(FlowCal.io.FCSData(str(path)))
    channels = list(fcs.channels)
    data = np.array(fcs, dtype=np.float32)

    duration = None
    try:
        btim = fcs.acquisition_start_time
        etim = fcs.acquisition_end_time
        duration = (etim - btim).seconds
    except Exception:
        pass

    return data, channels, duration


def import_fcs(
    path: str | Path,
    metadata: dict | None = None,
) -> AnnData:
    """Import a single FCS file into an AnnData object.

    Args:
        path: Path to the FCS file.
        metadata: Constant key→value pairs added as obs columns to every cell.

    Returns:
        AnnData where X is the cell × marker matrix, obs holds per-cell
        metadata, var holds channel names, and uns stores provenance info.
    """
    if metadata is None:
        metadata = {}

    path = Path(path)
    data, channels, duration = _import_fcs(path)
    n_cells = data.shape[0]

    obs = pd.DataFrame({'filename': [path.name] * n_cells})
    for key, val in metadata.items():
        obs[key] = val
    obs.index = obs.index.astype(str)

    return AnnData(
        X=data,
        obs=obs,
        var=pd.DataFrame(index=channels),
        uns={
            'folder': str(path.parent),
            'metadata': metadata,
            'n_files': 1,
            'acquisition_times': {path.name: duration},
        },
    )


def _import_series(
    folder: str | Path,
    layout: str | Path | pd.DataFrame,
    metadata: dict | None = None,
) -> AnnData:
    """Import all FCS files in one folder into a single AnnData object.

    Args:
        folder: Path to the directory containing FCS files.
        layout: DataFrame (or path to a CSV) with a 'filename' column and one
            column per condition (e.g. Subject, Time, Shear). Each row
            describes one FCS file.
        metadata: Constant key→value pairs added as obs columns to every cell
            (e.g. {'machine': 'A'}).

    Returns:
        AnnData where X is the cell × marker matrix, obs holds per-cell
        metadata, var holds channel names, and uns stores provenance info.
    """
    if metadata is None:
        metadata = {}

    folder = Path(folder)

    # Resolve layout
    if isinstance(layout, (str, Path)):
        layout = pd.read_csv(layout)
    else:
        layout = layout.copy()

    # Strip whitespace from all string columns
    layout.columns = layout.columns.str.strip()
    for col in layout.select_dtypes(include='object').columns:
        layout[col] = layout[col].str.strip()

    if 'filename' not in layout.columns:
        raise ValueError("layout must contain a 'filename' column")

    condition_cols = [c for c in layout.columns if c != 'filename']

    arrays = []
    obs_frames = []
    channel_names = None
    acq_times = {}

    for _, row in layout.iterrows():
        fpath = folder / row['filename']
        if not fpath.exists():
            warnings.warn(f"FCS file not found, skipping: {row['filename']}")
            continue

        data, channels, duration = _import_fcs(fpath)
        acq_times[row['filename']] = duration

        if channel_names is None:
            channel_names = channels
        elif channels != channel_names:
            raise ValueError(
                f"Channel mismatch in '{row['filename']}': "
                f"expected {channel_names}, got {channels}"
            )

        n_cells = data.shape[0]

        obs = pd.DataFrame(
            {col: [row[col]] * n_cells for col in condition_cols}
        )
        obs['filename'] = row['filename']
        for key, val in metadata.items():
            obs[key] = val

        arrays.append(data)
        obs_frames.append(obs)

    X = np.concatenate(arrays, axis=0)
    obs = pd.concat(obs_frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    var = pd.DataFrame(index=channel_names)

    return AnnData(
        X=X,
        obs=obs,
        var=var,
        uns={
            'folder': str(folder),
            'metadata': metadata,
            'n_files': len(layout),
            'acquisition_times': acq_times,
        },
    )


class FCSCollection:
    """Accumulate multiple FCS series and combine them into one AnnData.

    Example::

        col = (FCSCollection()
            .add_series('data/machineA/', layout_a, {'machine': 'A'})
            .add_series('data/machineB/', layout_b, {'machine': 'B'}))
        adata = col.combine()
    """

    def __init__(self) -> None:
        self._series: list[AnnData] = []

    def add_series(
        self,
        folder: str | Path,
        layout: str | Path | pd.DataFrame,
        metadata: dict | None = None,
    ) -> FCSCollection:
        """Import one folder of FCS files and store the result.

        Args:
            folder: Path to the directory containing FCS files.
            layout: DataFrame or CSV path with 'filename' + condition columns.
            metadata: Constant obs columns for all cells in this series
                (e.g. {'machine': 'A'}).

        Returns:
            self, to allow method chaining.
        """
        self._series.append(_import_series(folder, layout, metadata))
        return self

    def get(self, index: int | dict, **filters) -> AnnData:
        """Return a filtered view of one series.

        Args:
            index: Integer position, or a dict matching against uns['metadata']
                (e.g. {'machine': 'A'} to select the series where machine='A').
            **filters: Column=value pairs to filter obs by
                (e.g. donor='D1', agonist='unstim').

        Returns:
            Filtered AnnData view.

        Raises:
            KeyError: If no series matches the given metadata dict, or if
                multiple series match.
        """
        if isinstance(index, dict):
            matches = [
                s for s in self._series
                if all(s.uns.get('metadata', {}).get(k) == v for k, v in index.items())
            ]
            if len(matches) == 0:
                raise KeyError(f"No series found with metadata matching {index}")
            if len(matches) > 1:
                raise KeyError(f"Multiple series match metadata {index}; be more specific")
                
            adata = matches[0]
        else:
            adata = self._series[index]
        mask = pd.Series([True] * adata.n_obs, index=adata.obs.index)
        for col, val in filters.items():
            mask &= adata.obs[col] == val
        return adata[mask]

    def combine(self) -> AnnData:
        """Concatenate all stored series into a single AnnData object.

        Returns:
            Combined AnnData with unique obs names.

        Raises:
            RuntimeError: If no series have been added yet.
        """
        if not self._series:
            raise RuntimeError("No series have been added. Call add_series() first.")

        combined = anndata.concat(self._series)
        combined.obs_names_make_unique()
        return combined
