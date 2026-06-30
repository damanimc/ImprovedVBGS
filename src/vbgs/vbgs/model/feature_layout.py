"""Feature layout helpers for VBGS point / component vectors."""

from __future__ import annotations

N_SPATIAL = 3
N_COLOR = 3


def n_features(n_semantic: int) -> int:
    return N_SPATIAL + N_COLOR + int(n_semantic)


def split_features(d, n_semantic: int = 0):
    """Split (..., D, 1) or (..., D) last feature axis into spatial/color/semantic."""
    if d.ndim >= 1 and d.shape[-1] == 1:
        feat = d[..., 0]
        squeeze = True
    else:
        feat = d
        squeeze = False

    ds = feat[..., :N_SPATIAL]
    dc = feat[..., N_SPATIAL : N_SPATIAL + N_COLOR]
    if n_semantic > 0:
        dsem = feat[..., N_SPATIAL + N_COLOR :]
    else:
        dsem = None

    if squeeze:
        ds = ds[..., None]
        dc = dc[..., None]
        if dsem is not None:
            dsem = dsem[..., None]
    return ds, dc, dsem


def model_n_semantic(model) -> int:
    semantic = getattr(model, "semantic_delta", None)
    if semantic is None:
        return 0
    return int(semantic.event_shape[0])
