from __future__ import annotations

from typing import Any, Sequence, TypeVar, cast

from vstools import Nb, PlanesT, VSFunction, join, normalize_planes, normalize_seq, plane, to_arr, vs, KwargsT

from .enum import RemoveGrainMode, RepairMode

__all__ = [
    'wmean_matrix', 'mean_matrix',
    'norm_rmode_planes',
    'normalize_radius'
]

wmean_matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
mean_matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1]

RModeT = TypeVar('RModeT', RemoveGrainMode, RepairMode)


def norm_rmode_planes(
    clip: vs.VideoNode, mode: int | RModeT | Sequence[int | RModeT], planes: PlanesT = None
) -> list[int]:
    assert clip.format

    modes_array = normalize_seq(to_arr(mode), clip.format.num_planes)  # type: ignore[var-annotated,arg-type]

    planes = normalize_planes(clip, planes)

    return [
        cast(RModeT, rep if i in planes else 0) for i, rep in enumerate(modes_array, 0)
    ]


def normalize_radius(
    clip: vs.VideoNode, func: VSFunction, radius: list[Nb] | tuple[str, list[Nb]], planes: list[int], **kwargs: Any
) -> vs.VideoNode:
    name, radius = radius if isinstance(radius, tuple) else ('radius', radius)

    radius = normalize_seq(radius, clip.format.num_planes)

    def _get_kwargs(rad: Nb) -> KwargsT:
        return kwargs | {name: rad, 'planes': planes}

    if len(set(radius)) > 0:
        if len(planes) != 1:
            return join([
                func(plane(clip, i), **_get_kwargs(rad)) for i, rad in enumerate(radius)
            ])

        radius = radius[planes[0]]
    else:
        radius = radius[0]

    return func(clip, **_get_kwargs(radius))
