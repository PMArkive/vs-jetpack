from __future__ import annotations

from typing import Any, Sequence

from jetpytools import normalize_seq

from vstools import (
    GenericVSFunction,
    Planes,
    check_variable_format,
    join,
    normalize_planes,
    split,
    vs,
)

__all__ = ["normalize_radius"]


def normalize_radius(
    clip: vs.VideoNode,
    function: GenericVSFunction,
    radius: Sequence[float | int] | dict[str, Sequence[float | int]],
    planes: Planes,
    **kwargs: Any,
) -> vs.VideoNode:
    assert check_variable_format(clip, normalize_radius)

    if isinstance(radius, dict):
        name, radius = radius.popitem()
    else:
        name, radius = "radius", radius

    radius = normalize_seq(radius, clip.format.num_planes)
    planes = normalize_planes(clip, planes)

    if len(set(radius)) > 1:
        pplanes = [
            function(p, **kwargs | {name: rad, "planes": 0}) if i in planes else p
            for i, (rad, p) in enumerate(zip(radius, split(clip)))
        ]
        return join(pplanes)

    return function(clip, **kwargs | {name: radius[0], "planes": planes})
