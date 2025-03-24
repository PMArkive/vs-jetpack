from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vstools import ConstantFormatVideoNode, core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser

__all__ = [
    'SangNom', 'SangNomDR'
]


@dataclass
class SANGNOM(_Antialiaser):
    aa_strength: int | tuple[int, ...] = 48
    double_fps: bool = False

    # Class Variable
    _shift = -0.5

    def preprocess_clip(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        if self.double_fps:
            return clip.std.SeparateFields(self.field).std.DoubleWeave(self.field)
        return super().preprocess_clip(clip)

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(aa=self.aa_strength, order=0 if self.double_fps else self.field + 1)

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        interpolated = core.sangnom.SangNom(
            clip, dh=double_y or not self.drop_fields, **kwargs
        )

        return self.shift_interpolate(clip, interpolated, double_y, **kwargs)


class SangNomSS(SANGNOM, SuperSampler):
    _static_kernel_radius = 2


class SangNomSR(SANGNOM, SingleRater):
    ...


class SangNomDR(SANGNOM, DoubleRater):
    ...


class SangNom(SangNomSS, Antialiaser):
    ...
