from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from vsexprtools import ExprOp, combine, complexpr_available, expr_func, norm_expr
from vskernels import Catrom, Hermite, LinearScaler, Scaler, ScalerT
from vsmasktools import ringing_mask
from vsrgtools import RepairMode, box_blur, gauss_blur, repair
from vstools import (
    ConstantFormatVideoNode, CustomOverflowError, PlanesT, VSFunctionNoArgs,
    check_ref_clip, check_variable, check_variable_format, core, inject_self, scale_delta, vs
)

from .generic import BaseGenericScaler, GenericScaler

__all__ = [
    'ClampScaler',
    'DPID',
    'SSIM'
]


@dataclass
class ClampScaler(GenericScaler):
    """Clamp a reference Scaler."""

    base_scaler: ScalerT
    """Scaler to clamp."""

    reference: ScalerT | vs.VideoNode
    """Reference Scaler used to clamp base_scaler"""

    strength: int = 80
    """Strength of clamping."""

    overshoot: float | None = None
    """Overshoot threshold."""

    undershoot: float | None = None
    """Undershoot threshold."""

    limit: RepairMode | bool = True
    """Whether to use under/overshoot limit (True) or a reference repaired clip for limiting."""

    operator: Literal[ExprOp.MAX, ExprOp.MIN] | None = ExprOp.MIN
    """Whether to take the brightest or darkest pixels in the merge."""

    masked: bool = True
    """Whether to mask with a ringing mask or not."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.strength >= 100:
            raise CustomOverflowError('strength can\'t be more or equal to 100!', self.__class__)
        elif self.strength <= 0:
            raise CustomOverflowError('strength can\'t be less or equal to 0!', self.__class__)

        if self.overshoot is None:
            self.overshoot = self.strength / 100
        if self.undershoot is None:
            self.undershoot = self.overshoot

        self._base_scaler = self.ensure_scaler(self.base_scaler)
        self._reference = None if isinstance(self.reference, vs.VideoNode) else self.ensure_scaler(self.reference)

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[float, float] = (0, 0), *, smooth: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        width, height = self._wh_norm(clip, width, height)

        assert (self.undershoot or self.undershoot == 0) and (self.overshoot or self.overshoot == 0)

        ref = self._base_scaler.scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference

            if shift != (0, 0):
                smooth = self._kernel.shift(smooth, shift)
        else:
            assert self._reference
            smooth = self._reference.scale(clip, width, height, shift)

        assert smooth

        check_ref_clip(ref, smooth)

        merge_weight = self.strength / 100

        if self.limit is True:
            expression = 'x {merge_weight} * y {ref_weight} * + a {undershoot} - z {overshoot} + clip'

            merged = norm_expr(
                [ref, smooth, smooth.std.Maximum(), smooth.std.Minimum()],
                expression, merge_weight=merge_weight, ref_weight=1.0 - merge_weight,
                undershoot=scale_delta(self.undershoot, 32, clip),
                overshoot=scale_delta(self.overshoot, 32, clip),
                func=self.__class__
            )
        else:
            merged = smooth.std.Merge(ref, merge_weight)

            if isinstance(self.limit, RepairMode):
                merged = repair(merged, smooth, self.limit)

        if self.operator is not None:
            merge2 = combine([smooth, ref], self.operator)

            if self.masked:
                merged = merged.std.MaskedMerge(merge2, ringing_mask(smooth))
            else:
                merged = merge2
        elif self.masked:
            merged.std.MaskedMerge(smooth, ringing_mask(smooth))

        return merged

    @property
    def kernel_radius(self) -> int:  # type: ignore[override]
        if self._reference:
            return max(self._reference.kernel_radius, self._base_scaler.kernel_radius)
        return self._base_scaler.kernel_radius


class DPID(BaseGenericScaler):
    """Rapid, Detail-Preserving Image Downscaler for VapourSynth"""

    def __init__(
        self,
        sigma: float = 0.1,
        ref: vs.VideoNode | ScalerT = Catrom,
        planes: PlanesT = None,
        **kwargs: Any
    ) -> None:
        """
        :param sigma:       The power factor of range kernel. It can be used to tune the amplification
                            of the weights of pixels that represent detail—from a box filter over an emphasis
                            of distinct pixels towards a selection of only the most distinct pixels.
        :param ref:         VideoNode or Scaler to obtain the downscaled reference for DPID.
        :param planes:      Sets which planes will be processed. Any unprocessed planes will be simply copied from ref.
        """
        super().__init__(**kwargs)

        self.sigma = sigma
        self.ref = ref
        self.planes = planes

        if isinstance(ref, vs.VideoNode):
            self._ref_scaler = self.scaler
        else:
            self._ref_scaler = Scaler.ensure_obj(ref, self.__class__)

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        ref = clip

        if isinstance(self.ref, vs.VideoNode):
            check_ref_clip(clip, self.ref)

            if TYPE_CHECKING:
                assert check_variable_format(self.ref, self.__class__)

            ref = self.ref

        if (ref.width, ref.height) != (width, height):
            ref = self._ref_scaler.scale(ref, width, height)  # type: ignore[assignment]

        kwargs = {
            'lambda': self.sigma, 'planes': self.planes,
            'src_left': shift[1], 'src_top': shift[0]
        } | self.kwargs | kwargs | {'read_chromaloc': True}

        return core.dpid.DpidRaw(clip, ref, **kwargs)

    @inject_self.cached.property
    def kernel_radius(self) -> int:
        return self._ref_scaler.kernel_radius


class SSIM(LinearScaler):
    """
    SSIM downsampler is an image downscaling technique that aims to optimize
    for the perceptual quality of the downscaled results.

    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured
    using famous Structural SIMilarity (SSIM) index.

    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.
    """

    def __init__(
        self,
        scaler: ScalerT = Hermite,
        smooth: int | float | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param scaler:      Scaler to be used for downscaling, defaults to Hermite.
        :param smooth:      Image smoothening method.
                            If you pass an int, it specifies the "radius" of the internally-used boxfilter,
                            i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
                            If you pass a float, it specifies the "sigma" of gauss_blur,
                            i.e. the standard deviation of gaussian blur.
                            If you pass a function, it acts as a general smoother.
                            Default uses a gaussian blur based on the scaler's kernel radius.
        """
        super().__init__(**kwargs)

        self.scaler = Hermite.from_param(scaler)

        if smooth is None:
            smooth = (self.scaler.kernel_radius + 1.0) / 3

        if callable(smooth):
            self.filter_func = smooth
        elif isinstance(smooth, int):
            self.filter_func = partial(box_blur, radius=smooth)
        elif isinstance(smooth, float):
            self.filter_func = partial(gauss_blur, sigma=smooth)

    def _linear_scale(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        assert check_variable(clip, self.scale)

        l1 = self.scaler.scale(clip, width, height, shift, **(kwargs | self.kwargs))

        l1_sq, c_sq = [expr_func(x, 'x dup *') for x in (l1, clip)]

        l2 = self.scaler.scale(c_sq, width, height, shift, **(kwargs | self.kwargs))

        m, sl_m_square, sh_m_square = [self.filter_func(x) for x in (l1, l1_sq, l2)]

        if complexpr_available:
            merge_expr = f'z dup * SQ! x SQ@ - SQD! SQD@ {1e-6} < 0 y SQ@ - SQD@ / sqrt ?'
        else:
            merge_expr = f'x z dup * - {1e-6} < 0 y z dup * - x z dup * - / sqrt ?'

        r = expr_func([sl_m_square, sh_m_square, m], merge_expr)

        t = expr_func([r, m], 'x y *')

        return expr_func([self.filter_func(m), self.filter_func(r), l1, self.filter_func(t)], 'x y z * + a -')

    @inject_self.cached.property
    def kernel_radius(self) -> int:
        return self.scaler.kernel_radius
