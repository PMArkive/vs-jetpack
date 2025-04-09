from functools import partial
from math import factorial
from typing import Literal, MutableMapping, Protocol

from numpy import linalg, zeros
from typing_extensions import Self

from vsaa import Nnedi3
from vsaa.abstract import _Antialiaser
from vsdeband import AddNoise
from vsdenoise import (
    DFTTest, MaskMode, MotionVectors, MVDirection, MVTools, MVToolsPreset, MVToolsPresets, prefilter_to_full_range
)
from vsexprtools import norm_expr
from vsmasktools import Coordinates, Morpho
from vsrgtools import (
    BlurMatrix, MeanMode, RemoveGrainMode, RepairMode, gauss_blur, median_blur, remove_grain, repair, unsharpen
)
from vstools import (
    ConstantFormatVideoNode, ConvMode, CustomRuntimeError, FieldBased, FieldBasedT, KwargsT, VSFunctionKwArgs,
    check_variable, core, fallback, scale_delta, vs, vs_object
)

from .enums import (
    BackBlendMode, InputType, LosslessMode, NoiseDeintMode, NoiseProcessMode, SearchPostProcess, SharpLimitMode,
    SharpMode, SourceMatchMode
)
from .utils import reinterlace, scdetect

__all__ = [
    'QTempGaussMC'
]


class _DenoiseFuncTr(Protocol):
    def __call__(self, clip: vs.VideoNode, /, *, tr: int = ...) -> vs.VideoNode:
        ...


class QTempGaussMC(vs_object):
    """
    Quasi Temporal Gaussian Motion Compensated (QTGMC)
    """

    clip: ConstantFormatVideoNode
    """Clip to process."""

    draft: ConstantFormatVideoNode
    """Draft processed clip, used as a base for prefiltering & denoising."""

    bobbed: ConstantFormatVideoNode
    """High quality bobbed clip, initial spatial interpolation."""

    noise: ConstantFormatVideoNode | None
    """Extracted noise when noise processing is enabled."""

    prefilter_output: ConstantFormatVideoNode
    """Output of the prefilter stage."""

    denoise_output: ConstantFormatVideoNode
    """Output of the denoise stage."""

    basic_output: ConstantFormatVideoNode
    """Output of the basic stage."""

    final_output: ConstantFormatVideoNode
    """Output of the final stage."""

    motion_blur_output: ConstantFormatVideoNode
    """Output of the motion blur stage."""

    def __init__(
        self,
        clip: vs.VideoNode,
        input_type: InputType = InputType.INTERLACE,
        tff: FieldBasedT | bool | None = None,
    ) -> None:
        """
        :param clip:          Clip to process.
        :param input_type:    Nature of the clip - indicates processing routine.
        :param tff:           Field order of the clip.
        """

        assert check_variable(clip, self.__class__)

        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        self.clip = clip
        self.input_type = input_type
        self.tff = clip_fieldbased.is_tff
        self.field = -1 if not clip_fieldbased.is_inter else clip_fieldbased.field + 2

        if self.input_type == InputType.PROGRESSIVE and clip_fieldbased.is_inter:
            raise CustomRuntimeError(f'{self.input_type} incompatible with interlaced video!', self.__class__)

    def prefilter(
        self,
        *,
        tr: int = 2,
        sc_threshold: float | None | Literal[False] = None,
        postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
        strength: tuple[float, float] = (1.9, 0.1),
        limit: tuple[int | float, int | float, int | float] = (3, 7, 2),
        range_conversion_args: KwargsT | None | Literal[False] = KwargsT(range_conversion=2.0),
        mask_shimmer_args: KwargsT | None = None,
    ) -> Self:
        """
        :param tr:                       Radius of the initial temporal binomial smooth.
        :param sc_threshold:             Threshold for scene changes, disables sc detection if False.
        :param postprocess:              Post-processing routine to use.
        :param strength:                 Tuple containing gaussian blur sigma & blend weight of the blur.
        :param limit:                    3-step limiting thresholds for the gaussian blur post-processing.
        :param range_conversion_args:    Arguments passed to :py:attr:`prefilter_to_full_range`.
        :param mask_shimmer_args:        Arguments passed to :py:attr:`QTempGaussMC.mask_shimmer`.
        """

        self.prefilter_tr = tr
        self.prefilter_sc_threshold = sc_threshold
        self.prefilter_postprocess = postprocess
        self.prefilter_blur_strength = strength
        self.prefilter_soften_limit = limit
        self.prefilter_range_conversion_args = fallback(range_conversion_args, KwargsT())
        self.prefilter_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())

        return self

    def denoise(
        self,
        *,
        tr: int = 2,
        func: _DenoiseFuncTr | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] = partial(DFTTest.denoise, sigma=2),
        mode: NoiseProcessMode = NoiseProcessMode.IDENTIFY,
        deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
        stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
        func_comp_args: KwargsT | None = None,
        stabilize_comp_args: KwargsT | None = None,
    ) -> Self:
        """
        :param tr:                     Temporal radius of the denoising function & it's motion compensation.
        :param func:                   Denoising function to use.
        :param mode:                   Noise handling method to use.
        :param deint:                  Noise deinterlacing method to use.
        :param stabilize:              Weights to use when blending source noise with compensated noise.
        :param func_comp_args:         Arguments passed to :py:attr:`MVTools.compensate` for denoising.
        :param stabilize_comp_args:    Arguments passed to :py:attr:`MVTools.compensate` for stabilization.
        """

        self.denoise_tr = tr
        self.denoise_func = func
        self.denoise_mode = mode
        self.denoise_deint = deint
        self.denoise_stabilize: tuple[float, float] | Literal[False] = stabilize
        self.denoise_func_comp_args = fallback(func_comp_args, KwargsT())
        self.denoise_stabilize_comp_args = fallback(stabilize_comp_args, KwargsT())

        return self

    def basic(
        self,
        *,
        tr: int = 2,
        thsad: int | tuple[int, int] = 640,
        bobber: _Antialiaser = Nnedi3(qual=2, nsize=0, nns=4, pscrn=1),
        noise_restore: float = 0,
        degrain_args: KwargsT | None = None,
        mask_args: KwargsT | None | Literal[False] = None,
        mask_shimmer_args: KwargsT | None = KwargsT(erosion_distance=0),
    ) -> Self:
        """
        :param tr:                   Temporal radius of the motion compensated binomial smooth.
        :param thsad:                Thsad of the motion compensated binomial smooth.
        :param bobber:               Bobber to use for initial spatial interpolation.
        :param noise_restore:        How much noise to restore after this stage.
        :param degrain_args:         Arguments passed to :py:attr:`QTempGaussMC.binomial_degrain`.
        :param mask_args:            Arguments passed to :py:attr:`MVTools.mask` for :py:attr:`InputType.REPAIR`.
        :param mask_shimmer_args:    Arguments passed to :py:attr:`QTempGaussMC.mask_shimmer`.
        """

        self.basic_tr = tr
        self.basic_thsad = thsad if isinstance(thsad, tuple) else (thsad, thsad)
        self.basic_bobber = bobber.copy(field=self.field)
        self.basic_noise_restore = noise_restore
        self.basic_degrain_args = fallback(degrain_args, KwargsT())
        self.basic_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())
        self.basic_mask_args: KwargsT | Literal[False] = fallback(mask_args, KwargsT())

        return self

    def source_match(
        self,
        *,
        tr: int = 1,
        bobber: _Antialiaser | None = None,
        mode: SourceMatchMode = SourceMatchMode.NONE,
        similarity: float = 0.5,
        enhance: float = 0.5,
        degrain_args: KwargsT | None = None,
    ) -> Self:
        """
        :param tr:              Temporal radius of the refinement motion compensated binomial smooth.
        :param bobber:          Bobber to use for refined spatial interpolation.
        :param mode:            Specifies number of refinement steps to perform.
        :param similarity:      Temporal similarity of the error created by smoothing.
        :param enhance:         Sharpening strength prior to source match refinement.
        :param degrain_args:    Arguments passed to :py:attr:`QTempGaussMC.binomial_degrain`.
        """

        self.match_tr = tr
        self.match_bobber = fallback(bobber, self.basic_bobber).copy(field=self.field)
        self.match_mode = mode
        self.match_similarity = similarity
        self.match_enhance = enhance
        self.match_degrain_args = fallback(degrain_args, KwargsT())

        return self

    def lossless(
        self,
        *,
        mode: LosslessMode = LosslessMode.NONE,
    ) -> Self:
        """
        :param mode:    Specifies at which stage to re-weave the original fields.
        """

        self.lossless_mode = mode

        return self

    def sharpen(
        self,
        *,
        mode: SharpMode | None = None,
        strength: float = 1.0,
        clamp: int | float = 1,
        thin: float = 0.0,
    ) -> Self:
        """
        :param mode:        Specifies the type of sharpening to use.
        :param strength:    Sharpening strength.
        :param clamp:       Clamp the sharpening strength of :py:attr:`SharpMode.UNSHARP_MINMAX` to the min/max average plus this.
        :param thin:        How much to vertically thin edges.
        """

        if mode is None:
            self.sharp_mode = SharpMode.NONE if self.match_mode else SharpMode.UNSHARP_MINMAX
        else:
            self.sharp_mode = mode

        self.sharp_strength = strength
        self.sharp_clamp = clamp
        self.sharp_thin = thin

        return self

    def back_blend(
        self,
        *,
        mode: BackBlendMode = BackBlendMode.BOTH,
        sigma: float = 1.4,
    ) -> Self:
        """
        :param mode:     Specifies at which stage to perform back-blending.
        :param sigma:    Gaussian blur sigma.
        """

        self.backblend_mode = mode
        self.backblend_sigma = sigma

        return self

    def sharpen_limit(
        self,
        *,
        mode: SharpLimitMode | None = None,
        radius: int = 3,
        overshoot: int | float = 0,
        comp_args: KwargsT | None = None,
    ) -> Self:
        """
        :param mode:         Specifies type of limiting & at which stage to perform it.
        :param radius:       Radius of sharpness limiting.
        :param overshoot:    How much overshoot to allow.
        :param comp_args:    Arguments passed to :py:attr:`MVTools.compensate` for temporal limiting.
        """

        if mode is None:
            self.limit_mode = SharpLimitMode.NONE if self.match_mode else SharpLimitMode.TEMPORAL_PRESMOOTH
        else:
            self.limit_mode = mode

        self.limit_radius = radius
        self.limit_overshoot = overshoot
        self.limit_comp_args = fallback(comp_args, KwargsT())

        return self

    def final(
        self,
        *,
        tr: int = 3,
        thsad: int | tuple[int, int] = 256,
        noise_restore: float = 0.0,
        degrain_args: KwargsT | None = None,
        mask_shimmer_args: KwargsT | None = None,
    ) -> Self:
        """
        :param tr:                   Temporal radius of the motion compensated smooth.
        :param thsad:                Thsad of the motion compensated smooth.
        :param noise_restore:        How much noise to restore after this stage.
        :param degrain_args:         Arguments passed to :py:attr:`MVTools.degrain`.
        :param mask_shimmer_args:    Arguments passed to :py:attr:`QTempGaussMC.mask_shimmer`.
        """

        self.final_tr = tr
        self.final_thsad = thsad if isinstance(thsad, tuple) else (thsad, thsad)
        self.final_noise_restore = noise_restore
        self.final_degrain_args = fallback(degrain_args, KwargsT())
        self.final_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())

        return self

    def motion_blur(
        self,
        *,
        shutter_angle: tuple[int | float, int | float] = (180, 180),
        fps_divisor: int = 1,
        blur_args: KwargsT | None = None,
        mask_args: KwargsT | None | Literal[False] = KwargsT(ml=4),
    ) -> Self:
        """
        :param shutter_angle:    Tuple containing the source and output shutter angle. Will apply motion blur if they do not match.
        :param fps_divisor:      Factor by which to reduce framerate.
        :param blur_args:        Arguments passed to :py:attr:`MVTools.flow_blur`.
        :param mask_args:        Arguments passed to :py:attr:`MVTools.mask`.
        """

        self.motion_blur_shutter_angle = shutter_angle
        self.motion_blur_fps_divisor = fps_divisor
        self.motion_blur_args = fallback(blur_args, KwargsT())
        self.motion_blur_mask_args: KwargsT | Literal[False] = fallback(mask_args, KwargsT())

        return self

    def mask_shimmer(
        self,
        flt: vs.VideoNode,
        src: vs.VideoNode,
        threshold: float | int = 1,
        erosion_distance: int = 4,
        over_dilation: int = 0,
    ) -> ConstantFormatVideoNode:
        """
        :param flt:                 Processed clip to perform masking on.
        :param src:                 Unprocessed clip to restore from.
        :param threshold:           Threshold of change to perform masking.
        :param erosion_distance:    How much to deflate then reflate to remove thin areas.
        :param over_dilation:       Extra inflation to ensure areas to restore back are fully caught.
        """

        assert check_variable(flt, self.mask_shimmer)

        if not erosion_distance:
            return flt

        iter1 = 1 + (erosion_distance + 1) // 3
        iter2 = 1 + (erosion_distance + 2) // 3

        over1 = over_dilation // 3
        over2 = over_dilation % 3

        diff = src.std.MakeDiff(flt)

        opening = Morpho.minimum(diff, iterations=iter1, coords=Coordinates.VERTICAL)
        closing = Morpho.maximum(diff, iterations=iter1, coords=Coordinates.VERTICAL)

        if erosion_distance % 3:
            opening = Morpho.deflate(opening)
            closing = Morpho.inflate(closing)

            if erosion_distance % 3 == 2:
                opening = median_blur(opening)
                closing = median_blur(closing)

        opening = Morpho.maximum(opening, iterations=iter2, coords=Coordinates.VERTICAL)
        closing = Morpho.minimum(closing, iterations=iter2, coords=Coordinates.VERTICAL)

        if over_dilation:
            opening = Morpho.maximum(opening, iterations=over1)
            closing = Morpho.minimum(closing, iterations=over1)

            opening = Morpho.inflate(opening, iterations=over2)
            closing = Morpho.deflate(closing, iterations=over2)

        return norm_expr(
            [flt, diff, opening, closing],
            'y neutral - abs {thr} > y a neutral min z neutral max clip y ? neutral - x +',
            thr=scale_delta(threshold, 8, flt)
        )

    def binomial_degrain(self, clip: vs.VideoNode, tr: int) -> ConstantFormatVideoNode:
        def _get_weights(n: int) -> list[int]:
            k, rhs = 1, []
            mat = zeros((n + 1, n + 1))

            for i in range(1, n + 2):
                mat[n + 1 - i, i - 1] = mat[n, i - 1] = 1 / 3
                rhs.append(k)
                k = k * (2 * n + 1 - i) // i

            mat[n, 0] = 1

            return list(linalg.solve(mat, rhs))

        assert check_variable(clip, self.binomial_degrain)

        if not tr:
            return clip

        backward, forward = self.mv.get_vectors(tr=tr)
        vectors = MotionVectors()
        degrained = list[ConstantFormatVideoNode]()

        for delta in range(tr):
            vectors.set_vector(backward[delta], MVDirection.BACKWARD, 1)
            vectors.set_vector(forward[delta], MVDirection.FORWARD, 1)
            vectors.tr = 1

            degrained.append(
                self.mv.degrain(  # type: ignore
                    clip, vectors=vectors, thsad=self.basic_thsad, thscd=self.thscd, **self.basic_degrain_args
                )
            )
            vectors.clear()

        return core.std.AverageFrames([clip, *degrained], _get_weights(tr))

    def apply_prefilter(self) -> None:
        if self.input_type == InputType.REPAIR:
            search = BlurMatrix.BINOMIAL()(self.draft, mode=ConvMode.VERTICAL)
        else:
            search = self.draft

        if self.prefilter_tr:
            scenechange = self.prefilter_sc_threshold is not False

            scenes = scdetect(search, self.prefilter_sc_threshold) if scenechange else search
            smoothed = BlurMatrix.BINOMIAL(self.prefilter_tr, mode=ConvMode.TEMPORAL, scenechange=scenechange)(scenes)
            smoothed = self.mask_shimmer(smoothed, search, **self.prefilter_mask_shimmer_args)
        else:
            smoothed = search

        if self.prefilter_postprocess:
            gauss_sigma, blend_weight = self.prefilter_blur_strength

            blurred = core.std.Merge(gauss_blur(smoothed, gauss_sigma), smoothed, blend_weight)

            if self.prefilter_postprocess == SearchPostProcess.GAUSSBLUR_EDGESOFTEN:
                lim1, lim2, lim3 = [scale_delta(thr, 8, self.clip) for thr in self.prefilter_soften_limit]

                blurred = norm_expr(
                    [blurred, smoothed, search],
                    'z y {lim1} - y {lim1} + clip TWEAK! '
                    'x {lim2} + TWEAK@ < x {lim3} + x {lim2} - TWEAK@ > x {lim3} - x 51 * TWEAK@ 49 * + 100 / ? ?',
                    lim1=lim1, lim2=lim2, lim3=lim3,
                )
        else:
            blurred = smoothed

        if self.prefilter_range_conversion_args is not False:
            blurred = prefilter_to_full_range(blurred, **self.prefilter_range_conversion_args)  # type: ignore

        self.prefilter_output = blurred

    def apply_denoise(self) -> None:
        if not self.denoise_mode:
            self.noise = None
            self.denoise_output = self.clip
        else:
            if self.denoise_tr:
                denoised = self.mv.compensate(
                    self.draft, tr=self.denoise_tr, thscd=self.thscd,
                    temporal_func=lambda clip: self.denoise_func(clip, tr=self.denoise_tr),
                    **self.denoise_func_comp_args,
                )
            else:
                denoised = self.denoise_func(self.draft)

            if self.input_type == InputType.INTERLACE:
                denoised = reinterlace(denoised, self.tff)

            noise = self.clip.std.MakeDiff(denoised)

            if self.basic_noise_restore or self.final_noise_restore:
                if self.input_type == InputType.INTERLACE:
                    match self.denoise_deint:
                        case NoiseDeintMode.WEAVE:
                            noise = core.std.Interleave([noise] * 2)
                        case NoiseDeintMode.BOB:
                            noise = noise.resize.Bob(tff=self.tff)
                        case NoiseDeintMode.GENERATE:
                            noise_source = noise.std.SeparateFields(self.tff)

                            noise_max = Morpho.maximum(Morpho.maximum(noise_source), coords=Coordinates.HORIZONTAL)
                            noise_min = Morpho.minimum(Morpho.minimum(noise_source), coords=Coordinates.HORIZONTAL)

                            noise_new = AddNoise.GAUSS.grain(
                                noise_source, 2048, protect_chroma=False, fade_limits=False, neutral_out=True
                            )
                            noise_limit = norm_expr([noise_max, noise_min, noise_new], 'x y - z * range_size / y +')

                            noise = core.std.Interleave([noise_source, noise_limit]).std.DoubleWeave(self.tff)

                if self.denoise_stabilize:
                    weight1, weight2 = self.denoise_stabilize

                    noise_comp, _ = self.mv.compensate(
                        noise, direction=MVDirection.BACKWARD,
                        tr=1, thscd=self.thscd, interleave=False,
                        **self.denoise_stabilize_comp_args,
                    )

                    noise = norm_expr(
                        [noise, *noise_comp],
                        'x neutral - abs y neutral - abs > x y ? {weight1} * x y + {weight2} * +',
                        weight1=weight1, weight2=weight2,
                    )  # type: ignore

            self.noise = noise
            self.denoise_output = denoised if self.denoise_mode == NoiseProcessMode.DENOISE else self.clip  # type: ignore
        
        if self.input_type == InputType.REPAIR:
            self.denoise_output = reinterlace(self.denoise_output, self.tff)  # type: ignore

    def apply_basic(self) -> None:
        if self.input_type == InputType.PROGRESSIVE:
            self.bobbed = self.denoise_output
        else:
            self.bobbed = self.basic_bobber.interpolate(  # type: ignore
                self.denoise_output, False, **self.basic_bobber.get_aa_args(self.denoise_output)
            )

        if self.basic_mask_args is not False and self.input_type == InputType.REPAIR:
            mask = self.mv.mask(
                self.prefilter_output, direction=MVDirection.BACKWARD,
                kind=MaskMode.SAD, thscd=self.thscd, **self.basic_mask_args,
            )
            self.bobbed = self.denoise_output.std.MaskedMerge(self.bobbed, mask)

        smoothed = self.binomial_degrain(self.bobbed, self.basic_tr)
        smoothed = self.mask_shimmer(smoothed, self.bobbed, **self.basic_mask_shimmer_args)

        if self.match_mode:
            smoothed = self.apply_source_match(smoothed)

        if self.lossless_mode == LosslessMode.PRESHARPEN and self.input_type != InputType.PROGRESSIVE:
            smoothed = self.apply_lossless(smoothed)

        resharp = self.apply_sharpen(smoothed)

        if self.backblend_mode in (BackBlendMode.PRELIMIT, BackBlendMode.BOTH):
            resharp = self.apply_back_blend(resharp, smoothed)

        if self.limit_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.TEMPORAL_PRESMOOTH):
            resharp = self.apply_sharpen_limit(resharp)

        if self.backblend_mode in (BackBlendMode.POSTLIMIT, BackBlendMode.BOTH):
            resharp = self.apply_back_blend(resharp, smoothed)

        self.basic_output = self.apply_noise_restore(resharp, self.basic_noise_restore)

    def apply_source_match(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        def _error_adjustment(clip: vs.VideoNode, ref: vs.VideoNode, tr: int) -> ConstantFormatVideoNode:
            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2**tr_f / (binomial_coeff + self.match_similarity * (2**tr_f - binomial_coeff))

            return norm_expr([clip, ref], 'y {adj} 1 + * x {adj} * -', adj=error_adj)  # type: ignore

        if self.input_type != InputType.PROGRESSIVE:
            clip = reinterlace(clip, self.tff)

        adjusted1 = _error_adjustment(clip, self.denoise_output, self.basic_tr)
        if self.input_type == InputType.PROGRESSIVE:
            bobbed1 = adjusted1
        else:
            bobbed1 = self.basic_bobber.interpolate(adjusted1, False, **self.basic_bobber.get_aa_args(adjusted1))  # type: ignore
        match1 = self.binomial_degrain(bobbed1, self.basic_tr)

        if self.match_mode > SourceMatchMode.BASIC:
            if self.match_enhance:
                match1 = unsharpen(match1, self.match_enhance, BlurMatrix.BINOMIAL())

            if self.input_type != InputType.PROGRESSIVE:
                clip = reinterlace(match1, self.tff)

            diff = self.denoise_output.std.MakeDiff(clip)
            if self.input_type == InputType.PROGRESSIVE:
                bobbed2 = diff
            else:
                bobbed2 = self.match_bobber.interpolate(diff, False, **self.match_bobber.get_aa_args(diff))  # type: ignore
            match2 = self.binomial_degrain(bobbed2, self.match_tr)

            if self.match_mode == SourceMatchMode.TWICE_REFINED:
                adjusted2 = _error_adjustment(match2, bobbed2, self.match_tr)
                match2 = self.binomial_degrain(adjusted2, self.match_tr)

            out = match1.std.MergeDiff(match2)
        else:
            out = match1

        return out

    def apply_lossless(self, flt: vs.VideoNode) -> ConstantFormatVideoNode:
        def _reweave(clipa: vs.VideoNode, clipb: vs.VideoNode) -> ConstantFormatVideoNode:
            return core.std.Interleave([clipa, clipb]).std.SelectEvery(4, (0, 1, 3, 2)).std.DoubleWeave(self.tff)[::2]

        fields_src = self.denoise_output.std.SeparateFields(self.tff)

        if self.input_type == InputType.REPAIR:
            fields_src = fields_src.std.SelectEvery(4, (0, 3))  # type: ignore

        fields_flt = flt.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = _reweave(fields_src, fields_flt)

        median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL))
        fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        processed_diff = norm_expr(
            [fields_diff, median_blur(fields_diff, mode=ConvMode.VERTICAL)],
            'x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?',
        )
        processed_diff = repair(
            processed_diff, remove_grain(processed_diff, RemoveGrainMode.MINMAX_AROUND2), RepairMode.MINMAX_SQUARE1
        )

        return _reweave(fields_src, core.std.MakeDiff(fields_flt, processed_diff))

    def apply_sharpen(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.apply_sharpen)

        blur_kernel = BlurMatrix.BINOMIAL()

        match self.sharp_mode:
            case SharpMode.NONE:
                resharp = clip
            case SharpMode.UNSHARP:
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel)
            case SharpMode.UNSHARP_MINMAX:
                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL)

                clamp = norm_expr(
                    [clip, source_min, source_max],
                    'y z + 2 / AVG! x AVG@ {thr} - AVG@ {thr} + clip',
                    thr=scale_delta(self.sharp_clamp, 8, clip),
                )
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel(clamp))

        if self.sharp_thin:
            median_diff = norm_expr(
                [clip, median_blur(clip, mode=ConvMode.VERTICAL)], 'y x - {thin} * neutral +', thin=self.sharp_thin
            )
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff)

            resharp = norm_expr(
                [resharp, blurred_diff, blur_kernel(blurred_diff)],
                'y neutral - Y! z neutral - Z! Y@ abs Z@ abs < Y@ 0 ? x +',
            )

        return resharp

    def apply_back_blend(self, flt: vs.VideoNode, src: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(flt, self.apply_back_blend)

        if self.backblend_sigma:
            flt = flt.std.MakeDiff(gauss_blur(flt.std.MakeDiff(src), self.backblend_sigma))

        return flt

    def apply_sharpen_limit(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.apply_sharpen_limit)

        if self.sharp_mode:
            if self.limit_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.SPATIAL_POSTSMOOTH):
                if self.limit_radius == 1:
                    clip = repair(clip, self.bobbed, RepairMode.MINMAX_SQUARE1)
                elif self.limit_radius > 1:
                    clip = repair(
                        clip, repair(clip, self.bobbed, RepairMode.MINMAX_SQUARE_REF2), RepairMode.MINMAX_SQUARE1
                    )

            if self.limit_mode in (SharpLimitMode.TEMPORAL_PRESMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
                backward_comp, forward_comp = self.mv.compensate(
                    self.bobbed, tr=self.limit_radius, thscd=self.thscd, interleave=False, **self.limit_comp_args
                )

                comp_min = MeanMode.MINIMUM([self.bobbed, *backward_comp, *forward_comp])
                comp_max = MeanMode.MAXIMUM([self.bobbed, *backward_comp, *forward_comp])

                clip = norm_expr(
                    [clip, comp_min, comp_max],
                    'x y {thr} - z {thr} + clip',
                    thr=scale_delta(self.limit_overshoot, 8, clip),
                )

        return clip

    def apply_noise_restore(self, clip: vs.VideoNode, restore: float = 0.0) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.apply_noise_restore)

        if restore and self.noise:
            clip = norm_expr([clip, self.noise], 'y neutral - {restore} * x +', restore=restore)

        return clip

    def apply_final(self) -> None:
        smoothed = self.mv.degrain(
            self.basic_output, tr=self.final_tr, thsad=self.final_thsad, thscd=self.thscd, **self.final_degrain_args
        )
        smoothed = self.mask_shimmer(smoothed, self.bobbed, **self.final_mask_shimmer_args)

        if self.limit_mode in (SharpLimitMode.SPATIAL_POSTSMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
            smoothed = self.apply_sharpen_limit(smoothed)

        if self.lossless_mode == LosslessMode.POSTSMOOTH and self.input_type != InputType.PROGRESSIVE:
            smoothed = self.apply_lossless(smoothed)

        self.final_output = self.apply_noise_restore(smoothed, self.final_noise_restore)

    def apply_motion_blur(self) -> None:
        angle_in, angle_out = self.motion_blur_shutter_angle

        if not angle_out * self.motion_blur_fps_divisor == angle_in:
            blur_level = (angle_out * self.motion_blur_fps_divisor - angle_in) * 100 / 360

            processed = self.mv.flow_blur(self.final_output, blur=blur_level, thscd=self.thscd, **self.motion_blur_args)

            if self.motion_blur_mask_args is not False:
                mask = self.mv.mask(
                    self.prefilter_output, direction=MVDirection.BACKWARD,
                    kind=MaskMode.MOTION, thscd=self.thscd, **self.motion_blur_mask_args,
                )

                processed = self.final_output.std.MaskedMerge(processed, mask)
        else:
            processed = self.final_output

        if self.motion_blur_fps_divisor > 1:
            processed = processed[:: self.motion_blur_fps_divisor]

        self.motion_blur_output = processed  # type: ignore

    def process(
        self,
        *,
        force_tr: int = 1,
        preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
        blksize: int | tuple[int, int] = 16,
        refine: int = 1,
        thsad_recalc: int | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = (180, 38.5),
    ) -> ConstantFormatVideoNode:
        """
        :param force_tr:        Always analyze motion to at least this, even if otherwise unnecessary.
        :param preset:          MVTools preset defining base values for the MVTools object.
        :param blksize:         Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
        :param refine:          Number of times to recalculate motion vectors with halved block size.
        :param thsad_recalc:    Only bad quality new vectors with a SAD above thid will be re-estimated by search.
                                thsad value is scaled to 8x8 block size.
        :param thscd:           Scene change detection thresholds:
                                 - First value: SAD threshold for considering a block changed between frames.
                                 - Second value: Percentage of changed blocks needed to trigger a scene change.

        :return:                Deinterlaced clip.
        """

        def _floor_div_tuple(x: tuple[int, int]) -> tuple[int, int]:
            return (x[0] // 2, x[1] // 2)

        self.draft = self.clip.resize.Bob(tff=self.tff) if self.input_type == InputType.INTERLACE else self.clip
        self.thscd = thscd

        tr = max(1, force_tr, self.denoise_tr, self.basic_tr, self.match_tr, self.final_tr)
        blksize = blksize if isinstance(blksize, tuple) else (blksize, blksize)
        preset.pop('search_clip', None)

        self.apply_prefilter()

        self.mv = MVTools(self.draft, self.prefilter_output, **preset)
        self.mv.analyze(tr=tr, blksize=blksize, overlap=_floor_div_tuple(blksize))

        if refine:
            if thsad_recalc is None:
                thsad_recalc = self.basic_thsad[0] // 2

            for _ in range(refine):
                blksize = _floor_div_tuple(blksize)
                overlap = _floor_div_tuple(blksize)

                self.mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=overlap)

        self.apply_denoise()
        self.apply_basic()
        self.apply_final()
        self.apply_motion_blur()

        return self.motion_blur_output

    def __vs_del__(self, core_id: int) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, MutableMapping):
                for k2, v2 in v.items():
                    if isinstance(v2, vs.VideoNode):
                        v[k2] = None

            if isinstance(v, vs.VideoNode):
                setattr(self, k, None)
