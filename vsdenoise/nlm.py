"""
This module implements a wrapper for non local means denoisers
"""

from __future__ import annotations

import warnings

from typing import Any, Callable, Generic, NamedTuple, Sequence

from jetpytools import CustomRuntimeError, CustomStrEnum, P, R

from vstools import (
    ConstantFormatVideoNode, CustomIntEnum, PlanesT, check_variable, core, join, normalize_planes, normalize_seq,
    to_arr, vs
)

__all__ = [
    'nl_means'
]

class NLMeans(Generic[P, R]):
    """
    Class decorator that wraps the [nl_means][vsdenoise.nlm.nl_means] function
    and adds enumerations relevant to its implementation.

    It is not meant to be used directly.
    """

    def __init__(self, nl_means_func: Callable[P, R]) -> None:
        self._func = nl_means_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """        
        See [nl_means][vsdenoise.nlm.nl_means] here.
        """

        return self._func(*args, **kwargs)

    class DeviceType(CustomStrEnum):
        """Enum representing available device on which to run the plugin."""

        AUTO = 'auto'
        """
        Automatically selects the available device.
        Priority: "cuda" -> "accelerator" -> "gpu" -> "cpu" -> "ispc".
        """

        ACCELERATOR = 'accelerator'
        """Dedicated OpenCL accelerators."""

        GPU = 'gpu'
        """An OpenCL device that is a GPU."""

        CPU = 'cpu'
        """An OpenCL device that is the host processor."""

        ISPC = 'ispc'
        """ISPC (CPU-based) implementation."""

        CUDA = 'cuda'
        """CUDA (GPU-based) implementation."""

        def NLMeans(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
            """
            Applies the Non-Local Means denoising filter using the plugin associated with the selected device type.

            :param clip:                    Source clip.
            :param *args:                   Positional arguments to be passed to the selected plugin.
            :param **kwargs:                Keywords arguments to be passed to the selected plugin.        
            :raises CustomRuntimeError:     If the selected device is not available or unsupported.
            :return:                        Denoised clip.
            """

            if self == NLMeans.DeviceType.CUDA:
                return clip.nlm_cuda.NLMeans(*args, **kwargs)

            if self in [NLMeans.DeviceType.ACCELERATOR, NLMeans.DeviceType.GPU, NLMeans.DeviceType.CPU]:
                return clip.knlm.KNLMeansCL(*args, **kwargs | dict(device_type=self.value))

            if self == NLMeans.DeviceType.ISPC:
                return clip.nlm_ispc.NLMeans(*args, **kwargs)

            if hasattr(core, "nlm_cuda"):
                return NLMeans.DeviceType.CUDA.NLMeans(clip, *args, **kwargs)

            if hasattr(core, "knlm"):
                return clip.knlm.KNLMeansCL(*args, **kwargs | dict(device_type="auto"))

            if hasattr(core, "nlm_ispc"):
                return NLMeans.DeviceType.ISPC.NLMeans(clip, *args, **kwargs)

            raise CustomRuntimeError(
                "No compatible plugin found. Please install one from: "
                "https://github.com/AmusementClub/vs-nlm-cuda, https://github.com/AmusementClub/vs-nlm-ispc "
                "or https://github.com/Khanattila/KNLMeansCL"
            )

    class WeightMode(CustomIntEnum):
        """Enum of weighting modes for Non-Local Means (NLM) denoiser."""

        WELSCH = 0
        """
        Welsch weighting function has a faster decay, but still assigns positive weights to dissimilar blocks.
        Original Non-local means denoising weighting function.
        """

        BISQUARE_LR = 1
        """
        Modified Bisquare weighting function to be less robust.
        """

        BISQUARE_THR = 2
        """
        Bisquare weighting function use a soft threshold to compare neighbourhoods.
        The weight is 0 as soon as a given threshold is exceeded.
        """

        BISQUARE_HR = 3
        """
        Modified Bisquare weighting function to be even more robust.
        """

        def __call__(self, weight_ref: float | None = None) -> NLMeans.WeightModeAndRef:
            """
            :param weight_ref:  Amount of original pixel to contribute to the filter output,
                                relative to the weight of the most similar pixel found.

            :return:            Config with weight mode and ref.
            """
            return NLMeans.WeightModeAndRef(self, weight_ref)

    class WeightModeAndRef(NamedTuple):
        """
        Extended configuration for Non-Local Means (NLM) weighting,
        adding the weight reference.
        """

        weight_mode: NLMeans.WeightMode
        """See ``NLMWeightMode``"""

        weight_ref: float | None
        """
        Amount of original pixel to contribute to the filter output,
        relative to the weight of the most similar pixel found.
        """


@NLMeans
def nl_means(
    clip: vs.VideoNode,
    strength: float | Sequence[float] = 1.2,
    tr: int | Sequence[int] = 1,
    sr: int | Sequence[int] = 2,
    simr: int | Sequence[int] = 4,
    device_type: NLMeans.DeviceType = NLMeans.DeviceType.AUTO,
    ref: vs.VideoNode | None = None,
    wmode: NLMeans.WeightMode | NLMeans.WeightModeAndRef = NLMeans.WeightMode.WELSCH,
    planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Convenience wrapper for NLMeans implementations.

    Filter description [here](https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description).

    Example:
        ```py
        denoised = nl_means(clip, 0.4, device_type=nl_means.DeviceType.CUDA, ...)
        ```

    :param clip:            Source clip.

    :param strength:        Controls the strength of the filtering.
                            Larger values will remove more noise.
                            This is the ``h`` parameter.

    :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.
                            Sets the number of past and future frames to uses for denoising the current frame.
                            tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.
                            Usually, larger values result in better denoising.
                            This is the ``d`` parameter.

    :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.
                            Sets the radius of the search window.
                            sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.
                            Usually, larger values result in better denoising.
                            This is the ``a`` parameter.

    :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.
                            Sets the radius of the similarity neighbourhood window.
                            The impact on performance is low, therefore it depends on the nature of the noise.
                            This is the ``s`` parameter.

    :param device_type:     Set the device to use for processing.
    :param ref:             Reference clip to do weighting calculation.
                            This is the ``rclip`` parameter.
    :param wmode:           Weighting function to use.
    :param planes:          Which planes to process.
    :param kwargs:          Additional arguments passed to the plugin.

    :return:                Denoised clip.
    """

    assert check_variable(clip, nl_means)

    planes = normalize_planes(clip, planes)

    if not planes:
        return clip

    nstrength, ntr, nsr, nsimr = to_arr(strength), to_arr(tr), to_arr(sr), to_arr(simr)

    wmoder, wref = wmode if isinstance(wmode, NLMeans.WeightModeAndRef) else wmode()

    params = dict[str, list[float] | list[int]](strength=nstrength, tr=ntr, sr=nsr, simr=nsimr)

    def _nl_means(i: int, channels: str) -> vs.VideoNode:
        return device_type.NLMeans(
            clip, **dict(
                h=nstrength[i], d=ntr[i], a=nsr[i], s=nsimr[i],
                channels=channels, rclip=ref, wmode=wmoder.value, wref=wref
            ) | kwargs
        )

    if clip.format.color_family in {vs.GRAY, vs.RGB}:
        for doc, p in params.items():
            if len(set(p)) > 1:
                warnings.warn(
                    f'nl_means: only "{doc}" first value will be used since clip is {clip.format.color_family.name}',
                    UserWarning
                )

        return _nl_means(0, 'AUTO')

    if (
        all(len(p) < 2 for p in params.values())
        and clip.format.subsampling_w == clip.format.subsampling_h == 0
        and planes == [0, 1, 2]
    ):
        return _nl_means(0, 'YUV')

    nstrength, (ntr, nsr, nsimr) = normalize_seq(nstrength, 2), (normalize_seq(x, 2) for x in (ntr, nsr, nsimr))

    luma = _nl_means(0, 'Y') if 0 in planes else None
    chroma = _nl_means(1, 'UV') if 1 in planes or 2 in planes else None

    return join({None: clip, tuple(planes): chroma, 0: luma})
