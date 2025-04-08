from abc import ABC
from logging import warning
from typing import Any, ClassVar, Literal

from jetpytools import KwargsT

from vsexprtools import norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelT, ScalerT
from vstools import (
    ConstantFormatVideoNode, CustomValueError, DitherType, Matrix, ProcessVariableResClip, SPath, SPathLike,
    check_variable_format, core, depth, get_nvidia_version, get_video_format, get_y, inject_self, limiter, padder, vs
)

from .generic import BaseGenericScaler

__all__ = [
    "autoselect_backend",

    "BaseOnnxScaler",

    "GenericOnnxScaler",

    "ArtCNN",

    "Waifu2x"
]


def _clean_keywords(kwargs: dict[str, Any], backend: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in backend.__dataclass_fields__}


def autoselect_backend(**kwargs: Any) -> Any:
    """
    Try to select the best backend for the current system.
    If the system has an NVIDIA GPU: TRT > CUDA (ORT) > Vulkan > OpenVINO GPU
    Else: DirectML (D3D12) > MIGraphX > Vulkan > CPU (ORT) > CPU OpenVINO

    :param kwargs:        Additional arguments to pass to the backend.
    :return:              The selected backend.
    """
    import os

    from vsmlrt import Backend

    backend: Any

    if get_nvidia_version():
        if hasattr(core, "trt"):
            backend = Backend.TRT
        elif hasattr(core, "ort"):
            backend = Backend.ORT_CUDA
        elif hasattr(core, "ncnn"):
            backend = Backend.NCNN_VK
        else:
            backend = Backend.OV_GPU
    else:
        if hasattr(core, "ort") and os.name == "nt":
            backend = Backend.ORT_DML
        elif hasattr(core, "migx"):
            backend = Backend.MIGX
        elif hasattr(core, "ncnn"):
            backend = Backend.NCNN_VK
        elif hasattr(core, "ort"):
            backend = Backend.ORT_CPU
        else:
            backend = Backend.OV_CPU

    return backend(**_clean_keywords(kwargs, backend))


class BaseOnnxScaler(BaseGenericScaler, ABC):
    """Abstract generic scaler class for an ONNX model."""

    def __init__(
        self,
        model: SPathLike | None = None,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param model:           Path to the ONNX model file.
        :param backend:         The backend to be used with the vs-mlrt framework.
                                If set to None, the most suitable backend will be automatically selected, prioritizing fp16 support.
        :param tiles:           Whether to split the image into multiple tiles.
                                This can help reduce VRAM usage, but note that the model's behavior may vary when they are used.
        :param tilesize:        The size of each tile when splitting the image (if tiles are enabled).
        :param overlap:         The size of overlap between tiles.
        :param max_instances:   Maximum instances to spawn when scaling a variable resolution clip.
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        :param **kwargs:        Additional arguments to pass to the backend.
                                See the vsmlrt backend's docstring for more details.
        """
        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

        if model is not None:
            self.model = str(SPath(model).resolve())

        if backend is None:
            _default_args = KwargsT(fp16=True, output_format=1, use_cuda_graph=True, use_cublas=True, heuristic=True)
            self.backend = autoselect_backend(**_default_args | self.kwargs)
        else:
            self.backend = backend

        self.tiles = tiles
        self.tilesize = tilesize
        self.overlap = overlap

        if self.overlap is None:
            self.overlap_w = self.overlap_h = 8
        elif isinstance(self.overlap, int):
            self.overlap_w = self.overlap_h = self.overlap
        else:
            self.overlap_w, self.overlap_h = self.overlap

        self.max_instances = max_instances

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Scale the given clip using the ONNX model.

        :param clip:        The input clip to be scaled.
        :param width:       The target width for scaling. If None, the width of the input clip will be used.
        :param height:      The target height for scaling. If None, the height of the input clip will be used.
        :param shift:       A tuple representing the shift values for the x and y axes.
        :param **kwargs:    Additional arguments to be passed to the `preprocess_clip`, `postprocess_clip`,
                            `inference`, and `_final_scale` methods.
                            Use the prefix `preprocess_` or `postprocess_` to pass an argument to the respective method.
                            Use the prefix `inference_` to pass an argument to the inference method.

        :return:            The scaled clip.
        """
        from vsmlrt import Backend

        assert check_variable_format(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        preprocess_kwargs = dict[str, Any]()
        postprocess_kwargs = dict[str, Any]()
        inference_kwargs = dict[str, Any]()

        for k in kwargs.copy():
            for prefix, ckwargs in zip(
                ("preprocess_", "postprocess_", "inference_"),
                (preprocess_kwargs, postprocess_kwargs, inference_kwargs)
            ):
                if k.startswith(prefix):
                    ckwargs[k.removeprefix(prefix)] = kwargs.pop(k)
                    break

        wclip = self.preprocess_clip(clip, **preprocess_kwargs)

        if 0 not in {clip.width, clip.height}:
            scaled = self.inference(wclip, **inference_kwargs)
        else:
            if not isinstance(self.backend, Backend.TRT):
                raise CustomValueError(
                    "Variable resolution clips can only be processed with TRT Backend!", self.__class__, self.backend
                )

            warning(f"{self.__class__.__name__}: Variable resolution clip detected!")

            if self.backend.static_shape:
                warning("static_shape is True, setting it to False...")
                self.backend.static_shape = False

            if not self.backend.max_shapes:
                warning("max_shapes is None, setting it to (1936, 1088). You may want to adjust it...")
                self.backend.max_shapes = (1936, 1088)

            if not self.backend.opt_shapes:
                warning("opt_shapes is None, setting it to (64, 64). You may want to adjust it...")
                self.backend.opt_shapes = (64, 64)

            scaled = ProcessVariableResClip[ConstantFormatVideoNode].from_func(
                wclip, lambda c: self.inference(c, **inference_kwargs), False, wclip.format, self.max_instances
            )

        scaled = self.postprocess_clip(scaled, clip, **postprocess_kwargs)

        return self._finish_scale(scaled, clip, width, height, shift, **kwargs)

    def calc_tilesize(self, clip: vs.VideoNode) -> tuple[tuple[int, int], tuple[int, int]]:
        from vsmlrt import calc_tilesize

        return calc_tilesize(
            tiles=self.tiles,
            tilesize=self.tilesize,
            width=clip.width,
            height=clip.height,
            multiple=1,
            overlap_w=self.overlap_w,
            overlap_h=self.overlap_h,
        )

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        clip = depth(clip, 16 if self.backend.fp16 else 32, vs.FLOAT)
        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return depth(
            clip, input_clip, dither_type=DitherType.ORDERED if 0 in {clip.width, clip.height} else DitherType.AUTO
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        from vsmlrt import inference

        tiles, overlaps = self.calc_tilesize(clip)

        return inference(clip, self.model, overlaps, tiles, self.backend, **kwargs)


class GenericOnnxScaler(BaseOnnxScaler):
    """Generic scaler class for an ONNX model."""

    _static_kernel_radius = 2


class BaseArtCNN(BaseOnnxScaler):
    _model: ClassVar[int]
    _static_kernel_radius = 2

    def __init__(
        self,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param backend:         The backend to be used with the vs-mlrt framework.
                                If set to None, the most suitable backend will be automatically selected, prioritizing fp16 support.
        :param tiles:           Whether to split the image into multiple tiles.
                                This can help reduce VRAM usage, but note that the model's behavior may vary when they are used.
        :param tilesize:        The size of each tile when splitting the image (if tiles are enabled).
        :param overlap:         The size of overlap between tiles.
        :param max_instances:   Maximum instances to spawn when scaling a variable resolution clip.
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        :param **kwargs:        Additional arguments to pass to the backend.
                                See the vsmlrt backend's docstring for more details.
        """
        super().__init__(
            None, backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        from vsmlrt import ArtCNN as mlrt_ArtCNN
        from vsmlrt import ArtCNNModel

        return mlrt_ArtCNN(clip, self.tiles, self.tilesize, self.overlap, ArtCNNModel(self._model), self.backend)


class BaseArtCNNLuma(BaseArtCNN):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return super().preprocess_clip(get_y(clip), **kwargs)


class BaseArtCNNChroma(BaseArtCNN):
    def __init__(
        self,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        chroma_scaler: KernelT = Bilinear,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param backend:         The backend to be used with the vs-mlrt framework.
                                If set to None, the most suitable backend will be automatically selected, prioritizing fp16 support.
        :param tiles:           Whether to split the image into multiple tiles.
                                This can help reduce VRAM usage, but note that the model's behavior may vary when they are used.
        :param tilesize:        The size of each tile when splitting the image (if tiles are enabled).
        :param overlap:         The size of overlap between tiles.
        :param max_instances:   Maximum instances to spawn when scaling a variable resolution clip.
        :param chroma_scaler:   Scaler to upscale the chroma with. Defaults to Bilinear.
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        :param **kwargs:        Additional arguments to pass to the backend.
                                See the vsmlrt backend's docstring for more details.
        """
        self.chroma_scaler = Kernel.ensure_obj(chroma_scaler)

        super().__init__(
            backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable_format(clip, self.__class__)

        if clip.format.subsampling_h != 0 or clip.format.subsampling_w != 0:
            clip = self.chroma_scaler.resample(
                clip, clip.format.replace(
                    subsampling_h=0, subsampling_w=0,
                    sample_type=vs.FLOAT, bits_per_sample=16 if self.backend.fp16 else 32
                )
            )
            return limiter(clip, func=self.__class__)

        return super().preprocess_clip(clip, **kwargs)


class ArtCNN(BaseArtCNNLuma):
    """
    Super-Resolution Convolutional Neural Networks optimised for anime.

    A quick reminder that vs-mlrt does not ship these in the base package.\n
    You will have to grab the extended models pack or get it from the repo itself.\n
    (And create an "ArtCNN" folder in your models folder yourself)

    https://github.com/Artoriuz/ArtCNN/releases/latest

    Defaults to R8F64.
    """

    _model = 7

    class C4F32(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 32 filters each.\n
        If you need an even faster model.
        """

        _model = 0

    class C4F32_DS(BaseArtCNNLuma):
        """The same as C4F32 but intended to also sharpen and denoise."""

        _model = 1

    class C16F64(BaseArtCNNLuma):
        """
        Very fast and good enough for AA purposes but the onnx variant is officially deprecated.\n
        This has 16 internal convolution layers with 64 filters each.

        ONNX files available at https://github.com/Artoriuz/ArtCNN/tree/388b91797ff2e675fd03065953cc1147d6f972c2/ONNX
        """

        _model = 2

    class C16F64_DS(BaseArtCNNLuma):
        """The same as C16F64 but intended to also sharpen and denoise."""

        _model = 3

    class C4F32_Chroma(BaseArtCNNChroma):
        """
        The smaller of the two chroma models.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 4

    class C16F64_Chroma(BaseArtCNNChroma):
        """
        The bigger of the two chroma models.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 5

    class R16F96(BaseArtCNNLuma):
        """
        The biggest model. Can compete with or outperform Waifu2x Cunet.\n
        Also quite a bit slower but is less heavy on vram.
        """

        _model = 6

    class R8F64(BaseArtCNNLuma):
        """
        A smaller and faster version of R16F96 but very competitive.
        """

        _model = 7

    class R8F64_DS(BaseArtCNNLuma):
        """The same as R8F64 but intended to also sharpen and denoise."""

        _model = 8

    class R8F64_Chroma(BaseArtCNNChroma):
        """
        The new and fancy big chroma model.\n
        These don't double the input clip and rather just try to enhance the chroma using luma information.
        """

        _model = 9

    class C4F16(BaseArtCNNLuma):
        """
        This has 4 internal convolution layers with 16 filters each.\n
        The currently fastest variant. Not really recommended for any filtering.\n
        Should strictly be used for real-time applications and even then the other non R ones should be fast enough...
        """

        _model = 10

    class C4F16_DS(BaseArtCNNLuma):
        """The same as C4F16 but intended to also sharpen and denoise."""

        _model = 11


class BaseWaifu2x(BaseOnnxScaler):
    scale_w2x: Literal[1, 2, 4]
    noise: Literal[-1, 0, 1, 2, 3]

    _model: ClassVar[int]
    _static_kernel_radius = 2

    def __init__(
        self,
        scale: Literal[1, 2, 4] = 2,
        noise: Literal[-1, 0, 1, 2, 3] = -1,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        """
        :param scale:           Upscaling factor. 1 = no uspcaling, 2 = 2x, 4 = 4x.
        :param noise:           Noise reduction level. -1 = none, 0 = low, 1 = medium, 2 = high, 3 = highest.
        :param backend:         The backend to be used with the vs-mlrt framework.
                                If set to None, the most suitable backend will be automatically selected, prioritizing fp16 support.
        :param tiles:           Whether to split the image into multiple tiles.
                                This can help reduce VRAM usage, but note that the model's behavior may vary when they are used.
        :param tilesize:        The size of each tile when splitting the image (if tiles are enabled).
        :param overlap:         The size of overlap between tiles.
        :param max_instances:   Maximum instances to spawn when scaling a variable resolution clip.
        :param kernel:          Base kernel to be used for certain scaling/shifting/resampling operations.
                                Defaults to Catrom.
        :param scaler:          Scaler used for scaling operations. Defaults to kernel.
        :param shifter:         Kernel used for shifting operations. Defaults to scaler.
        :param **kwargs:        Additional arguments to pass to the backend.
                                See the vsmlrt backend's docstring for more details.
        """
        self.scale_w2x = scale
        self.noise = noise
        super().__init__(
            None, backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        from vsmlrt import Waifu2x as mlrt_Waifu2x
        from vsmlrt import Waifu2xModel

        return mlrt_Waifu2x(
            clip,
            self.noise,
            self.scale_w2x,
            self.tiles,
            self.tilesize,
            self.overlap,
            Waifu2xModel(self._model),
            self.backend,
            **kwargs
        )


class BaseWaifu2xRGB(BaseWaifu2x):
    def preprocess_clip(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        clip = self.kernel.resample(clip, vs.RGBH if self.backend.fp16 else vs.RGBS, Matrix.RGB)
        return limiter(clip, func=self.__class__)

    def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable_format(clip, self.__class__)

        if clip.format != get_video_format(input_clip):
            kwargs = dict(dither_type=DitherType.ORDERED) | kwargs
            clip = self.kernel.resample(clip, input_clip, Matrix.from_video(input_clip, func=self.__class__), **kwargs)

        return clip


class BaseWaifu2xMlrtPreprocess(BaseWaifu2x):
    def __init__(
        self,
        scale: Literal[1, 2, 4] = 2,
        noise: Literal[-1, 0, 1, 2, 3] = -1,
        preprocess: bool = True,
        backend: Any | None = None,
        tiles: int | tuple[int, int] | None = None,
        tilesize: int | tuple[int, int] | None = None,
        overlap: int | tuple[int, int] | None = None,
        max_instances: int = 2,
        *,
        kernel: KernelT = Catrom,
        scaler: ScalerT | None = None,
        shifter: KernelT | None = None,
        **kwargs: Any
    ) -> None:
        self.preprocess = preprocess
        super().__init__(
            scale, noise, backend, tiles, tilesize, overlap, max_instances, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs
        )

    def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return super().inference(clip, preprocess=self.preprocess, **kwargs)


class Waifu2x(BaseWaifu2xRGB):
    _model = 6

    class AnimeStyleArt(BaseWaifu2xMlrtPreprocess, BaseWaifu2x):
        _model = 0

    class AnimeStyleArtRGB(BaseWaifu2xMlrtPreprocess, BaseWaifu2xRGB):
        _model = 1

    class Photo(BaseWaifu2xMlrtPreprocess, BaseWaifu2xRGB):
        _model = 2

    class UpConv7AnimeStyleArt(BaseWaifu2xRGB):
        _model = 3

    class UpConv7Photo(BaseWaifu2xRGB):
        _model = 4

    class UpResNet10(BaseWaifu2xRGB):
        _model = 5

    class Cunet(BaseWaifu2xRGB):
        _model = 6

        def inference(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
            # Cunet model ruins image borders, so we need to pad it before upscale and crop it after.
            with padder.ctx(16, 4) as pad:
                padded = pad.MIRROR(clip)
                scaled = super().inference(padded, **kwargs)
                cropped = pad.CROP(scaled)

            return cropped

        def postprocess_clip(self, clip: vs.VideoNode, input_clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
            # Cunet model also has a tint issue
            tint_fix = norm_expr(
                clip, 'x 0.5 255 / + 0 1 clamp',
                planes=0 if get_video_format(input_clip).color_family is vs.GRAY else None,
                func="Waifu2x." + self.__class__.__name__
            )
            return super().postprocess_clip(tint_fix, input_clip, **kwargs)

    class SwinUnetArt(BaseWaifu2xRGB):
        _model = 7

    class SwinUnetPhoto(BaseWaifu2xRGB):
        _model = 8

    class SwinUnetPhotoV2(BaseWaifu2xRGB):
        _model = 9

    class SwinUnetArtScan(BaseWaifu2xRGB):
        _model = 10
