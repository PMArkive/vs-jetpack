from __future__ import annotations

from functools import partial
from typing import Any, Callable, Generic, Literal, Protocol, Sequence, TypeVar, Union, overload

import vapoursynth as vs

from jetpytools import P, R, CustomValueError, fallback, flatten, interleave_arr, ranges_product

from ..functions import check_ref_clip
from ..types import ConstantFormatVideoNode, FrameRangeN, FrameRangesN, VideoNodeT

__all__ = [
    'replace_ranges',

    'remap_frames',

    'replace_every',

    'ranges_product',

    'interleave_arr',
]


_gc_func_gigacope = list[Any]()

_VideoFrameT_contra = TypeVar(
    "_VideoFrameT_contra",
    vs.VideoFrame, Sequence[vs.VideoFrame], vs.VideoFrame | Sequence[vs.VideoFrame],
    contravariant=True
)


class _RangesCallBack(Protocol):
    def __call__(self, n: int, /) -> bool:
        ...


class _RangesCallBackF(Protocol[_VideoFrameT_contra]):
    def __call__(self, f: _VideoFrameT_contra, /) -> bool:
        ...


class _RangesCallBackNF(Protocol[_VideoFrameT_contra]):
    def __call__(self, n: int, f: _VideoFrameT_contra, /) -> bool:
        ...


_RangesCallBackT = Union[
    _RangesCallBack,
    _RangesCallBackF[vs.VideoFrame],
    _RangesCallBackNF[vs.VideoFrame],
    _RangesCallBackF[Sequence[vs.VideoFrame]],
    _RangesCallBackNF[Sequence[vs.VideoFrame]],
]


class ReplaceRanges(Generic[P, R]):
    """
    Class decorator that wraps the [replace_ranges][vstools.utils.replace_ranges] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    exclusive: bool | None
    """
    Whether to use exclusive ranges globally (Default: None).
    If set to True, all calls of replace_ranges will use exclusive ranges.
    If set to False, all calls of replace_ranges will use inclusive ranges.
    """

    def __init__(self, replace_ranges: Callable[P, R]) -> None:
        self._func = replace_ranges
        self.exclusive = None

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN,
        exclusive: bool | None = None,
        mismatch: Literal[False] = ...
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def __call__(
        self,
        clip_a: VideoNodeT,
        clip_b: VideoNodeT,
        ranges: FrameRangeN | FrameRangesN,
        exclusive: bool | None = None,
        mismatch: Literal[True] | bool = ...
    ) -> VideoNodeT:
        ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBack,
        *,
        mismatch: bool = False
    ) -> vs.VideoNode:
        ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBackF[vs.VideoFrame] | _RangesCallBackNF[vs.VideoFrame],
        *,
        mismatch: bool = False,
        prop_src: vs.VideoNode
    ) -> vs.VideoNode:
        ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBackF[Sequence[vs.VideoFrame]] | _RangesCallBackNF[Sequence[vs.VideoFrame]],
        *,
        mismatch: bool = False,
        prop_src: Sequence[vs.VideoNode]
    ) -> vs.VideoNode:
        ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN | _RangesCallBackT | None,
        exclusive: bool | None = None,
        mismatch: bool = False,
        *,
        prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None
    ) -> vs.VideoNode:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)


@ReplaceRanges
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | _RangesCallBackT | None,
    exclusive: bool | None = None,
    mismatch: bool = False,
    *,
    prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None
) -> vs.VideoNode:
    """
    Replaces frames in a clip, either with pre-calculated indices or on-the-fly with a callback.

    Frame ranges are inclusive by default.
    This behaviour can be changed by setting `exclusive=True` for one-time use,
    or set `replace_ranges.exclusive = True` to apply the change globally.

    Examples with clips ``black`` and ``white`` of equal length:
        ``` py

        # Replaces frames 0 and 1 with ``white``
        replace_ranges(black, white, [(0, 1)])

        # Replaces the entire clip with ``white``
        replace_ranges(black, white, [(None, None)])

        # Same as previous
        replace_ranges(black, white, [(0, None)])

        # Replaces 200 until the end with ``white``
        replace_ranges(black, white, [(200, None)])

        # Replaces 200 until the end with ``white``, leaving 1 frame of ``black``
        replace_ranges(black, white, [(200, -1)])
        ```

    A callback function can be used to replace frames based on frame properties.
    The function must return a boolean value.

    Example of using a callback function:
        ```py
        # Replaces frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P.
        replace_ranges(clip_a, clip_b, lambda f: get_prop(f, '_PictType', str) == 'P', prop_src=clip_a)``
        ```

    Optional Dependencies:

    - [vs-zip](https://github.com/dnjulek/vapoursynth-zip) (highly recommended!)

    :param clip_a:      Original clip.
    :param clip_b:      Replacement clip.
    :param ranges:      Ranges to replace clip_a (original clip) with clip_b (replacement clip).
                        Integer values in the list indicate single frames,
                        Tuple values indicate inclusive ranges.
                        Callbacks must return true to replace a with b.
                        Negative integer values will be wrapped around based on clip_b's length.
                        None values are context dependent:
                            * None provided as sole value to ranges: no-op
                            * Single None value in list: Last frame in clip_b
                            * None as first value of tuple: 0
                            * None as second value of tuple: Last frame in clip_b
    :param exclusive:   Use exclusive ranges (Default: False).
    :param mismatch:    Accept format or resolution mismatch between clips.
    :param prop_src:    Source clip(s) to use for frame properties in the callback.
                        This is required if you're using a callback.

    :return:            Clip with ranges from clip_a replaced with clip_b.
    """
    from ..functions import invert_ranges, normalize_ranges

    if ranges != 0 and not ranges or clip_a is clip_b:
        return clip_a

    if not mismatch:
        check_ref_clip(clip_a, clip_b)

    if callable(ranges):
        from inspect import Signature

        signature = Signature.from_callable(ranges, eval_str=True)

        params = set(signature.parameters.keys())

        base_clip = clip_a.std.BlankClip(keep=True, varformat=mismatch, varsize=mismatch)

        callback = ranges

        if 'f' in params and not prop_src:
            raise CustomValueError(
                'Frame properties can only be accessed within the callback (parameter "f") '
                'if one or more source clips are explicitly provided via the `prop_src` parameter.',
                replace_ranges
            )

        def _func_nf(
            n: int, f: vs.VideoFrame | Sequence[vs.VideoFrame],
            callback: _RangesCallBackNF[vs.VideoFrame | Sequence[vs.VideoFrame]]
        ) -> vs.VideoNode:
            return clip_b if callback(n, f) else clip_a

        def _func_f(
            n: int, f: vs.VideoFrame | Sequence[vs.VideoFrame],
            callback: _RangesCallBackF[vs.VideoFrame | Sequence[vs.VideoFrame]]
        ) -> vs.VideoNode:
            return clip_b if callback(f) else clip_a

        def _func_n(n: int, callback: _RangesCallBack) -> vs.VideoNode:
            return clip_b if callback(n) else clip_a

        _func: Callable[..., vs.VideoNode]

        if 'f' in params and 'n' in params:
            _func = _func_nf
        elif 'f' in params:
            _func = _func_f
        elif 'n' in params:
            _func = _func_n
        else:
            raise CustomValueError(
                'Callback must have signature ((n, f) | (n) | (f)) -> bool!', replace_ranges, callback
            )

        _func.__callback = callback  # type: ignore[attr-defined]
        _gc_func_gigacope.append(_func)

        return vs.core.std.FrameEval(
            base_clip, partial(_func, callback=callback), prop_src if 'f' in params else None, [clip_a, clip_b]
        )

    exclusive = fallback(exclusive, replace_ranges.exclusive, False)

    b_ranges = normalize_ranges(clip_b, ranges, exclusive)

    if hasattr(vs.core, 'vszip'):
        return vs.core.vszip.RFS(
            clip_a, clip_b, [y for (s, e) in b_ranges for y in range(s, e + (not exclusive))], mismatch=mismatch
        )

    a_ranges = invert_ranges(clip_a, clip_b, b_ranges, exclusive)

    a_trims = [clip_a[max(0, start - exclusive):end + (not exclusive)] for start, end in a_ranges]
    b_trims = [clip_b[start:end + (not exclusive)] for start, end in b_ranges]

    if a_ranges:
        main, other = (a_trims, b_trims) if (a_ranges[0][0] == 0) else (b_trims, a_trims)
    else:
        main, other = (b_trims, a_trims) if (b_ranges[0][0] == 0) else (a_trims, b_trims)

    return vs.core.std.Splice(list(interleave_arr(main, other, 1)), mismatch)


def remap_frames(clip: vs.VideoNode, ranges: Sequence[int | tuple[int, int]]) -> ConstantFormatVideoNode:
    frame_map = list[int](flatten(
        f if isinstance(f, int) else range(f[0], f[1] + 1) for f in ranges
    ))

    base = vs.core.std.BlankClip(clip, length=len(frame_map))

    return vs.core.std.FrameEval(base, lambda n: clip[frame_map[n]], None, clip)


def replace_every(
    clipa: vs.VideoNode, clipb: vs.VideoNode, cycle: int, offsets: Sequence[int], modify_duration: bool = True
) -> ConstantFormatVideoNode:
    offsets_a = [x * 2 for x in range(cycle) if x not in offsets]
    offsets_b = [x * 2 + 1 for x in offsets]
    offsets = sorted(offsets_a + offsets_b)

    interleaved = vs.core.std.Interleave([clipa, clipb])

    return vs.core.std.SelectEvery(interleaved, cycle * 2, offsets, modify_duration)
