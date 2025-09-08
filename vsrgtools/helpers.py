from typing import Any

from jetpytools import CustomTypeError, FuncExcept

from vstools import Planes, VSFunctionNoArgs, vs
from vstools.functions.funcs import FunctionUtil

from .blur import gauss_blur

__all__ = ["pre_gauss"]


def pre_gauss(
    clip: vs.VideoNode,
    function: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    sigma: float = 1.5,
    planes: Planes = None,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Apply a Gaussian blur preprocessing pattern to a clip.

    This function applies a Gaussian blur, then applies a given function to the blurred clip,
    and finally merges the difference back to preserve detail.

    Args:
        clip: The clip to process.
        function: A function to apply to the Gaussian-blurred clip.
        sigma: The sigma value for the Gaussian blur. Defaults to 1.5.
        planes: Which planes to process.
        func: An optional function to use for error handling.
        **kwargs: Additional keyword arguments passed to the function.

    Returns:
        A clip with the Gaussian preprocessing pattern applied.
    """

    func_util = FunctionUtil(clip, func or pre_gauss, planes)

    if not function:
        raise CustomTypeError

    blurred = gauss_blur(func_util.work_clip, sigma=sigma, planes=planes)
    diff_clip = func_util.work_clip.std.MakeDiff(blurred)

    processed = function(blurred, **kwargs)
    result = processed.std.MergeDiff(diff_clip)

    return func_util.return_clip(result)
