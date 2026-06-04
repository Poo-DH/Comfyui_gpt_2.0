import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image


@dataclass
class ParallelCallResult:
    ok: bool
    value: Any = None
    error: str = ""
    elapsed_seconds: float = 0.0


def run_parallel_calls(calls: Dict[str, Callable[[], Any]]) -> Dict[str, ParallelCallResult]:
    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as executor:
        futures = {
            name: executor.submit(_run_timed_call, call)
            for name, call in calls.items()
        }
        return {name: future.result() for name, future in futures.items()}


def make_side_by_side_preview(
    nano_images: torch.Tensor,
    gpt_images: torch.Tensor,
    max_height: int = 1024,
) -> torch.Tensor:
    nano_pil = _first_tensor_image_to_pil(nano_images)
    gpt_pil = _first_tensor_image_to_pil(gpt_images)

    target_height = min(max(nano_pil.height, gpt_pil.height), max_height)
    nano_pil = _resize_to_height(nano_pil, target_height)
    gpt_pil = _resize_to_height(gpt_pil, target_height)

    preview = Image.new("RGB", (nano_pil.width + gpt_pil.width, target_height), (0, 0, 0))
    preview.paste(nano_pil, (0, 0))
    preview.paste(gpt_pil, (nano_pil.width, 0))
    return _pil_to_tensor(preview)


def empty_image() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def image_shape_text(image: torch.Tensor) -> str:
    if image is None:
        return "None"
    return "x".join(str(part) for part in image.shape)


def _run_timed_call(call: Callable[[], Any]) -> ParallelCallResult:
    started = time.perf_counter()
    try:
        return ParallelCallResult(
            ok=True,
            value=call(),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return ParallelCallResult(
            ok=False,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - started,
        )


def _first_tensor_image_to_pil(images: torch.Tensor) -> Image.Image:
    if images is None:
        raise ValueError("image tensor cannot be None")
    if images.ndim == 4:
        image = images[0]
    elif images.ndim == 3:
        image = images
    else:
        raise ValueError(f"unsupported image tensor shape: {tuple(images.shape)}")

    image_np = image.detach().cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    return Image.fromarray(image_np).convert("RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np)[None, ...]


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    if image.height == target_height:
        return image
    scale = target_height / float(image.height)
    target_width = max(1, round(image.width * scale))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
