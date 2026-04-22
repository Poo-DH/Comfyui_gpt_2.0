import io
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    if image_tensor is None:
        raise ValueError("image_tensor cannot be None")

    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]

    image_np = image_tensor.detach().cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def mask_to_png_bytes(mask_tensor: torch.Tensor) -> bytes:
    if mask_tensor is None:
        raise ValueError("mask_tensor cannot be None")

    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]

    mask_np = mask_tensor.detach().cpu().numpy()
    mask_np = np.clip(mask_np, 0.0, 1.0)

    # ComfyUI masks usually use white as the selected edit area.
    # OpenAI expects transparent pixels as editable regions.
    alpha_np = np.clip((1.0 - mask_np) * 255.0, 0, 255).astype(np.uint8)
    rgba_np = np.zeros((alpha_np.shape[0], alpha_np.shape[1], 4), dtype=np.uint8)
    rgba_np[..., 0:3] = 255
    rgba_np[..., 3] = alpha_np

    buffer = io.BytesIO()
    Image.fromarray(rgba_np, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def pil_to_comfy_image(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    if image.mode == "RGBA":
        rgba_np = np.array(image).astype(np.float32) / 255.0
        rgb_np = rgba_np[..., :3]
        alpha_np = rgba_np[..., 3]
    else:
        rgb_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        alpha_np = np.zeros((rgb_np.shape[0], rgb_np.shape[1]), dtype=np.float32)

    image_tensor = torch.from_numpy(rgb_np)[None, ...]
    mask_tensor = 1.0 - torch.from_numpy(alpha_np)
    return image_tensor, mask_tensor


def collect_optional_images(images: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    return [image for image in images if image is not None]
