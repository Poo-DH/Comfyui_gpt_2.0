import base64
import io
import json
from typing import Any, Dict, List

import requests
import torch
from PIL import Image

from ..core import get_api_key
from ..utils import (
    collect_optional_images,
    mask_to_png_bytes,
    pil_to_comfy_image,
    pil_to_png_bytes,
    tensor_to_pil,
)


IMAGE_API_BASE = "https://api.openai.com/v1/images"
MODEL_OPTIONS = ["gpt-image-2", "gpt-image-2-2026-04-21"]
ASPECT_RATIO_OPTIONS = ["Auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
IMAGE_SIZE_OPTIONS = ["1K", "2K", "4K"]
QUALITY_OPTIONS = ["auto", "low", "medium", "high"]
BACKGROUND_OPTIONS = ["auto", "opaque", "transparent"]
FORMAT_OPTIONS = ["png", "jpeg", "webp"]
MODERATION_OPTIONS = ["auto", "low"]
BASE_DIMENSIONS = {
    "1:1": (1024, 1024),
    "2:3": (832, 1248),
    "3:2": (1248, 832),
    "3:4": (864, 1184),
    "4:3": (1184, 864),
    "4:5": (896, 1152),
    "5:4": (1152, 896),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
    "21:9": (1536, 672),
}
SIZE_FACTORS = {
    "1K": 1,
    "2K": 2,
    "4K": 4,
}


class GPTImage2AIO:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A polished product photo of a futuristic eyewear display."}),
                "model_name": (MODEL_OPTIONS, {"default": MODEL_OPTIONS[0]}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "Auto"}),
                "image_size": (IMAGE_SIZE_OPTIONS, {"default": "2K"}),
                "quality": (QUALITY_OPTIONS, {"default": "auto"}),
                "background": (BACKGROUND_OPTIONS, {"default": "auto"}),
                "output_format": (FORMAT_OPTIONS, {"default": "png"}),
                "moderation": (MODERATION_OPTIONS, {"default": "auto"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "mask": ("MASK",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("images", "alpha_mask", "revised_prompt", "metadata")
    FUNCTION = "run"
    CATEGORY = "POODH/GPT Image"

    def run(
        self,
        prompt,
        model_name,
        image_count,
        aspect_ratio,
        image_size,
        quality,
        background,
        output_format,
        moderation,
        api_key="",
        mask=None,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        image_7=None,
        image_8=None,
    ):
        try:
            api_key = get_api_key(api_key)
            prompt = (prompt or "").strip()
            if not prompt:
                raise ValueError("prompt cannot be empty")

            input_images = collect_optional_images(
                [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]
            )

            api_size, target_size, resize_note = self._resolve_sizes(aspect_ratio, image_size, input_images)

            if input_images:
                response_json = self._edit_images(
                    api_key=api_key,
                    prompt=prompt,
                    model_name=model_name,
                    api_size=api_size,
                    quality=quality,
                    background=background,
                    output_format=output_format,
                    moderation=moderation,
                    input_images=input_images,
                    mask=mask,
                )
            else:
                response_json = self._generate_images(
                    api_key=api_key,
                    prompt=prompt,
                    model_name=model_name,
                    image_count=image_count,
                    api_size=api_size,
                    quality=quality,
                    background=background,
                    output_format=output_format,
                    moderation=moderation,
                )

            return self._decode_response(response_json, target_size, resize_note, aspect_ratio, image_size, api_size)
        except Exception as exc:
            return self._error_result(exc)

    def _generate_images(
        self,
        *,
        api_key: str,
        prompt: str,
        model_name: str,
        image_count: int,
        api_size: str,
        quality: str,
        background: str,
        output_format: str,
        moderation: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "n": image_count,
            "size": api_size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "moderation": moderation,
        }
        return self._post_json("/generations", api_key, payload)

    def _edit_images(
        self,
        *,
        api_key: str,
        prompt: str,
        model_name: str,
        api_size: str,
        quality: str,
        background: str,
        output_format: str,
        moderation: str,
        input_images: List[torch.Tensor],
        mask: torch.Tensor | None,
    ) -> Dict[str, Any]:
        data = {
            "model": model_name,
            "prompt": prompt,
            "size": api_size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
            "moderation": moderation,
        }

        files = []
        for index, image_tensor in enumerate(input_images, start=1):
            image_bytes = pil_to_png_bytes(tensor_to_pil(image_tensor))
            files.append(("image[]", (f"image_{index}.png", image_bytes, "image/png")))

        if mask is not None:
            files.append(("mask", ("mask.png", mask_to_png_bytes(mask), "image/png")))

        return self._post_multipart("/edits", api_key, data, files)

    def _post_json(self, path: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{IMAGE_API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        return self._parse_response(response)

    def _post_multipart(
        self,
        path: str,
        api_key: str,
        data: Dict[str, Any],
        files: List[tuple[str, tuple[str, bytes, str]]],
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{IMAGE_API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            data=data,
            files=files,
            timeout=300,
        )
        return self._parse_response(response)

    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"OpenAI returned a non-JSON response ({response.status_code}).") from exc

        if not response.ok:
            error = payload.get("error", {})
            message = error.get("message") or f"OpenAI request failed with status {response.status_code}"
            raise RuntimeError(message)

        if not payload.get("data"):
            raise RuntimeError("OpenAI returned no image data.")

        return payload

    def _decode_response(
        self,
        response_json: Dict[str, Any],
        target_size: tuple[int, int] | None,
        resize_note: str,
        aspect_ratio: str,
        image_size: str,
        api_size: str,
    ):
        image_tensors = []
        mask_tensors = []
        revised_prompts = []

        for item in response_json.get("data", []):
            b64_json = item.get("b64_json")
            if not b64_json:
                continue

            image_bytes = base64.b64decode(b64_json)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            pil_image = self._resize_image_if_needed(pil_image, target_size)
            image_tensor, mask_tensor = pil_to_comfy_image(pil_image)
            image_tensors.append(image_tensor)
            mask_tensors.append(mask_tensor[None, ...] if mask_tensor.ndim == 2 else mask_tensor)

            revised_prompt = item.get("revised_prompt", "")
            if revised_prompt:
                revised_prompts.append(revised_prompt)

        if not image_tensors:
            raise RuntimeError("OpenAI returned items, but none included b64_json image content.")

        combined_images = torch.cat(image_tensors, dim=0)
        combined_masks = torch.cat(mask_tensors, dim=0)
        revised_prompt_text = "\n\n".join(revised_prompts)

        metadata = self._build_metadata(
            response_json,
            revised_prompts,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            api_size=api_size,
            target_size=target_size,
            resize_note=resize_note,
        )
        return combined_images, combined_masks, revised_prompt_text, metadata

    def _build_metadata(
        self,
        response_json: Dict[str, Any],
        revised_prompts: List[str],
        *,
        aspect_ratio: str,
        image_size: str,
        api_size: str,
        target_size: tuple[int, int] | None,
        resize_note: str,
    ) -> str:
        compact_data = []
        for item in response_json.get("data", []):
            compact_data.append(
                {
                    "revised_prompt": item.get("revised_prompt", ""),
                }
            )

        metadata = {
            "created": response_json.get("created"),
            "images_returned": len(compact_data),
            "requested_aspect_ratio": aspect_ratio,
            "requested_image_size": image_size,
            "api_size": api_size,
            "final_size": (
                {"width": target_size[0], "height": target_size[1]}
                if target_size is not None
                else None
            ),
            "resize_note": resize_note,
            "revised_prompts": revised_prompts,
            "data": compact_data,
        }
        return json.dumps(metadata, ensure_ascii=False, indent=2)

    def _resolve_sizes(
        self,
        aspect_ratio: str,
        image_size: str,
        input_images: List[torch.Tensor],
    ) -> tuple[str, tuple[int, int] | None, str]:
        if aspect_ratio == "Auto":
            if input_images:
                source_width, source_height = self._tensor_dimensions(input_images[0])
                target_size = self._scaled_dimensions_from_source(source_width, source_height, image_size)
                resize_note = (
                    "OpenAI chooses the base composition automatically. The node preserves the first input image aspect ratio."
                )
                return "auto", target_size, resize_note

            resize_note = (
                "OpenAI chooses the base composition automatically. The node keeps the API returned aspect ratio when no input image is connected."
            )
            return "auto", None, resize_note

        width_ratio, height_ratio = self._parse_ratio(aspect_ratio)

        if width_ratio == height_ratio:
            api_size = "1024x1024"
        elif width_ratio > height_ratio:
            api_size = "1536x1024"
        else:
            api_size = "1024x1536"

        target_size = self._preset_dimensions(aspect_ratio, image_size)

        if image_size == "1K" and aspect_ratio in ("1:1", "2:3", "3:2"):
            resize_note = "Direct native/OpenAI-compatible size."
        else:
            resize_note = (
                "Generated at the closest native OpenAI size and resized locally to match the requested aspect ratio/size."
            )

        return api_size, target_size, resize_note

    def _parse_ratio(self, ratio_text: str) -> tuple[int, int]:
        width_text, height_text = ratio_text.split(":")
        return int(width_text), int(height_text)

    def _preset_dimensions(self, aspect_ratio: str, image_size: str) -> tuple[int, int]:
        base_width, base_height = BASE_DIMENSIONS[aspect_ratio]
        factor = SIZE_FACTORS[image_size]
        return base_width * factor, base_height * factor

    def _tensor_dimensions(self, image_tensor: torch.Tensor) -> tuple[int, int]:
        if image_tensor.ndim == 4:
            _, height, width, _ = image_tensor.shape
        elif image_tensor.ndim == 3:
            height, width, _ = image_tensor.shape
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(image_tensor.shape)}")
        return width, height

    def _scaled_dimensions_from_source(
        self,
        source_width: int,
        source_height: int,
        image_size: str,
    ) -> tuple[int, int]:
        target_long_side = {
            "1K": 1024,
            "2K": 2048,
            "4K": 4096,
        }[image_size]
        source_long_side = max(source_width, source_height)
        scale = target_long_side / float(source_long_side)
        width = max(1, round(source_width * scale))
        height = max(1, round(source_height * scale))
        return width, height

    def _resize_image_if_needed(
        self,
        image: Image.Image,
        target_size: tuple[int, int] | None,
    ) -> Image.Image:
        if target_size is None or image.size == target_size:
            return image
        return image.resize(target_size, Image.Resampling.LANCZOS)

    def _error_result(self, exc: Exception):
        print(f"[GPTImage2AIO] ERROR: {exc}")
        empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        metadata = json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)
        return empty_image, empty_mask, "", metadata
