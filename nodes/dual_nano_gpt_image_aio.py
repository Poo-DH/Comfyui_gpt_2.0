import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from ..utils.dual_compare import (
    empty_image,
    image_shape_text,
    make_side_by_side_preview,
    run_parallel_calls,
)


GPT_MODEL_OPTIONS = ["gpt-image-2", "gpt-image-2-2026-04-21"]
NANO_MODEL_OPTIONS = ["gemini-3.1-flash-image"]
ASPECT_RATIO_OPTIONS = ["Auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
IMAGE_SIZE_OPTIONS = ["1K", "2K", "4K"]
GPT_QUALITY_OPTIONS = ["auto", "low", "medium", "high"]
GPT_BACKGROUND_OPTIONS = ["auto", "opaque", "transparent"]
GPT_FORMAT_OPTIONS = ["png", "jpeg", "webp"]
GPT_MODERATION_OPTIONS = ["auto", "low"]


class DualNanoGPTImageAIO:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A polished product photo of futuristic eyewear."}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "aspect_ratio": (ASPECT_RATIO_OPTIONS, {"default": "1:1"}),
                "image_size": (IMAGE_SIZE_OPTIONS, {"default": "2K"}),
                "nano_model_name": (NANO_MODEL_OPTIONS, {"default": NANO_MODEL_OPTIONS[0]}),
                "nano_temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "nano_use_search": ("BOOLEAN", {"default": False}),
                "nano_use_image_search": ("BOOLEAN", {"default": False}),
                "gpt_model_name": (GPT_MODEL_OPTIONS, {"default": GPT_MODEL_OPTIONS[0]}),
                "gpt_quality": (GPT_QUALITY_OPTIONS, {"default": "auto"}),
                "gpt_background": (GPT_BACKGROUND_OPTIONS, {"default": "auto"}),
                "gpt_output_format": (GPT_FORMAT_OPTIONS, {"default": "png"}),
                "gpt_moderation": (GPT_MODERATION_OPTIONS, {"default": "auto"}),
            },
            "optional": {
                "openai_api_key": ("STRING", {"default": "", "multiline": False}),
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("nano_images", "gpt_images", "comparison", "metadata")
    FUNCTION = "run"
    CATEGORY = "POODH/Compare"

    def run(
        self,
        prompt,
        image_count,
        aspect_ratio,
        image_size,
        nano_model_name,
        nano_temperature,
        nano_use_search,
        nano_use_image_search,
        gpt_model_name,
        gpt_quality,
        gpt_background,
        gpt_output_format,
        gpt_moderation,
        openai_api_key="",
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
        shared = {
            "prompt": prompt,
            "image_count": image_count,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "image_1": image_1,
            "image_2": image_2,
            "image_3": image_3,
            "image_4": image_4,
            "image_5": image_5,
            "image_6": image_6,
            "image_7": image_7,
            "image_8": image_8,
        }

        results = run_parallel_calls(
            {
                "nano": lambda: self._run_nano(
                    **shared,
                    nano_model_name=nano_model_name,
                    nano_temperature=nano_temperature,
                    nano_use_search=nano_use_search,
                    nano_use_image_search=nano_use_image_search,
                ),
                "gpt": lambda: self._run_gpt(
                    **shared,
                    gpt_model_name=gpt_model_name,
                    gpt_quality=gpt_quality,
                    gpt_background=gpt_background,
                    gpt_output_format=gpt_output_format,
                    gpt_moderation=gpt_moderation,
                    openai_api_key=openai_api_key,
                    mask=mask,
                ),
            }
        )

        nano_images, nano_thinking, nano_grounding = self._unwrap_nano_result(results["nano"])
        gpt_images, _gpt_mask, gpt_revised_prompt, gpt_metadata = self._unwrap_gpt_result(results["gpt"])
        comparison = make_side_by_side_preview(nano_images, gpt_images)

        metadata = self._build_metadata(
            results=results,
            nano_images=nano_images,
            gpt_images=gpt_images,
            nano_thinking=nano_thinking,
            nano_grounding=nano_grounding,
            gpt_revised_prompt=gpt_revised_prompt,
            gpt_metadata=gpt_metadata,
        )
        return nano_images, gpt_images, comparison, metadata

    def _run_nano(self, **kwargs):
        NanoBanana2AIO = _load_nano_banana_2_node()
        node = NanoBanana2AIO()
        return node.generate_unified(
            kwargs["nano_model_name"],
            kwargs["prompt"],
            kwargs["image_count"],
            kwargs["nano_use_search"],
            kwargs["nano_use_image_search"],
            kwargs.get("image_1"),
            kwargs.get("image_2"),
            kwargs.get("image_3"),
            kwargs.get("image_4"),
            kwargs.get("image_5"),
            kwargs.get("image_6"),
            kwargs.get("image_7"),
            kwargs.get("image_8"),
            None,
            None,
            None,
            None,
            None,
            None,
            kwargs["aspect_ratio"],
            kwargs["image_size"],
            kwargs["nano_temperature"],
        )

    def _run_gpt(self, **kwargs):
        from .gpt_image_2_aio import GPTImage2AIO

        node = GPTImage2AIO()
        return node.run(
            kwargs["prompt"],
            kwargs["gpt_model_name"],
            kwargs["image_count"],
            kwargs["aspect_ratio"],
            kwargs["image_size"],
            kwargs["gpt_quality"],
            kwargs["gpt_background"],
            kwargs["gpt_output_format"],
            kwargs["gpt_moderation"],
            kwargs.get("openai_api_key", ""),
            kwargs.get("mask"),
            kwargs.get("image_1"),
            kwargs.get("image_2"),
            kwargs.get("image_3"),
            kwargs.get("image_4"),
            kwargs.get("image_5"),
            kwargs.get("image_6"),
            kwargs.get("image_7"),
            kwargs.get("image_8"),
        )

    def _unwrap_nano_result(self, result):
        if not result.ok:
            return empty_image(), "", ""
        return result.value

    def _unwrap_gpt_result(self, result):
        if not result.ok:
            return empty_image(), torch.zeros((1, 64, 64), dtype=torch.float32), "", ""
        return result.value

    def _build_metadata(
        self,
        *,
        results,
        nano_images,
        gpt_images,
        nano_thinking,
        nano_grounding,
        gpt_revised_prompt,
        gpt_metadata,
    ) -> str:
        data = {
            "nano": {
                "ok": results["nano"].ok,
                "error": results["nano"].error,
                "elapsed_seconds": round(results["nano"].elapsed_seconds, 3),
                "image_shape": image_shape_text(nano_images),
                "thinking": nano_thinking,
                "grounding_sources": nano_grounding,
            },
            "gpt": {
                "ok": results["gpt"].ok and not _metadata_has_error(gpt_metadata),
                "error": results["gpt"].error or _metadata_error(gpt_metadata),
                "elapsed_seconds": round(results["gpt"].elapsed_seconds, 3),
                "image_shape": image_shape_text(gpt_images),
                "revised_prompt": gpt_revised_prompt,
                "metadata": _json_or_text(gpt_metadata),
            },
            "comparison": {
                "layout": "nano_left_gpt_right",
                "max_height": 1024,
            },
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


def _load_nano_banana_2_node():
    custom_nodes_dir = Path(__file__).resolve().parents[2]
    custom_nodes_text = str(custom_nodes_dir)
    if custom_nodes_text not in sys.path:
        sys.path.insert(0, custom_nodes_text)

    try:
        module = importlib.import_module("ComfyUI_Nano_Banana.nodes.nano_banana_2_aio")
    except ImportError as exc:
        raise RuntimeError(
            "ComfyUI_Nano_Banana is not installed next to this node. "
            "Install ru4ls/ComfyUI_Nano_Banana or use the existing Nano Banana node separately."
        ) from exc
    return module.NanoBanana2AIO


def _json_or_text(text: str) -> Any:
    if not text:
        return ""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _metadata_has_error(text: str) -> bool:
    data = _json_or_text(text)
    return isinstance(data, dict) and bool(data.get("error"))


def _metadata_error(text: str) -> str:
    data = _json_or_text(text)
    if isinstance(data, dict):
        return str(data.get("error", ""))
    return ""
