# Comfyui_gpt_2.0

ComfyUI custom node for OpenAI `gpt-image-2`.

This package adds a single AIO node that supports:

- text-to-image generation
- multi-image editing with up to 8 reference images
- optional ComfyUI `MASK` input for localized edits
- `.env` or direct node input API key configuration

## Node

`GPT Image 2 AIO`

Category:

`POODH/GPT Image`

## Inputs

Required:

- `prompt`
- `model_name`
- `image_count`
- `aspect_ratio`
- `image_size`
- `quality`
- `background`
- `output_format`
- `moderation`

Optional:

- `api_key`
- `mask`
- `image_1` to `image_8`

Behavior:

- If no input image is connected, the node calls `v1/images/generations`
- If one or more input images are connected, the node calls `v1/images/edits`

## Outputs

- `images`: ComfyUI `IMAGE`
- `alpha_mask`: extracted alpha as ComfyUI `MASK`
- `revised_prompt`: joined revised prompt text from the API
- `metadata`: compact JSON metadata

## Size controls

The node now exposes:

- `aspect_ratio`: `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`
- `aspect_ratio`: `Auto`, `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`
- `image_size`: `1K`, `2K`, `4K`

OpenAI `gpt-image-2` only supports a smaller set of native API sizes. The node therefore generates at the closest native OpenAI size and locally resizes the result when needed so the ComfyUI output matches the selected ratio and size.

When `aspect_ratio` is `Auto`, the node sends `size=auto` to OpenAI and returns a square output sized to the selected `image_size`.

## Installation

1. Copy this folder into your ComfyUI `custom_nodes` directory.
2. Install dependencies:

```bash
cd ComfyUI/custom_nodes/Comfyui_gpt_2.0
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

4. Restart ComfyUI.

## Mask behavior

The `mask` input expects a ComfyUI `MASK`.

- white area = edit this region
- black area = preserve this region

Internally the node converts that mask to a PNG alpha mask because the OpenAI Image Edit API expects transparent pixels as editable areas.

## Notes

- Default model is `gpt-image-2`
- Snapshot option `gpt-image-2-2026-04-21` is included
- Public OpenAI docs also list `v1/images/generations` and `v1/images/edits` for GPT Image 2

Official references:

- https://developers.openai.com/api/docs/models/gpt-image-2
- https://developers.openai.com/api/docs/guides/image-generation
