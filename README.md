# Comfyui_gpt_2.0

ComfyUI custom node for OpenAI `gpt-image-2`.  
OpenAI `gpt-image-2`용 ComfyUI 커스텀 노드입니다.

## Overview | 개요

This package adds a single AIO node for OpenAI image generation and editing workflows in ComfyUI.  
이 패키지는 ComfyUI에서 OpenAI 이미지 생성 및 편집을 한 번에 처리하는 단일 AIO 노드를 제공합니다.

Main features:  
주요 기능:

- text-to-image generation  
  텍스트만으로 이미지 생성
- multi-image editing with up to 8 reference images  
  최대 8장의 reference 이미지를 사용한 이미지 편집
- optional ComfyUI `MASK` input for localized edits  
  특정 영역만 수정할 수 있는 ComfyUI `MASK` 입력 지원
- `.env` or direct node input API key configuration  
  `.env` 또는 노드 입력칸으로 API 키 설정 가능

## Node | 노드

- Node name: `GPT Image 2 AIO`  
  노드 이름: `GPT Image 2 AIO`
- Category: `POODH/GPT Image`  
  카테고리: `POODH/GPT Image`

## Inputs | 입력

Required inputs:  
필수 입력:

- `prompt`
- `model_name`
- `image_count`
- `aspect_ratio`
- `image_size`
- `quality`
- `background`
- `output_format`
- `moderation`

Optional inputs:  
선택 입력:

- `api_key`
- `mask`
- `image_1` to `image_8`

Behavior:  
동작 방식:

- If no input image is connected, the node calls `v1/images/generations`.  
  입력 이미지가 없으면 `v1/images/generations`를 호출합니다.
- If one or more input images are connected, the node calls `v1/images/edits`.  
  하나 이상의 입력 이미지가 연결되면 `v1/images/edits`를 호출합니다.
- If `aspect_ratio` is `Auto` and an input image is connected, the output follows the first input image's aspect ratio.  
  `aspect_ratio`가 `Auto`이고 입력 이미지가 연결되어 있으면 첫 번째 입력 이미지의 비율을 따라갑니다.

## Outputs | 출력

- `images`: ComfyUI `IMAGE`  
  생성 또는 편집 결과 이미지
- `alpha_mask`: extracted alpha as ComfyUI `MASK`  
  결과 이미지의 alpha를 ComfyUI `MASK`로 반환
- `revised_prompt`: joined revised prompt text from the API  
  OpenAI API가 수정한 프롬프트 텍스트
- `metadata`: compact JSON metadata  
  간단한 JSON 메타데이터

## Size Controls | 해상도 및 비율

Available `aspect_ratio` options:  
사용 가능한 `aspect_ratio` 옵션:

- `Auto`
- `1:1`
- `2:3`
- `3:2`
- `3:4`
- `4:3`
- `4:5`
- `5:4`
- `9:16`
- `16:9`
- `21:9`

Available `image_size` options:  
사용 가능한 `image_size` 옵션:

- `1K`
- `2K`
- `4K`

How it works:  
동작 방식:

- For fixed aspect ratios, the node maps the request to a standard output size preset.  
  고정 비율 선택 시 표준 출력 해상도 프리셋으로 매핑합니다.
- For `Auto`, the node preserves the first input image aspect ratio when an input image is connected.  
  `Auto` 선택 시 입력 이미지가 있으면 첫 번째 입력 이미지 비율을 유지합니다.
- For `Auto` without any input image, the node keeps the aspect ratio returned by OpenAI.  
  입력 이미지 없이 `Auto`를 쓰면 OpenAI가 반환한 원래 비율을 유지합니다.

Examples:  
예시:

- `1:1 / 1K` -> `1024x1024`
- `4:5 / 1K` -> `896x1152`
- `9:16 / 1K` -> `768x1344`
- `16:9 / 1K` -> `1344x768`
- `21:9 / 1K` -> `1536x672`

## Installation | 설치

1. Copy this folder into your ComfyUI `custom_nodes` directory.  
   이 폴더를 ComfyUI `custom_nodes` 디렉터리 안으로 복사합니다.

2. Install dependencies.  
   의존성을 설치합니다.

```bash
cd ComfyUI/custom_nodes/Comfyui_gpt_2.0
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`.  
   `.env.example`를 참고해서 `.env` 파일을 만듭니다.

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

4. Restart ComfyUI.  
   ComfyUI를 재시작합니다.

## API Key | API 키

You can provide the OpenAI API key in either of these ways:  
OpenAI API 키는 아래 두 방식 중 하나로 넣을 수 있습니다.

- Set `OPENAI_API_KEY` in `.env`  
  `.env`에 `OPENAI_API_KEY` 설정
- Enter the key directly in the node `api_key` input  
  노드의 `api_key` 입력칸에 직접 입력

Priority order:  
우선순위:

- node input `api_key`
- `.env` `OPENAI_API_KEY`

## Mask Behavior | 마스크 동작

The `mask` input expects a ComfyUI `MASK`.  
`mask` 입력은 ComfyUI `MASK` 형식을 기대합니다.

- white area = edit this region  
  흰색 영역 = 수정할 영역
- black area = preserve this region  
  검은색 영역 = 유지할 영역

Internally, the node converts the mask into a PNG alpha mask because the OpenAI Image Edit API expects transparent pixels as editable regions.  
내부적으로는 OpenAI Image Edit API가 투명 픽셀을 수정 가능 영역으로 해석하므로, ComfyUI 마스크를 PNG alpha mask로 변환합니다.

## Models | 모델

Included model options:  
포함된 모델 옵션:

- `gpt-image-2`
- `gpt-image-2-2026-04-21`

Recommended usage:  
권장 사용:

- Use `gpt-image-2` for the latest default model alias.  
  최신 기본 alias를 쓰려면 `gpt-image-2`
- Use `gpt-image-2-2026-04-21` when you want a fixed snapshot.  
  고정 스냅샷을 쓰려면 `gpt-image-2-2026-04-21`

## Notes | 참고

- Default model is `gpt-image-2`.  
  기본 모델은 `gpt-image-2`입니다.
- This node uses OpenAI image generation and image edit endpoints.  
  이 노드는 OpenAI 이미지 생성 및 편집 엔드포인트를 사용합니다.
- Some models or features may require OpenAI organization verification.  
  일부 모델 또는 기능은 OpenAI 조직 인증이 필요할 수 있습니다.

## Official References | 공식 문서

- https://developers.openai.com/api/docs/models/gpt-image-2
- https://developers.openai.com/api/docs/guides/image-generation
