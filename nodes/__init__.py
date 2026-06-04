from .gpt_image_2_aio import GPTImage2AIO
from .dual_nano_gpt_image_aio import DualNanoGPTImageAIO

NODE_CLASS_MAPPINGS = {
    "GPTImage2AIO": GPTImage2AIO,
    "DualNanoGPTImageAIO": DualNanoGPTImageAIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImage2AIO": "GPT Image 2 AIO",
    "DualNanoGPTImageAIO": "Dual Nano Banana + GPT Image",
}
