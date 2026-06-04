import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfy_gpt_test"


def load_dual_node_module():
    root_pkg = types.ModuleType(PACKAGE_NAME)
    root_pkg.__path__ = [str(ROOT)]
    nodes_pkg = types.ModuleType(f"{PACKAGE_NAME}.nodes")
    nodes_pkg.__path__ = [str(ROOT / "nodes")]
    utils_pkg = types.ModuleType(f"{PACKAGE_NAME}.utils")
    utils_pkg.__path__ = [str(ROOT / "utils")]
    core_pkg = types.ModuleType(f"{PACKAGE_NAME}.core")
    core_pkg.__path__ = [str(ROOT / "core")]

    sys.modules[PACKAGE_NAME] = root_pkg
    sys.modules[f"{PACKAGE_NAME}.nodes"] = nodes_pkg
    sys.modules[f"{PACKAGE_NAME}.utils"] = utils_pkg
    sys.modules[f"{PACKAGE_NAME}.core"] = core_pkg

    module_path = ROOT / "nodes" / "dual_nano_gpt_image_aio.py"
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.nodes.dual_nano_gpt_image_aio",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DualNanoGPTImageNodeTests(unittest.TestCase):
    def test_run_returns_separate_images_preview_and_metadata(self):
        module = load_dual_node_module()
        node = module.DualNanoGPTImageAIO()
        nano_image = torch.zeros((1, 10, 20, 3), dtype=torch.float32)
        gpt_image = torch.ones((1, 10, 5, 3), dtype=torch.float32)

        node._run_nano = lambda **kwargs: (nano_image, "nano thinking", "nano grounding")
        node._run_gpt = lambda **kwargs: (gpt_image, torch.zeros((1, 10, 5)), "gpt revised", "{}")

        nano, gpt, preview, metadata = node.run(
            prompt="test prompt",
            image_count=1,
            aspect_ratio="1:1",
            image_size="1K",
            nano_model_name="gemini-3.1-flash-image",
            nano_temperature=1.0,
            nano_use_search=False,
            nano_use_image_search=False,
            gpt_model_name="gpt-image-2",
            gpt_quality="auto",
            gpt_background="auto",
            gpt_output_format="png",
            gpt_moderation="auto",
        )

        self.assertEqual(tuple(nano.shape), (1, 10, 20, 3))
        self.assertEqual(tuple(gpt.shape), (1, 10, 5, 3))
        self.assertEqual(tuple(preview.shape), (1, 10, 25, 3))
        metadata_json = json.loads(metadata)
        self.assertTrue(metadata_json["nano"]["ok"])
        self.assertTrue(metadata_json["gpt"]["ok"])
        self.assertEqual(metadata_json["nano"]["thinking"], "nano thinking")
        self.assertEqual(metadata_json["gpt"]["revised_prompt"], "gpt revised")


if __name__ == "__main__":
    unittest.main()
