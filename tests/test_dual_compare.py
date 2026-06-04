import importlib.util
import time
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    module_path = ROOT / "utils" / "dual_compare.py"
    spec = importlib.util.spec_from_file_location("dual_compare", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DualCompareTests(unittest.TestCase):
    def test_run_parallel_calls_keeps_success_when_other_call_fails(self):
        dual_compare = load_module()

        def nano_call():
            time.sleep(0.05)
            return "nano-ok"

        def gpt_call():
            raise RuntimeError("gpt failed")

        results = dual_compare.run_parallel_calls(
            {"nano": nano_call, "gpt": gpt_call}
        )

        self.assertTrue(results["nano"].ok)
        self.assertEqual(results["nano"].value, "nano-ok")
        self.assertFalse(results["gpt"].ok)
        self.assertEqual(results["gpt"].error, "gpt failed")

    def test_make_side_by_side_preview_combines_first_nano_and_gpt_images(self):
        dual_compare = load_module()
        nano = torch.zeros((1, 10, 20, 3), dtype=torch.float32)
        gpt = torch.ones((1, 10, 5, 3), dtype=torch.float32)

        preview = dual_compare.make_side_by_side_preview(nano, gpt)

        self.assertEqual(tuple(preview.shape), (1, 10, 25, 3))
        self.assertTrue(torch.allclose(preview[:, :, :20, :], torch.zeros((1, 10, 20, 3))))
        self.assertTrue(torch.allclose(preview[:, :, 20:, :], torch.ones((1, 10, 5, 3))))


if __name__ == "__main__":
    unittest.main()
