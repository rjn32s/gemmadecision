"""Unit checks use tiny synthetic tensors only: no downloads or model loading."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "python/gemmadecision/backends/torch.py"
spec = importlib.util.spec_from_file_location("gemmadecision_torch_under_test", MODULE)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class BatchPlanTests(unittest.TestCase):
    def test_padded_tokens_bound_and_restorable_order(self):
        lengths = [8, 3, 7, 2, 3, 10]
        groups = backend.plan_batches(lengths, max_batch_tokens=20, max_batch_size=3)
        self.assertEqual(groups, [[3, 1, 4], [2, 0], [5]])
        self.assertEqual(sorted(index for group in groups for index in group), list(range(6)))
        for group in groups:
            self.assertLessEqual(max(lengths[i] for i in group) * len(group), 20)

    def test_single_pair_batch_shape(self):
        self.assertEqual(backend.plan_batches([8, 3, 7], 20, 1), [[1], [2], [0]])

    def test_no_silent_truncation(self):
        for lengths, tokens, size in [([100], 99, 32), ([0], 100, 32), ([1], 0, 32), ([1], 100, 0)]:
            with self.assertRaises(ValueError):
                backend.plan_batches(lengths, tokens, size)
        self.assertEqual(backend.plan_batches([], 100, 32), [])

    def test_import_is_lazy(self):
        code = (
            "import importlib.util,sys; "
            f"s=importlib.util.spec_from_file_location('backend',{str(MODULE)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_integrity_checks_actual_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "model.safetensors"
            file.write_bytes(b"expected-test-bytes")
            hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()}
            with patch.object(backend, "FROZEN_FILES", hashes):
                backend.verify_model_files(directory)
                file.write_bytes(b"altered-test-bytes")
                with self.assertRaisesRegex(ValueError, "integrity check failed"):
                    backend.verify_model_files(directory)
                file.unlink()
                with self.assertRaisesRegex(ValueError, "missing"):
                    backend.verify_model_files(directory)


class TorchPoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Optional torch dependency is not installed")
        cls.torch = torch

    def make_backend(self, batch_size=3):
        torch = self.torch

        class TinyTokenizer:
            pad_token_id = 0

            def __call__(self, texts, **kwargs):
                assert kwargs["add_special_tokens"] is True
                assert kwargs["truncation"] is False
                assert kwargs["padding"] is False
                encode = lambda text: [2] + [ord(c) % 20 + 1 for c in text]
                return {"input_ids": encode(texts) if isinstance(texts, str) else [encode(t) for t in texts]}

            def pad(self, value, **kwargs):
                assert kwargs["return_attention_mask"] is True
                longest = max(map(len, value["input_ids"]))
                ids = [list(row) + [0] * (longest - len(row)) for row in value["input_ids"]]
                masks = [[1] * len(row) + [0] * (longest - len(row)) for row in value["input_ids"]]
                return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

        class TinyEncoder:
            def __init__(self):
                self.calls = []

            def __call__(self, *, input_ids, attention_mask, use_cache):
                assert use_cache is False
                assert not torch.is_grad_enabled()
                self.calls.append(tuple(input_ids.shape))
                # Padding has a distinctive vector. Pooling the final padded
                # position instead of the final real token would fail this test.
                values = torch.stack((input_ids.float(), torch.ones_like(input_ids).float()), dim=-1)
                return SimpleNamespace(last_hidden_state=values)

        value = backend.TorchBackend.__new__(backend.TorchBackend)
        value._torch = torch
        value.device = torch.device("cpu")
        value.tokenizer = TinyTokenizer()
        value.encoder = TinyEncoder()
        value.head = lambda vectors: vectors @ torch.tensor([[1.0], [2.0]])
        value.max_batch_tokens = 32
        value.max_batch_size = batch_size
        value.context_limit = 20
        value._forward_lock = threading.Lock()
        return value

    def test_last_nonpadding_pooling_normalization_and_order(self):
        value = self.make_backend()
        texts = ["abcd", "z", "mn"]
        actual = value.score_pairs(texts)
        expected = []
        for text in texts:
            last_token = ord(text[-1]) % 20 + 1
            expected.append((last_token + 2) / (last_token**2 + 1)**0.5)
        self.assertEqual(value.encoder.calls, [(3, 5)])
        for a, e in zip(actual, expected, strict=True):
            self.assertAlmostEqual(a, e, places=6)

    def test_batched_and_singleton_match_for_synthetic_encoder(self):
        texts = ["abcd", "z", "mn"]
        batched = self.make_backend(3).score_pairs(texts)
        single = self.make_backend(1)
        strict_scores = single.score_pairs(texts)
        self.assertEqual(len(single.encoder.calls), 3)
        for a, b in zip(batched, strict_scores, strict=True):
            self.assertAlmostEqual(a, b, places=6)

    def test_validate_all_before_encoder_work(self):
        value = self.make_backend()
        with self.assertRaisesRegex(ValueError, "encoder limit"):
            value.score_pairs(["ok", "x" * 21])
        self.assertEqual(value.encoder.calls, [])
        self.assertEqual(value.score_pairs([]), [])
        self.assertEqual(value.count_tokens("ab"), 3)
        with self.assertRaises(ValueError):
            value.score_pairs([""])
        with self.assertRaises(TypeError):
            value.score_pairs("not a list")

    def test_nonfinite_score_fails(self):
        value = self.make_backend()
        value.head = lambda vectors: self.torch.full((len(vectors), 1), float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Non-finite ranking score"):
            value.score_pairs(["valid"])


if __name__ == "__main__":
    unittest.main()
