"""Protocol/export checks without loading a decoder or requiring CUDA."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
import unittest

from gemmadecision.export_vllm import export_vllm
from gemmadecision.backends.vllm import VLLMBackend


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.config = {"model_type": "gemma3_text", "architectures": ["Gemma3TextModel"],
                       "hidden_size": 640, "rope_parameters": {"full_attention": {"rope_theta": 1000000}}}
        (self.model / "config.json").write_text(json.dumps(self.config))
        (self.model / "joint_config.json").write_text(json.dumps({"format": "gemmadecision-full-joint-v4"}))
        # Byte fixtures verify export mechanics, not a fake successful weight load.
        (self.model / "model.safetensors").write_bytes(b"encoder bytes unchanged")
        (self.model / "joint_head.safetensors").write_bytes(b"head must not reach vllm loader")
        (self.model / "tokenizer_config.json").write_text("{}")
        (self.model / "tokenizer.json").write_text("{}")

    def test_encoder_only_export_leaves_source_unchanged(self):
        target = export_vllm(self.model, self.root / "export")
        self.assertEqual((target / "model.safetensors").read_bytes(),
                         (self.model / "model.safetensors").read_bytes())
        self.assertFalse((target / "joint_head.safetensors").exists())
        self.assertEqual(json.loads((self.model / "config.json").read_text()), self.config)
        expected = {**self.config, "architectures": ["Gemma3ForCausalLM"]}
        self.assertEqual(json.loads((target / "config.json").read_text()), expected)
        self.assertEqual(export_vllm(self.model, target), target)

    def test_rejects_modified_export_and_extra_head(self):
        target = export_vllm(self.model, self.root / "export")
        (target / "joint_head.safetensors").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "unexpected weight"):
            export_vllm(self.model, target)
        (target / "joint_head.safetensors").unlink()
        (target / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "integrity"):
            export_vllm(self.model, target)

    def test_refuses_source_destination_and_other_model(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            export_vllm(self.model, self.model)
        (self.model / "joint_config.json").write_text('{"format": "other"}')
        with self.assertRaisesRegex(ValueError, "full-joint"):
            export_vllm(self.model, self.root / "export")


class PoolingContractTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional Torch dependency is not installed")
        self.torch = torch
        self.backend = object.__new__(VLLMBackend)
        self.backend.context_limit = 10
        self.backend.hidden = 2
        self.backend._lock = Lock()
        self.backend._pooling_params = object()
        self.seen = []
        def tokenizer(text, **kwargs):
            self.assertEqual(kwargs, {"add_special_tokens": True, "truncation": False, "padding": False})
            return {"input_ids": [2] + [ord(c) for c in text]}
        self.backend.tokenizer = tokenizer
        def embed(prompts, **kwargs):
            self.seen.append((prompts, kwargs))
            return [SimpleNamespace(outputs=SimpleNamespace(embedding=[3., 4.])),
                    SimpleNamespace(outputs=SimpleNamespace(embedding=[0., 20.]))]
        self.backend.llm = SimpleNamespace(embed=embed)
        self.vectors = []
        def head(vectors):
            self.vectors.append(vectors)
            return vectors[:, :1]
        self.backend.head = head

    def test_exact_tokens_fp32_normalization_and_order(self):
        result = self.backend.score_pairs(["a", "b"])
        self.assertAlmostEqual(result[0], .6, places=6)
        self.assertEqual(result[1], 0.)
        self.assertEqual(self.seen[0][0], [{"prompt_token_ids": [2, 97]}, {"prompt_token_ids": [2, 98]}])
        self.assertFalse(self.seen[0][1]["use_tqdm"])
        self.assertEqual(self.vectors[0].dtype, self.torch.float32)
        self.assertTrue(self.torch.allclose(self.vectors[0].norm(dim=-1), self.torch.ones(2)))

    def test_limit_failure_happens_before_inference(self):
        with self.assertRaisesRegex(ValueError, "Joint input"):
            self.backend.score_pairs(["a", "a" * 20])
        self.assertEqual(self.seen, [])
        self.assertEqual(self.backend.score_pairs([]), [])

    def test_bad_embeddings_fail_closed(self):
        self.backend.llm.embed = lambda *a, **k: []
        with self.assertRaisesRegex(RuntimeError, "number"):
            self.backend.score_pairs(["a"])
        self.backend.llm.embed = lambda *a, **k: [SimpleNamespace(outputs=SimpleNamespace(embedding=[float("nan"), 1.]))]
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            self.backend.score_pairs(["a"])


if __name__ == "__main__":
    unittest.main()
