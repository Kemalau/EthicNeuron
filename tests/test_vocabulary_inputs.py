"""Offline checks of the native decoder, effective weights and external lens binding."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from discovery import decoder_layers, sha256
from methods.vocabulary_inputs import (LENSES, load_readout_model, load_transport,
                                      readout_modules, write_columns)
from test_methods import tiny_model


class ReadoutInputsTests(unittest.TestCase):
    def test_native_text_norm_and_original_wrapped_head(self):
        model = tiny_model()
        norm, head = readout_modules(model)
        self.assertIs(norm, model.get_base_model().model.norm)
        self.assertIs(head, model.get_base_model().lm_head)
        # The multimodal model's text norm uses its own scale convention.
        wrap = SimpleNamespace(original_module=head)
        core = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"),
                               model=SimpleNamespace(language_model=SimpleNamespace(norm=norm)),
                               get_output_embeddings=lambda: wrap)
        self.assertEqual(readout_modules(core), (norm, head))

    def test_preserve_saved_fp32_lora_factors(self):
        model = tiny_model()
        original = {name.replace(".default.", "."): p.detach().clone()
                    for name, p in model.named_parameters() if ".lora_" in name}
        model.bfloat16()
        with TemporaryDirectory() as tmp:
            save_file(original, str(Path(tmp) / "adapter_model.safetensors"))
            with patch("methods.vocabulary_inputs.load_model", return_value=(model, None)) as load:
                restored, _ = load_readout_model("toy", tmp, "cpu", write_weights="effective")
                self.assertEqual(load.call_args.kwargs["dtype"], torch.bfloat16)
        down = decoder_layers(restored)[0].mlp.down_proj
        prefix = "base_model.model.model.layers.0.mlp.down_proj."
        a = original[prefix + "lora_A.weight"]
        b = original[prefix + "lora_B.weight"]
        self.assertTrue(torch.equal(a, down.lora_A["default"].weight))
        ids = [5, 2]
        base_cols = down.base_layer.weight.detach().float()[:, ids]
        self.assertTrue(torch.equal(write_columns(restored, 0, ids, "base"), base_cols))
        expected = base_cols + down.scaling["default"] * (b @ a[:, ids])
        torch.testing.assert_close(write_columns(restored, 0, ids, "effective"), expected)
        self.assertEqual(readout_modules(restored)[1].weight.dtype, torch.bfloat16)

    def test_model_specific_transport_orientation_and_provenance(self):
        j = torch.tensor([[0, 2], [3, 0]], dtype=torch.float16)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "lens.pt"
            torch.save(dict(J={0: j}, source_layers=[0], d_model=2, n_prompts=7), path)
            spec = dict(LENSES["qwen2", "jacobian"], sha256=sha256(path))
            with patch.dict(LENSES, {("qwen2", "jacobian"): spec}):
                def load(layers=(0,), size=2, base=spec["model"]):
                    return load_transport(path, layers, size, "qwen2", "jacobian", base=base)
                maps, metadata = load()
                torch.testing.assert_close(maps[0] @ torch.tensor([1., 4.]), torch.tensor([8., 3.]))
                self.assertEqual(metadata["n_prompts"], 7)
                self.assertEqual(metadata["base_binding"], "hub_model_id")
                with self.assertRaisesRegex(ValueError, "published map expects"):
                    load(base="Qwen/Qwen2.5-7B")
                with self.assertRaisesRegex(ValueError, "verified base revision"):
                    load_transport(path, [0], 2, "qwen2", "jacobian",
                                   base=spec["model"], base_revision="other")
                with self.assertRaisesRegex(ValueError, "layer 1"):
                    load(layers=[1])
                with self.assertRaisesRegex(ValueError, "hidden dimension"):
                    load(size=3)
                path.write_bytes(b"another lens")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load()

    def test_local_copied_shards_binding(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / "lens.pt"
            torch.save(dict(J={0: torch.eye(2)}, d_model=2, n_prompts=1), path)
            shard = directory / "model.safetensors"
            shard.write_bytes(b"fixture weight shard")
            spec = dict(LENSES["qwen2", "jacobian"], sha256=sha256(path),
                        shard_hashes=(sha256(shard),))
            with patch.dict(LENSES, {("qwen2", "jacobian"): spec}):
                _, info = load_transport(path, [0], 2, "qwen2", "jacobian", base=directory)
                self.assertEqual(info["base_binding"], "local_weight_hashes")
                shard.write_bytes(b"different checkpoint")
                with self.assertRaisesRegex(ValueError, "Local checkpoint"):
                    load_transport(path, [0], 2, "qwen2", "jacobian", base=directory)


if __name__ == "__main__":
    unittest.main()
