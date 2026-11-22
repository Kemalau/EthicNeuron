"""Checkpoint inputs for the published layer transports and static vocabulary readout."""
from pathlib import Path

import torch

from discovery import decoder_layers, load_model, sha256
from matched_controls import effective_columns


LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "d2527c3cdbaa60dd4b65cbca1604921f044e989a"

# Each record binds a published map to its fitting model. Standard Jacobian
# files store dimensions and layer numbers but omit the model name.
LENSES = {
    ("qwen2", "jacobian"): dict(
        model="Qwen/Qwen2.5-7B-Instruct",
        revisions={"Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28"},
        filename="qwen2.5-7b-it/jlens/Salesforce-wikitext/Qwen2.5-7B-Instruct_jacobian_lens.pt",
        sha256="3b3ab44cd67c2ad1f26e9f66eb269db31289b6328585b0459a47e6d3814cba29",
        shard_hashes=(
            "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
            "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
            "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
            "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
        ),
    ),
    ("llama", "jacobian"): dict(
        model="meta-llama/Llama-3.1-8B-Instruct",
        revisions={
            "meta-llama/Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
            "NousResearch/Meta-Llama-3.1-8B-Instruct": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        },
        filename="llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
        sha256="d5300a1575c0b5ad6e01b098c75adb072f3a6c246f184ce2fb35a3550fc8c68b",
        shard_hashes=(
            "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
            "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
            "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
            "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
        ),
    ),
    ("qwen3_5", "jpp"): dict(
        model="Qwen/Qwen3.5-9B",
        revisions={
            "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "unsloth/Qwen3.5-9B": "005429cee5cb648998cf2b70eebdd83175989c9a",
        },
        filename="qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt",
        sha256="8cda12f7ddf11e70fcfc69a47a9574f626fe084c67807a4781f4f67106e739c5",
        shard_hashes=(
            "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
            "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
            "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
            "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
        ),
    ),
}


def _base_binding(base, revision, spec):
    """Match a Hub ID, cached snapshot, local Hub download or copied weights."""
    path = Path(base)
    if not path.is_dir():
        if str(base) in spec["revisions"]:
            if revision is not None and revision != spec["revisions"][str(base)]:
                raise ValueError(f"Use the verified base revision for {base}: {spec['revisions'][str(base)]}")
            return "hub_model_id"
        raise ValueError(f"The published map expects {spec['model']}, got {base}")
    path = path.resolve()
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        repo = path.parent.parent.name.removeprefix("models--").replace("--", "/")
        if repo not in spec["revisions"]:
            raise ValueError(f"The published map expects {spec['model']}, got {repo}")
        if revision is not None and revision != path.name:
            raise ValueError("Discovery base revision differs from the local snapshot")
        if path.name == spec["revisions"][repo]:
            return "hub_snapshot"
    metadata = path / ".cache/huggingface/download/config.json.metadata"
    known = spec["revisions"].values()
    if metadata.is_file():
        commit = metadata.read_text().partition("\n")[0]
        if commit in known and (revision is None or revision == commit):
            return "local_hub_metadata"
    # For a renamed standalone directory, filenames and dimensions alone do
    # not distinguish post-trained weights from base weights.
    shards = sorted(path.glob("*.safetensors"))
    if (len(shards) == len(spec["shard_hashes"]) and
            all(sha256(file) == expected for file, expected in zip(shards, spec["shard_hashes"]))):
        return "local_weight_hashes"
    raise ValueError(f"Local checkpoint could not be bound to {spec['model']}")


def load_transport(path, layers, hidden_size, model_type, lens_kind, *, base=None, base_revision=None):
    """Load the published maps on CPU in [final-hidden, layer-hidden] orientation."""
    spec = LENSES.get((model_type, lens_kind))
    if spec is None:
        raise ValueError(f"No published {lens_kind} map packaged for model type {model_type}")
    binding = _base_binding(base, base_revision, spec) if base is not None else "architecture_only"
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Lens SHA-256 mismatch for {spec['filename']}")
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if data["d_model"] != hidden_size:
        raise ValueError("Lens hidden dimension differs from the decoder")
    matrices = {}
    for layer in layers:
        matrix = data["J"].get(layer)
        if matrix is None or matrix.shape != (hidden_size, hidden_size):
            raise ValueError(f"No matching transport matrix for layer {layer}")
        matrices[layer] = matrix.float()
    metadata = dict(repo=LENS_REPO, revision=LENS_REVISION, filename=spec["filename"],
                    sha256=spec["sha256"], kind=lens_kind, fitted_model=spec["model"],
                    verified_base_revisions=spec["revisions"],
                    n_prompts=data["n_prompts"], base_binding=binding)
    return matrices, metadata


def _original_module(module):
    if getattr(module, "merged", False):
        raise ValueError("Vocabulary readout requires unmerged model weights")
    if hasattr(module, "original_module"):
        return module.original_module
    if hasattr(module, "get_base_layer"):
        return module.get_base_layer()
    return module


def readout_modules(model):
    """Original model's final text normalization and unembedding, including tied heads."""
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = core.model
    if core.config.model_type in {"qwen3_5", "gemma3"}:
        decoder = decoder.language_model
    return _original_module(decoder.norm), _original_module(core.get_output_embeddings())


def write_columns(model, layer, neurons, write_weights):
    """Selected down-projection columns in [hidden, selected channel] order in FP32."""
    if write_weights not in {"base", "effective"}:
        raise ValueError("Write weights must be base or effective")
    layers = decoder_layers(model)
    if type(layer) is not int or not 0 <= layer < len(layers):
        raise ValueError(f"Invalid layer {layer}")
    down = layers[layer].mlp.down_proj
    if hasattr(down, "modules_to_save"):
        if write_weights == "effective":
            raise ValueError("Effective write columns require additive LoRA weights")
        down = _original_module(down)
    return effective_columns(down, neurons, "base" if write_weights == "base" else "medical")


def load_readout_model(base, adapter, device, base_revision=None, adapter_revision=None,
                       local_files_only=False, *, write_weights="base"):
    """Load the BF16 decoder and preserve stored LoRA precision for effective writes."""
    if write_weights not in {"base", "effective"}:
        raise ValueError("Write weights must be base or effective")
    model, tokenizer = load_model(base, adapter, device, dtype=torch.bfloat16,
                                  base_revision=base_revision, adapter_revision=adapter_revision,
                                  local_files_only=local_files_only)
    if write_weights == "base":
        return model, tokenizer

    # Discovery uses a BF16-loaded adapter. Eq. 7's effective column uses the
    # checkpoint's stored A and B factors, composed in FP32.
    from safetensors import safe_open

    path = Path(adapter) / "adapter_model.safetensors"
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(adapter, path.name, revision=adapter_revision,
                               local_files_only=local_files_only)
    with safe_open(path, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name, module in model.named_modules():
            if not name.endswith(".mlp.down_proj") or not hasattr(module, "lora_A"):
                continue
            for side in ("lora_A", "lora_B"):
                key = f"{name}.{side}.weight"
                # Some text-only adapters for multimodal Qwen use model.layers.
                if key not in keys:
                    key = key.replace(".model.language_model.layers.", ".model.layers.")
                if key not in keys:
                    raise ValueError(f"Missing checkpoint factor {key}")
                parameter = getattr(module, side)["default"].weight
                tensor = source.get_tensor(key)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"Checkpoint shape mismatch for {key}")
                getattr(module, side)["default"].weight = torch.nn.Parameter(
                    tensor.to(device=parameter.device, dtype=torch.float32), requires_grad=False)
    return model, tokenizer
