"""Model loading that copes with qwen3_5 multimodal checkpoints (text-only use)."""

import torch
from transformers import AutoConfig, AutoTokenizer


def load_model(model_id, dtype=torch.float32, device="cuda", trainable=True,
               device_map=None):
    """Load as causal LM if possible, else the multimodal class (text-only fwd).

    Returns (model, tokenizer, n_layers). Weights in fp32 by default so that
    captured G/H are not quantized by bf16 master weights; forward runs under
    autocast(bf16) in the train loop.

    device_map="auto" enables naive model parallelism across all visible GPUs
    (layers sharded, optimizer states colocated with their shards). Identical
    math to single-GPU — used for 9B+ where fp32 + AdamW exceeds one card.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"dtype": dtype}
    if device_map:
        kwargs["device_map"] = device_map
    model = None
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception:
        from transformers import AutoModelForMultimodalLM  # transformers >= 5
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **kwargs)
    if not device_map:
        model = model.to(device)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None)
    if n_layers is None:  # multimodal wrapper: text config nested
        tc = getattr(cfg, "text_config", None)
        n_layers = getattr(tc, "num_hidden_layers")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok, n_layers


def decoder_param_groups(model, weight_decay=0.0):
    """Split params: 2D decoder matrices (Muon-eligible) vs everything else."""
    matrix, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.dim() == 2 and ("layers." in name) and ("embed" not in name)
                and "visual" not in name and not name.startswith("mtp.")):
            matrix.append(p)
        else:
            other.append(p)
    return matrix, other
