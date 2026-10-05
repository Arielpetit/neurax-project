"""
Fine-tuning a published model, inside the harness's own loop.

`train.py` trains what `model.py` builds; for a fine-tuning (`fineTune` in the
request) it trains the published model instead, built here by the libraries
that published it (`transformers`, `peft`), from the folder the service
filled and checked — offline: nothing reaches the network during a run.

Before a weight is read, every file is checked against the hash the
repository published; a file that is not the published one stops the run
before its first step. The model is wrapped so the loop feeds it as it feeds
any design: token ids or images in, logits out. Checkpoints hold what is
trained, not the published weights the client already has; the export is the
adapter beside the merged model, both loadable with `transformers`.
"""
import copy
import hashlib
import json
import os
from pathlib import Path


class Refused(Exception):
    """A fine-tuning that cannot run as asked, with the reason."""


def offline():
    """No network for the rest of the run: the weights are on disk, checked."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _hash(path, kind):
    if kind == "sha256":
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_files(ft):
    """Every file the fine-tuning takes is there, of its published size and hash."""
    folder = Path(ft["weightsDir"])
    for entry in ft.get("files") or []:
        path = folder / entry["path"]
        if not path.is_file():
            raise Refused(f"{entry['path']} is missing from {folder}: the weights must be downloaded before the run")
        if entry.get("size") and path.stat().st_size != int(entry["size"]):
            raise Refused(f"{entry['path']} is not the published file: {path.stat().st_size} bytes, "
                          f"the repository says {entry['size']}")
        kind = "sha256" if entry.get("sha256") else "gitOid" if entry.get("gitOid") else None
        if kind is None:
            raise Refused(f"{entry['path']} has no published hash to be checked against")
        found = _hash(path, kind)
        if found.lower() != str(entry[kind]).lower():
            raise Refused(f"{entry['path']} is not the published file: its hash is {found}, "
                          f"the repository says {entry[kind]}")


def _pickle_only(folder):
    names = {p.name for p in Path(folder).iterdir()}
    return not any(n.endswith(".safetensors") for n in names) and any(n.endswith(".bin") for n in names)


def head_of(model):
    """The task head: the model's top-level parts with weights other than its
    base (`base_model_prefix`) — `pre_classifier` and `classifier`, `score`…"""
    prefix = getattr(model, "base_model_prefix", None)
    return [name for name, child in model.named_children()
            if name != prefix and any(True for _ in child.parameters())]


_TOKENIZERS = {}


def tokenizer(req):
    """The published tokenizer, from the weights folder."""
    folder = req["fineTune"]["weightsDir"]
    if folder not in _TOKENIZERS:
        offline()
        from transformers import AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(folder)
        if tok.pad_token_id is None:
            # GPT-2 and its kin pad with their end-of-text token.
            tok.pad_token = tok.eos_token
        _TOKENIZERS[folder] = tok
    return _TOKENIZERS[folder]


def image_statistics(req):
    """`(mean, std)` per channel, on the loader's scale (pixels / 255), from
    the published processor: the weights were trained on its values, not on
    the data's. `preprocessor_config.json` missing is a refusal, never a guess."""
    path = Path(req["fineTune"]["weightsDir"]) / "preprocessor_config.json"
    if not path.is_file():
        raise Refused(f"preprocessor_config.json is missing from {path.parent}: an image model is fed as its "
                      "published processor says, and without it NEURAX would have to invent its normalization")
    config = json.loads(path.read_text())
    rescale = float(config.get("rescale_factor", 1 / 255)) if config.get("do_rescale", True) else 1.0
    if config.get("do_normalize", True):
        mean, std = config.get("image_mean"), config.get("image_std")
        if mean is None or std is None:
            raise Refused(f"{path} normalizes images but states no image_mean or image_std")
    else:
        mean, std = [0.0], [1.0]
    # The loader gives pixels / 255; the processor computes (pixels · r − m) / s,
    # which is (pixels / 255 − m / (255 r)) / (s / (255 r)).
    unit = 255.0 * rescale
    return [float(m) / unit for m in mean], [float(v) / unit for v in std]


def warm_start_differences(model, folder, torch):
    """`(weights, why not)` for a run starting from an earlier NEURAX run's
    export: its weights when this design takes them, strictly, else why not —
    a parameter missing, extra or of another shape, the first three named.
    Verification and the run both ask, so a changed design is refused before
    approval."""
    folder = Path(folder)
    chosen = next((folder / n for n in ("best.safetensors", "last.safetensors", "best.pt", "last.pt") if (folder / n).is_file()), None)
    if chosen is None:
        return None, f"{folder} holds no weights to start from (best.pt or last.pt): choose the export folder of a finished run"
    if chosen.suffix == ".safetensors":
        from safetensors.torch import load_file  # noqa: PLC0415

        weights = load_file(str(chosen))
    else:
        weights = torch.load(chosen, map_location="cpu", weights_only=True)
    own = model.state_dict()
    differences = [f"{k} is missing from the export" for k in own if k not in weights]
    differences += [f"{k} is in the export but not in this design" for k in weights if k not in own]
    differences += [f"{k} is {list(own[k].shape)} here and {list(weights[k].shape)} in the export"
                    for k in own if k in weights and tuple(own[k].shape) != tuple(weights[k].shape)]
    if not differences:
        return weights, None
    return None, (f"this design is not the one {folder} was trained with: " + "; ".join(differences[:3])
                  + (f"; and {len(differences) - 3} more" if len(differences) > 3 else ""))


def build(req, device, torch):
    """`(model, info)`: the published model with the method applied, wrapped
    to take what the loop feeds; `info` says what was built."""
    import torch.nn as nn  # noqa: PLC0415

    ft = req["fineTune"]
    offline()
    check_files(ft)
    folder = ft["weightsDir"]
    pickle = _pickle_only(folder)
    if pickle:
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        if (major, minor) < (2, 6):
            raise Refused("these weights are a pickle file (.bin), read safely only with PyTorch 2.6 or newer "
                          f"(this is {torch.__version__}): prepare the training environment again")
    from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification  # noqa: PLC0415

    task = ft.get("task", "sequence-classification")
    classes = {"sequence-classification": AutoModelForSequenceClassification,
               "image-classification": AutoModelForImageClassification}
    if task not in classes:
        raise Refused(f"fine-tuning for `{task}` is not available yet")
    published = classes[task].from_pretrained(
        folder, num_labels=int(req["numClasses"]), ignore_mismatched_sizes=True, use_safetensors=not pickle)
    head = head_of(published)
    if task == "image-classification":
        image_statistics(req)  # refused here, before any step, when it cannot be followed
    pad = None
    if task == "sequence-classification":
        pad = tokenizer(req).pad_token_id
        published.config.pad_token_id = pad
    method = ft.get("method", "lora")
    if method == "lora":
        import peft  # noqa: PLC0415

        settings = ft.get("lora") or {}
        saved = list(settings.get("modulesToSave") or head)
        model = peft.get_peft_model(published, peft.LoraConfig(
            r=int(settings.get("rank", 8)), lora_alpha=int(settings.get("alpha", 16)),
            lora_dropout=float(settings.get("dropout", 0.0)), target_modules=settings.get("targetModules", "all-linear"),
            modules_to_save=saved, use_rslora=settings.get("scaling") == "rs"))
    elif method == "head":
        model = published
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.split(".", 1)[0] in head
    elif method == "full":
        model = published
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise Refused(f"unknown method `{method}`")
    if req.get("gradientCheckpointing"):
        # Frozen weights carry no gradient: without input gradients enabled,
        # recomputation fails with "does not require grad".
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    class Published(nn.Module):
        """The published model, fed as the loop feeds a design."""

        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            if pad is not None:
                return self.model(input_ids=x, attention_mask=(x != pad).long()).logits
            return self.model(pixel_values=x).logits

    wrapped = Published().to(device)
    print(f"[neurax] fine-tuning {ft.get('source', {}).get('repo', folder)} by {method}: "
          f"{sum(p.numel() for p in wrapped.parameters() if p.requires_grad):,} trained", flush=True)
    return wrapped, {"head": head, "method": method, "pickle": pickle, "pad": pad}


def trained_state(model):
    """What a checkpoint keeps: the trained parameters only. The published
    weights are on the client's disk already, checked."""
    return {name: p.detach().to("cpu").clone() for name, p in model.named_parameters() if p.requires_grad}


def export(model, req, folder, which, tokenizer_folder=None):
    """`export/adapter` (LoRA) and `export/model`, the published model with
    what was trained merged into it, in the base's own precision — both as
    `transformers` saves them. `best` keeps the trained state; `last` writes
    the merged model from the best state when there is one."""
    published = model.model
    folder = Path(folder)
    method = req["fineTune"].get("method", "lora")
    if which == "best":
        _BEST[id(model)] = trained_state(model)
        if method == "lora":
            published.save_pretrained(folder / "adapter")
        return
    best = _BEST.get(id(model))
    if best is not None:
        model.load_state_dict(best, strict=False)
    merged = copy.deepcopy(published)
    if method == "lora":
        if best is None:
            published.save_pretrained(folder / "adapter")
        merged = merged.merge_and_unload()
    merged.save_pretrained(folder / "model", safe_serialization=True)
    source = Path(tokenizer_folder or req["fineTune"]["weightsDir"])
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "spiece.model", "sentencepiece.bpe.model", "tokenizer.model",
                 "preprocessor_config.json"):
        if (source / name).is_file():
            (folder / "model" / name).write_bytes((source / name).read_bytes())


_BEST = {}


def describe(req):
    """What the export's config records of the fine-tuning."""
    ft = req["fineTune"]
    return {"method": ft.get("method", "lora"), "source": ft.get("source"), "task": ft.get("task"),
            "weightsDir": ft.get("weightsDir"),
            "lora": ft.get("lora"), "files": [f["path"] for f in ft.get("files") or []]}


def write_log(run_dir, **fields):
    path = Path(run_dir) / "finetune.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2))
