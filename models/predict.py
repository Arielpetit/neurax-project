"""
Use the model NEURAX trained: a raw sample in, an answer out.

    from predict import Model

    model = Model()                      # the weights at the best validation loss
    model.predict({"age": 41, "city": "Lyon"})   # a table row → {"label", "probability", "scores"} or {"value"}
    model.predict("a sentence to classify")       # text → {"label", "probability", "scores"}
    model.predict("photo.jpg")                    # an image file → {"label", "probability", "scores"}
    model.generate("Once upon a time", max_new_tokens=40, temperature=0.8, top_k=40)  # a language model

Everything it needs is in this folder, written by the run that trained it:
`model.py` and the weights, `preprocessing.json` (how a raw sample becomes
what the model reads — fitted on the training part), `labels.json` (what each
output means) and `tokenizer.json` (how text becomes tokens). A sample is
prepared exactly as the run prepared its training data, and the output is
decoded back into the data's own terms.
"""

import importlib.util
import json
import math
import re
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _read(name):
    path = HERE / name
    return json.loads(path.read_text()) if path.exists() else None


class Model:
    def __init__(self, weights=None, device="cpu", adapter=False):
        self.config = _read("config.json") or {}
        self.plan = _read("preprocessing.json") or {}
        self.labels = _read("labels.json")
        self.device = torch.device(device)
        self._published = None
        # Set before any branch returns: a fine-tuned export returns early.
        self._graph = None
        if self.config.get("fineTune"):
            self._load_published(adapter)
            return
        spec = importlib.util.spec_from_file_location("neurax_model", HERE / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.net = getattr(module, self.config["modelClass"])().to(self.device)
        # A graph network reads the model's own `Graph` (see `predict`).
        self._graph = getattr(module, "Graph", None) if self.config.get("inputKind") == "graph" else None
        chosen = weights or ("best.pt" if (HERE / "best.pt").exists() else "last.pt")
        self.net.load_state_dict(torch.load(HERE / chosen, map_location=self.device, weights_only=True))
        self.net.eval()
        self._encode = self._decode = None
        if self.plan.get("kind") == "text":
            self._encode, self._decode = _text_codec()

    def _load_published(self, adapter):
        """A fine-tuned published model: `model/` with what was trained merged
        in — or, with `adapter=True`, the published model the run started from
        and `adapter/` on top of it."""
        from transformers import (  # noqa: PLC0415
            AutoModelForImageClassification, AutoModelForSequenceClassification, AutoTokenizer)

        fine = self.config["fineTune"]
        text = fine.get("task", "sequence-classification") == "sequence-classification"
        cls = AutoModelForSequenceClassification if text else AutoModelForImageClassification
        if adapter:
            import peft  # noqa: PLC0415

            base = cls.from_pretrained(fine["weightsDir"], num_labels=len(self.labels or []) or None,
                                       ignore_mismatched_sizes=True)
            net = peft.PeftModel.from_pretrained(base, HERE / "adapter")
        else:
            net = cls.from_pretrained(HERE / "model")
        self._published = net.to(self.device).eval()
        if text:
            self._tokenizer = AutoTokenizer.from_pretrained(HERE / "model")
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        # Images are prepared from `preprocessing.json`, which holds the published
        # processor's statistics: `AutoImageProcessor` needs torchvision, which a
        # training environment does not have.
        self.net = self._forward_published

    def _forward_published(self, inputs):
        return self._published(**inputs).logits

    # ── From a raw sample to what the model reads ────────────────────────────

    def _image(self, sample):
        """An image file as the run read it: resized, pixels / 255, then the
        mean and std of `preprocessing.json`."""
        from PIL import Image
        import numpy as np

        h, w = self.plan["size"]
        image = Image.open(sample).convert("RGB").resize((w, h))
        x = torch.from_numpy(np.asarray(image, dtype="float32")).permute(2, 0, 1)[: self.plan["channels"]] / 255.0
        mean, std = torch.tensor(self.plan["mean"]).view(-1, 1, 1), torch.tensor(self.plan["std"]).view(-1, 1, 1)
        return ((x - mean) / std).unsqueeze(0).to(self.device)

    def prepare(self, sample):
        if self._published is not None:
            if hasattr(self, "_tokenizer"):
                length = int(self.plan.get("sequenceLength") or (self.config.get("inputShape") or [128])[-1])
                encoded = self._tokenizer(str(sample), truncation=True, max_length=length, return_tensors="pt")
                return {k: v.to(self.device) for k, v in encoded.items()}
            return {"pixel_values": self._image(sample)}
        kind = self.plan.get("kind")
        if kind == "table":
            return torch.tensor([self._row(sample)], dtype=torch.float32, device=self.device)
        if kind == "text":
            length = int(self.plan["sequenceLength"])
            ids = self._encode(str(sample))[:length]
            return torch.tensor([ids + [0] * (length - len(ids))], dtype=torch.long, device=self.device)
        if kind == "image":
            return self._image(sample)
        raise ValueError(f"this folder's model reads {kind or 'an input'} that predict.py does not prepare")

    def _row(self, sample):
        features = self.plan["features"]
        if not isinstance(sample, dict):
            sample = {f["name"]: v for f, v in zip(features, sample)}
        values = []
        for f in features:
            raw = sample.get(f["name"])
            if f["kind"] == "category":
                text = "" if raw is None else str(raw).strip()
                value = float(f["categories"].index(text)) if text in f["categories"] else float(len(f["categories"]))
            else:
                try:
                    value = float(raw)
                    value = value if math.isfinite(value) else f["fill"]
                except (TypeError, ValueError):
                    value = f["fill"]  # an empty cell takes the training median, as in training
            values.append((value - f["mean"]) / f["std"])
        return values

    # ── From the model's output to an answer ────────────────────────────────

    @torch.no_grad()
    def predict(self, sample):
        if self._graph is not None:
            return self._predict_graph(sample)
        out = self.net(self.prepare(sample)).float()
        task = self.config.get("task")
        if task == "regression":
            target = self.plan.get("target") or {}
            value = float(out.reshape(-1)[0])
            return {"value": value * target.get("std", 1.0) + target.get("mean", 0.0)}
        if task == "language_modeling":
            raise ValueError("a language model writes text: use generate(prompt)")
        probabilities = torch.softmax(out.reshape(-1, out.shape[-1])[-1], dim=-1)
        names = self.labels or [str(i) for i in range(probabilities.shape[0])]
        best = int(probabilities.argmax())
        return {"label": names[best] if best < len(names) else str(best), "probability": float(probabilities[best]),
                "scores": {names[i] if i < len(names) else str(i): float(p) for i, p in enumerate(probabilities)}}

    def _predict_graph(self, sample):
        """A class for every node of a graph: a `.pt` file saved by torch.save, or
        a dict, holding `x` (nodes × features) and `edge_index` (2 × edges)."""
        data = torch.load(sample, map_location="cpu", weights_only=True) if isinstance(sample, (str, Path)) else sample
        x = data["x"].float().to(self.device)
        graph = self._graph(x, data["edge_index"].long().to(self.device),
                            torch.zeros(x.shape[0], dtype=torch.long, device=self.device), 1)
        out = self.net(graph)
        out = (out.x if hasattr(out, "_fields") else out).float()
        probabilities = torch.softmax(out, dim=-1)
        best = probabilities.argmax(dim=-1)
        names = self.labels or [str(i) for i in range(probabilities.shape[-1])]
        return {"labels": [names[i] if i < len(names) else str(i) for i in best.tolist()],
                "probabilities": probabilities.max(dim=-1).values.tolist()}

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=None):
        """Text after `prompt`, one token at a time: temperature 0 takes the most
        likely token; above, it samples among the `top_k` most likely (all when None)."""
        length = int(self.plan["sequenceLength"])
        if self.plan.get("targetLength"):
            return self._translate(prompt, length, int(self.plan["targetLength"]), max_new_tokens, temperature, top_k)
        ids = self._encode(prompt) or [1]
        for _ in range(max_new_tokens):
            window = ids[-length:]
            x = torch.tensor([window + [0] * (length - len(window))], dtype=torch.long, device=self.device)
            logits = self.net(x).float()[0, len(window) - 1]
            logits[0] = -float("inf")  # padding is never written
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k:
                    kept = torch.topk(logits, min(top_k, logits.shape[0])).values[-1]
                    logits[logits < kept] = -float("inf")
                token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            ids.append(token)
        return self._decode(ids)

    def _translate(self, source, source_len, target_len, max_new_tokens, temperature, top_k):
        """An encoder-decoder's answer to `source`: the decoder starts on the
        padding token, as the run trained it, and writes one token at a time."""
        ids = (self._encode(str(source)) or [1])[:source_len]
        src = torch.tensor([ids + [0] * (source_len - len(ids))], dtype=torch.long, device=self.device)
        written = [0]
        for _ in range(min(max_new_tokens, target_len - 1)):
            x = torch.tensor([written + [0] * (target_len - len(written))], dtype=torch.long, device=self.device)
            logits = self.net(src, x).float()[0, len(written) - 1]
            logits[0] = -float("inf")  # padding is never written
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k:
                    kept = torch.topk(logits, min(top_k, logits.shape[0])).values[-1]
                    logits[logits < kept] = -float("inf")
                token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            written.append(token)
        return self._decode(written[1:])


def _text_codec():
    """Encode and decode text as the run did, from its tokenizer.json."""
    data = json.loads((HERE / "tokenizer.json").read_text())
    if data.get("kind") == "words":
        index = data["vocabulary"]
        words = {i: w for w, i in index.items()}
        encode = lambda text: [index.get(w, data["unknown"]) for w in re.findall(data["pattern"], text.lower())]  # noqa: E731
        decode = lambda ids: " ".join(words.get(i, "?") for i in ids if i > 1)  # noqa: E731
        return encode, decode
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(HERE / "tokenizer.json"))
    return (lambda text: tokenizer.encode(text).ids), (lambda ids: tokenizer.decode([i for i in ids if i > 1]))
