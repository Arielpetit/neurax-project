"""
The harness that trains a NEURAX design.

Written to be read. Anyone debugging a run will open this file inside the run
directory it was copied into, next to the `model.py` it trains and the
`steps.jsonl` it wrote, and should be able to follow what happened without
consulting anything else.

Three properties matter more than speed here:

  * **It reports the real parameter count before the first step.** That single
    number confronts eleven IR phases of prediction in about a second, with no
    GPU time spent. If it disagrees with what NEURAX computed, a formula is
    wrong, and finding that out now costs nothing — so it is written first,
    and the studio compares it immediately.

  * **It stops at a step boundary, never inside one.** Pause and stop arrive
    as a file, read between steps. Stopping mid-kernel would leave memory
    allocated, a half-applied optimizer update, and no checkpoint.

  * **It says what went wrong.** Every failure path writes `status: failed`
    with the exception into `state.json`, because a run that simply stops
    reports the same thing as one that was killed, and those need different
    responses from the user.
"""

import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent

#: Whether this process writes the run's files: always, but for the second and
#: later processes of a multi-GPU run.
MAIN = True

#: This process's share of a multi-GPU run: `(rank, processes)`. Every process
#: shuffles an epoch the same way and reads every `processes`-th row from its
#: rank, so together they read the epoch once — not each the whole of it.
SHARD = (0, 1)
#: Epochs of shuffling started so far, the same count in every process.
_EPOCHS = {"n": 0}


def _epoch_order(count):
    """A new shuffle of `count` rows for the next epoch, identical in every process
    of a run (seeded by the run's seed and the epoch), cut to this process's share."""
    _EPOCHS["n"] += 1
    order = list(range(count))
    random.Random(_EPOCHS["n"] * 1_000_003 + SEED["value"]).shuffle(order)
    rank, processes = SHARD
    return order[rank::processes]


#: The run's seed, for the shuffles above.
SEED = {"value": 0}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads((RUN_DIR / path).read_text())
    except Exception:
        return default


def write_json(path, payload):
    # Written and renamed, so the studio never reads a half-written file. It
    # polls these while training is writing them.
    tmp = RUN_DIR / (path + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(RUN_DIR / path)


_last_state_write = 0.0


def update_state(force=False, **fields):
    """
    Publish where the run has got to.

    Throttled, and that is not premature optimisation. The first version wrote
    and renamed `state.json` on every step: a training loop that manages a
    thousand steps a second would issue a thousand writes and a thousand
    renames a second, and the run would spend more time publishing its
    progress than making any. Every half-second is far finer than a human
    reads, and the studio's own polling is slower than that anyway.

    A status change is never throttled — `paused`, `finished` and `failed` are
    the transitions everything downstream waits on, and delaying one by half a
    second is how a stop request looks ignored.
    """
    global _last_state_write
    if not MAIN:
        return None
    is_status_change = "status" in fields
    if not (force or is_status_change) and time.monotonic() - _last_state_write < 0.5:
        return None

    state = read_json("state.json", {}) or {}
    state.update(fields)
    state["lastHeartbeat"] = now()
    write_json("state.json", state)
    _last_state_write = time.monotonic()
    return state


def fail(message, trace=None):
    """End the run as failed. `message` is what the client reads; a Python
    trace, when there is one, is kept apart in `trace` (and the log)."""
    update_state(status="failed", error=message, **({"trace": trace} if trace else {}))
    print(f"[neurax] failed: {message}", file=sys.stderr)
    if trace:
        print(trace, file=sys.stderr)
    sys.exit(1)


def fail_on_exception(sentence):
    """`fail` for the exception being handled: the sentence, then what went
    wrong in the trace's own last line, in brackets."""
    trace = traceback.format_exc()
    if _disk_full(sys.exc_info()[1]):
        import shutil

        try:
            free = shutil.disk_usage(RUN_DIR).free
            left = f"{free / 1e6:,.0f} MB free" if free < 1e9 else f"{free / 1e9:,.1f} GB free"
        except OSError:
            left = "free space unknown"
        message = (f"The disk holding this run is full ({left} where {RUN_DIR} is): free some space, then resume "
                   "the run, which picks up from its last checkpoint. The full trace is in the run's log.")
        # Interrupted, not failed: nothing is wrong with the run, and an
        # interrupted run is the one the studio and the service resume.
        update_state(status="interrupted", error=message, trace=trace)
        print(f"[neurax] interrupted: {message}", file=sys.stderr)
        sys.exit(1)
    cause = next((line.strip() for line in reversed(trace.splitlines()) if line.strip()), "no detail")
    fail(f"{sentence} ({cause}). The full trace is in the run's log.", trace)


def _disk_full(exc):
    """Whether an exception, or one it was raised from, is a full disk: an
    OSError 28, or what torch.save says when the disk fills under it."""
    seen = 0
    while exc is not None and seen < 8:
        if isinstance(exc, OSError) and exc.errno == 28:
            return True
        text = str(exc)
        if "No space left on device" in text or ("inline_container" in text and "unexpected pos" in text):
            return True
        exc, seen = exc.__cause__ or exc.__context__, seen + 1
    return False


def cpu_threads(req, torch):
    """The threads a CPU run computes with: the request's own count, else one
    fewer than PyTorch would take (one per physical core), never below one —
    a core left to the studio and the rest of the machine."""
    asked = int(req.get("cpuThreads") or 0)
    return asked if asked > 0 else max(1, torch.get_num_threads() - 1)


def _load_published_parts(model):
    """Refuse to train when a model component requires unavailable published weights."""
    for part in model.modules():
        load = getattr(part, "load_published", None)
        if not callable(load):
            continue
        try:
            loaded = load()
        except Exception as exc:  # noqa: BLE001
            fail(f"could not load published weights for {type(part).__name__}: {exc}")
        if loaded is not True:
            detail = "reported that loading failed" if loaded is False else "did not confirm a successful load"
            fail(f"{type(part).__name__} {detail}; refusing to train without its published weights")


# ── Data ────────────────────────────────────────────────────────────────────


def split(count, fraction, seed=0, labels=None, groups=None):
    """Indices of the training and validation parts of `count` samples.

    The same every run and on every machine: a fixed shuffle, `fraction` held
    out (at most half). With no fraction nothing is held out.

    With `labels`, each class is held out in its own proportion, so a rare
    class is validated rather than left to chance. With `groups`, samples of
    one group (a row and its duplicate) fall on the same side: a duplicate on
    each side measures memory, not learning. Without either, the split is the
    plain shuffle it always was."""
    share = max(0.0, min(0.5, float(fraction or 0.0)))
    if labels is None and groups is None:
        order = list(range(count))
        random.Random(seed).shuffle(order)
        held = int(round(count * share))
        if held == 0 or held >= count:
            return order, []
        return order[held:], order[:held]
    keys = list(groups) if groups is not None else list(range(count))
    members = {}
    for i, key in enumerate(keys):
        members.setdefault(key, []).append(i)
    by_class = {}
    for key, rows in members.items():
        by_class.setdefault(labels[rows[0]] if labels is not None else None, []).append(key)
    rng = random.Random(seed)
    held_keys = set()
    for label in sorted(by_class, key=str):
        kept = sorted(by_class[label], key=str)
        rng.shuffle(kept)
        held = int(round(len(kept) * share))
        if share > 0 and held == 0 and len(kept) >= 2 and labels is not None:
            held = 1  # a class of a few samples still gets one validated
        held_keys.update(kept[:held])
    val_idx = sorted(i for key in held_keys for i in members[key])
    if not val_idx or len(val_idx) >= count:
        return list(range(count)), []
    held_set = set(val_idx)
    return [i for i in range(count) if i not in held_set], val_idx


#: The parts a dataset's own split names, and the part each stands for.
SPLIT_NAMES = {"train": "train", "training": "train", "val": "validation", "valid": "validation",
               "validation": "validation", "dev": "validation", "test": "test", "testing": "test"}


#: Splits computed this run, by what they were computed from (`_part`).
_SPLITS = {}
#: Datasets read this run, by file identity: each evaluation read the whole
#: file again — on a 1 GB file, a minute and a half every epoch.
_DATA = {}


def _read_once(kind, path, read):
    """`read()` the first time this run asks for `path` as `kind`, then its result."""
    stat = Path(path).stat()
    key = (kind, str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DATA:
        _DATA[key] = read()
    return _DATA[key]


def _same_sample(group):
    """The key two rows share when they are one sample to a model: text is
    compared lowercased with its spacing collapsed."""
    return " ".join(group.lower().split()) if isinstance(group, str) else group


def _part(items, req, part, labels=None, groups=None, given=None):
    """The rows of `items` in the requested part, recorded in `split.json`.

    `given` is the part each row already belongs to, when the data says it
    (`train`/`validation` folders, a `split` column): it is kept as it is, and
    a validation row that is also in training is counted and said — it would
    measure memory, not learning. Rows of a `test` part are neither trained nor
    validated on."""
    if groups is not None:
        groups = [_same_sample(g) for g in groups]
    leaked = None
    if given is not None:
        train_idx = [i for i, g in enumerate(given) if g == "train"]
        val_idx = [i for i, g in enumerate(given) if g == "validation"]
        if groups is not None:
            seen = {groups[i] for i in train_idx}
            leaked = sum(1 for i in val_idx if groups[i] in seen)
    else:
        # Computed once a run: every evaluation asked again, re-sorting every
        # row of the dataset each epoch. The key is the content, not the lists.
        key = (len(items), req.get("validationFraction"), seed_of(req),
               hash(tuple(labels)) if labels is not None else None,
               hash(tuple(groups)) if groups is not None else None)
        if key not in _SPLITS:
            _SPLITS[key] = split(len(items), req.get("validationFraction"), seed_of(req), labels=labels, groups=groups)
        train_idx, val_idx = _SPLITS[key]
    _record_split(train_idx, val_idx, labels, given is not None, groups is not None, leaked)
    chosen = val_idx if part == "validation" else train_idx
    return [items[i] for i in chosen]


def _record_split(train_idx, val_idx, labels, given, grouped, leaked=None):
    """Which rows were held out, written beside the run (and exported with the model)."""
    if not MAIN:
        return
    warnings = []
    if leaked:
        warnings.append(f"{leaked} validation row{'s are' if leaked != 1 else ' is'} also in training: "
                        "the validation accuracy is higher than what the model really learned")
        print(f"[neurax] {warnings[-1]}", flush=True)
    if labels is not None and val_idx:
        trained = {str(labels[i]) for i in train_idx}
        validated = {str(labels[i]) for i in val_idx}
        missing = sorted(trained - validated)
        if missing:
            warnings.append(f"class {', '.join(missing[:10])} has too few samples to be validated: its accuracy is not measured")
            print(f"[neurax] {warnings[-1]}", flush=True)
    record = {"train": len(train_idx), "validation": len(val_idx), "stratified": labels is not None and not given,
              "givenByTheData": given, "duplicatesKeptTogether": grouped, "validationRows": val_idx, "warnings": warnings}
    if leaked is not None:
        record["leakedRows"] = leaked
    try:
        write_json("split.json", record)
    except OSError:
        pass


def load_batches(req, input_shape, device, torch, part="train", once=False):
    """
    Yield `(inputs, targets)` forever — or, with `once`, a single pass, which
    is how the validation part is evaluated.

    `part` is `train` or `validation`: the validation share the dataset
    profile reports (`validationFraction`) is held out of training and only
    ever evaluated.

    The user's own data, in the form the design's first layer takes. A
    folder of images is read with PIL directly rather than through
    torchvision, which is not installed on every machine and would turn a
    missing optional dependency into a run that cannot start. A table is read
    as numbers for a design that takes features, and as text for one that
    takes tokens (`inputKind`). Anything else is refused: a run never trains
    on random tensors.
    """
    batch = int(req["batchSize"])
    path = req.get("datasetPath")
    num_classes = max(1, int(req.get("numClasses") or 1))
    needs_context = any(extra.get("name") == "context" for extra in (req.get("extras") or []))
    if needs_context and not (path and Path(path).is_file() and Path(path).suffix.lower() == ".pt"
                              and req.get("inputKind") == "image"):
        fail("this conditional image model needs caption embeddings: choose a .pt dataset with images and context tensors")

    if path and Path(path).is_dir():
        from PIL import Image

        root = Path(path)
        tops = [d for d in root.iterdir() if d.is_dir()]
        # `train/`, `val/` (and `test/`) folders are the dataset's own parts, each
        # holding the class folders: the parts, not classes.
        parts = {d: SPLIT_NAMES[d.name.lower()] for d in tops} if tops and all(d.name.lower() in SPLIT_NAMES for d in tops) else {root: None}
        classes = sorted({c.name for d in parts for c in d.iterdir() if c.is_dir()})
        if WARM_START:
            classes = _warm_classes({}, classes, False)
        _require_class_outputs(classes, req.get("numClasses"))
        files, given = [], []
        for folder, named in parts.items():
            for label, name in enumerate(classes):
                if not (folder / name).is_dir():
                    continue
                for f in sorted((folder / name).iterdir()):
                    if f.suffix.lower() in IMAGE_SUFFIXES:
                        files.append((f, label))
                        given.append(named)
        if not files:
            fail(f"no readable images under {path}")
        named = given if given and given[0] is not None else None
        trained = _part(files, req, "train", labels=[label for _, label in files], given=named)
        files = _part(files, req, part, labels=[label for _, label in files], given=named)
        if not files:
            return

        if len(input_shape) < 4:
            fail(
                f"this design takes {input_shape[1:]} per sample, which is not an image shape — "
                f"a folder of images cannot be fed to it"
            )
        c, h, w = input_shape[-3], input_shape[-2], input_shape[-1]
        print(f"[neurax] {len(files)} images, {len(classes)} classes", flush=True)

        def read(f):
            img = Image.open(f).convert("RGB").resize((w, h))
            return torch.from_numpy(_to_array(img)).permute(2, 0, 1).float()[:c] / 255.0

        # Each channel standardized by the training part's own statistics (a sample
        # of it, at the model's size), so the first layer sees values around zero.
        key = ("image", str(root), h, w, c)
        if key not in _ENCODERS:
            sample = []
            for f, _ in trained[:: max(1, len(trained) // 256)][:256]:
                try:
                    sample.append(read(f))
                except Exception:  # noqa: BLE001
                    continue
            stacked = torch.stack(sample) if sample else torch.zeros(1, c, h, w)
            mean = stacked.transpose(0, 1).reshape(c, -1).mean(1)
            std = stacked.transpose(0, 1).reshape(c, -1).std(1).clamp_min(1e-3)
            fitted = "training part"
            earlier = _warm_plan("image")
            if earlier is not None:
                # Started from an earlier run: its normalization, which its weights learnt.
                mean, std = torch.tensor(earlier["mean"]), torch.tensor(earlier["std"])
                fitted = f"the run it starts from ({WARM_START['dir']})"
            elif FINE_TUNE:
                # A published model is fed as its processor says, not by the data's statistics.
                import finetune  # noqa: PLC0415

                published_mean, published_std = finetune.image_statistics(FINE_TUNE)
                mean = torch.tensor((published_mean * c)[:c] if len(published_mean) == 1 else published_mean[:c])
                std = torch.tensor((published_std * c)[:c] if len(published_std) == 1 else published_std[:c])
                fitted = "the published processor"
            weights = _class_weights([label for _, label in trained], max(num_classes, len(classes)))
            _ENCODERS[key] = (mean, std)
            if MAIN:
                write_json("labels.json", classes)
                write_json("preprocessing.json", {
                    "kind": "image", "size": [h, w], "channels": c, "scale": "pixels / 255", "resize": "stretched to size",
                    "mean": mean.tolist(), "std": std.tolist(), "augment": bool(req.get("augment")),
                    "fittedOn": fitted, **({"classWeights": weights} if weights else {})})
        mean, std = _ENCODERS[key]
        flips = bool(req.get("augment")) and part == "train" and not once

        while True:
            epoch = files if once else [files[i] for i in _epoch_order(len(files))]
            for i in range(0, len(epoch) - batch + 1 if not once else len(epoch), batch):
                chunk = epoch[i : i + batch]
                arrays, labels = [], []
                for f, label in chunk:
                    try:
                        array = read(f)
                    except Exception:
                        # A few unreadable files must not end a run — the
                        # dataset profile already warned that some exist.
                        continue
                    if flips and float(torch.rand(1)) < 0.5:
                        # Asked for (`augment`), training only: a mirrored image is the same class.
                        array = array.flip(-1)
                    arrays.append((array - mean.view(-1, 1, 1)) / std.view(-1, 1, 1))
                    labels.append(label)
                if not arrays:
                    continue
                x = torch.stack(arrays).to(device)
                y = torch.tensor(labels, device=device)
                yield x, y
            if once:
                return

    elif path and Path(path).is_file() and req.get("inputKind") == "tokens":
        # A text model's first layer is an embedding, which takes integer
        # indices. Reading the file as floats — the numeric path below — gave
        # it NaNs where the sentences were and it refused them on the first
        # batch, so no text design could train at all.
        seq_len = int(input_shape[-1]) if len(input_shape) > 1 else 128
        if len(req.get("inputShapes") or []) == 2:
            yield from _seq2seq_batches(req, Path(path), device, torch, part, once)
            return
        if req.get("task") == "language_modeling":
            yield from _corpus_batches(req, Path(path), seq_len, device, torch, part, once)
            return
        def read_texts():
            given = []
            texts, labels = _read_text_table(Path(path), given, target=req.get("target"))
            return texts, labels, given

        texts, labels, given = _read_once("text", path, read_texts)
        vocab = int(req.get("vocabSize") or 0)
        if vocab < 2:
            fail("this design reads tokens but states no vocabulary size, so text cannot be encoded for it")
        classes = sorted(set(labels), key=_label_order)
        if WARM_START:
            classes = _warm_classes({}, classes, False)
        _require_class_outputs(classes, req.get("numClasses"))
        index = {label: i for i, label in enumerate(classes)}
        # The parts are row indices: the texts are read once and never copied.
        rows_all = list(range(len(texts)))
        trained = _part(rows_all, req, "train", labels=labels, groups=texts, given=given or None)
        chosen = trained if part == "train" else _part(rows_all, req, part, labels=labels, groups=texts, given=given or None)
        stat = Path(path).stat()
        key = (str(path), vocab, stat.st_size, stat.st_mtime_ns)
        _fit_text_encoder([texts[i] for i in trained], vocab, key)
        kind = _ENCODERS[key][1]
        if MAIN:
            write_json("labels.json", classes)
        # A blank text (spaces only, or nothing) would be an all-padding
        # sequence, which the model's padding mask refuses — one empty row
        # stopped a run whenever its batch came. Such rows are set aside, and
        # counted. Known without encoding anything.
        trained_kept = [i for i in trained if texts[i].strip()]
        empty = len(trained) - len(trained_kept)
        if trained and not trained_kept:
            fail(f"{Path(path).name} has no text to train on: every text of the training part is empty")
        weights = _class_weights([index[labels[i]] for i in trained_kept], max(num_classes, len(classes)))
        # The share of texts longer than the model reads, measured on a fixed
        # sample: counting it on every text meant encoding them all first.
        probe = trained_kept if len(trained_kept) <= CUT_SAMPLE else sorted(random.Random(1).sample(trained_kept, CUT_SAMPLE))
        many = _ENCODE_MANY[key]
        cut = sum(len(seq) > seq_len for seq in many([texts[i] for i in probe])) / max(1, len(probe))
        sample, of_texts = FITTED.get(key, (len(trained), len(trained)))
        _text_plan(kind, vocab, seq_len, cut,
                   **({"truncatedMeasuredOn": len(probe)} if len(probe) < len(trained_kept) else {}),
                   **({"fittedOnTexts": sample, "ofTrainingTexts": of_texts} if sample < of_texts else {}),
                   **({"classWeights": weights} if weights else {}),
                   **({"setAside": {"emptyText": empty}} if empty else {}))
        if empty:
            print(f"[neurax] {empty} training rows with an empty text set aside", flush=True)
        if not chosen:
            return
        kept = trained_kept if part == "train" else [i for i in chosen if texts[i].strip()]
        if not kept:
            fail(f"{Path(path).name} has no text in its {part} part: every one is empty")
        table = _token_table(texts, key, seq_len)
        targets = torch.tensor([index[labels[i]] for i in kept], dtype=torch.long)
        print(f"[neurax] {len(kept)} texts ({part}), {len(classes)} classes, {seq_len} tokens each, encoded as they are read",
              flush=True)
        yield from _table_batches_of(table, kept, targets, batch, device, torch, once)

    elif path and Path(path).is_file() and Path(path).suffix.lower() == ".pt" and req.get("inputKind") == "image":
        yield from _conditioned_image_batches(req, Path(path), input_shape, device, torch, part, once)

    elif path and Path(path).is_file() and req.get("inputKind") == "graph":
        yield from _graph_batches(req, Path(path), device, torch, part, once)

    elif path and Path(path).is_file():
        yield from _table_batches(req, Path(path), batch, device, torch, part, once, num_classes)

    elif path:
        fail(f"the dataset {path} is not on this machine: training reads a local folder of images or a CSV file")
    else:
        fail("no dataset: a run trains on real data, never on random tensors")


def prefetch(batches, depth=2):
    """The same batches, read and prepared `depth` ahead in a background thread.

    Reading images, decoding them and stacking a batch happen while the model
    computes the previous step rather than between steps: the step time is the
    longer of the two, not their sum. A failure while preparing (a `fail`, an
    unreadable file) is raised in the training loop, where it ends the run."""
    import queue
    import threading

    ready = queue.Queue(maxsize=max(1, depth))
    done = object()

    def produce():
        try:
            for item in batches:
                ready.put(item)
            ready.put(done)
        except BaseException as error:  # noqa: BLE001 — SystemExit from fail() included
            ready.put(error)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = ready.get()
        if item is done:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _tensor_batches(inputs, targets, batch, device, torch, once):
    """Shuffled batches of rows, forever; one ordered pass with `once`."""
    count = inputs.shape[0]
    while True:
        order = torch.arange(count) if once else torch.tensor(_epoch_order(count), dtype=torch.long)
        stop = len(order) if once else len(order) - batch + 1
        for i in range(0, max(stop, 0), batch):
            idx = order[i : i + batch]
            x = inputs[idx]
            # Token ids are stored compact (16 or 32 bits); the model reads int64.
            compact = not x.is_floating_point() and x.dtype != torch.long and x.dtype != torch.bool
            yield (x.long() if compact else x).to(device), targets[idx].to(device)
        if once:
            return


def _json_records(path):
    """The records of a JSON file: one per line (JSON Lines), or — for a
    `.json` whose content is one array — that array, as the dataset profile
    reads it (`datasource.py`)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        head = handle.read(64).lstrip()
        handle.seek(0)
        if path.suffix.lower() == ".json" and head.startswith("["):
            try:
                value = json.load(handle)
            except ValueError as exc:
                fail(f"{path.name} is not valid JSON: {exc}")
            return value if isinstance(value, list) else []
        rows = []
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows


def _read_corpus(path):
    """A corpus's text: a `.txt` file's lines, or every field of a table's rows."""
    if path.suffix.lower() in {".txt", ".md", ".text"}:
        return [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if path.suffix.lower() in {".jsonl", ".ndjson", ".json"}:
        return [" ".join(str(v) for v in (row.values() if isinstance(row, dict) else [row])) for row in _json_records(path)]
    return [" ".join(str(v if v is not None else "") for v in row.values()) for row in _table_rows(path)]


def _corpus_batches(req, path, seq_len, device, torch, part, once):
    """A language model's data: the corpus as one stream of token ids, cut
    into windows of `seq_len`; each window is its own input and, shifted by
    one, its own target."""
    vocab = int(req.get("vocabSize") or 0)
    if vocab < 2:
        fail("this design reads tokens but states no vocabulary size, so text cannot be encoded for it")
    corpus = _read_corpus(path)
    encode, kind = _fit_text_encoder(_part(corpus, req, "train", groups=corpus), vocab, (str(path), vocab))
    _text_plan(kind, vocab, seq_len, 0.0, stream=True)
    lines = _part(corpus, req, part, groups=corpus)
    stream = [i for line in lines for i in encode(line)]
    windows = len(stream) // seq_len
    if windows == 0:
        if part == "validation":
            return
        fail(f"the corpus has {len(stream)} tokens, fewer than one window of {seq_len}")
    ids = torch.tensor(stream[: windows * seq_len], dtype=torch.long).view(windows, seq_len)
    print(f"[neurax] {len(stream)} tokens ({part}), {windows} windows of {seq_len}", flush=True)
    yield from _tensor_batches(ids, ids, int(req["batchSize"]), device, torch, once)


def _seq2seq_batches(req, path, device, torch, part, once):
    """Paired source/target text for a two-input encoder/decoder model."""
    if path.suffix.lower() not in {".csv", ".tsv", ".jsonl", ".ndjson", ".json", ".parquet"}:
        fail("paired text must be a CSV, TSV, Parquet, JSONL or JSON file with source and target fields")
    rows = _table_rows(path)
    if not rows or not all(isinstance(row, dict) and "source" in row and "target" in row for row in rows):
        fail(f"{path} needs source and target text fields in every row")
    shapes = req["inputShapes"]
    if len(shapes) != 2 or len(shapes[0]) != 1 or len(shapes[1]) != 1:
        fail("this seq2seq model needs two token sequence inputs")
    source_len, target_len = int(shapes[0][0]), int(shapes[1][0])
    vocab = int(req.get("vocabSize") or 0)
    if vocab < 2:
        fail("this seq2seq model states no vocabulary size")
    # A pair with a blank side would be an all-padding sequence, which the
    # padding mask refuses; such pairs are set aside, and counted.
    total = len(rows)
    rows = [row for row in rows if str(row["source"] or "").strip() and str(row["target"] or "").strip()]
    empty = total - len(rows)
    if not rows:
        fail(f"{path.name} has no text to train on: every pair has an empty source or target")
    if empty:
        print(f"[neurax] {empty} pairs with an empty source or target set aside", flush=True)
    sources = [str(row["source"]) for row in rows]
    trained = _part(rows, req, "train", groups=sources)
    encode, kind = _fit_text_encoder([str(r["source"]) for r in trained] + [str(r["target"]) for r in trained], vocab,
                                     (str(path), vocab))
    _text_plan(kind, vocab, source_len, 0.0, targetLength=target_len,
               **({"setAside": {"emptyText": empty}} if empty else {}))
    rows = _part(rows, req, part, groups=sources)
    batch = int(req["batchSize"])
    print(f"[neurax] {len(rows)} source/target pairs ({part})", flush=True)
    while True:
        epoch = rows if once else [rows[i] for i in _epoch_order(len(rows))]
        stop = len(epoch) if once else len(epoch) - batch + 1
        for i in range(0, max(stop, 0), batch):
            chunk = epoch[i : i + batch]
            if not chunk:
                continue
            source = torch.tensor([_fit(encode(str(row["source"])), source_len) for row in chunk], device=device)
            target = torch.tensor([_fit(encode(str(row["target"])), target_len) for row in chunk], device=device)
            decoder = torch.cat((torch.zeros_like(target[:, :1]), target[:, :-1]), dim=1)
            yield (source, decoder), target
        if once:
            return


def _graph_batches(req, path, device, torch, part, once):
    """A graph saved by `torch.save`: `x`, `edge_index`, `y`, and the node
    masks PyTorch Geometric writes (`train_mask`, `val_mask`). The whole graph
    is each step's batch; the loss reads the part's nodes."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {"x", "edge_index", "y"} <= set(data):
        fail(f"{path} is not a graph: it needs x, edge_index and y tensors")
    nodes = data["x"].shape[0]
    mask = data.get("val_mask" if part == "validation" else "train_mask")
    if mask is None:
        train_idx, val_idx = split(nodes, req.get("validationFraction") or 0.1)
        mask = torch.zeros(nodes, dtype=torch.bool)
        mask[val_idx if part == "validation" else train_idx] = True
    graph = (data["x"].float().to(device), data["edge_index"].long().to(device),
             torch.zeros(nodes, dtype=torch.long, device=device), 1)
    target = (data["y"].to(device), mask.to(device))
    while True:
        yield graph, target
        if once:
            return


def _conditioned_image_batches(req, path, input_shape, device, torch, part, once):
    """Images and precomputed caption embeddings, aligned by sample index."""
    if req.get("task") not in ("image", "reconstruction"):
        fail("a .pt image/context dataset is for an image denoiser or autoencoder")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {"images", "context"} <= set(data):
        fail(f"{path} needs images [samples, channels, height, width] and context [samples, tokens, width]")
    images, context = data["images"], data["context"]
    if not torch.is_tensor(images) or not torch.is_tensor(context) or images.ndim != 4 or context.ndim != 3:
        fail(f"{path} has invalid images or context tensor shapes")
    if images.shape[0] != context.shape[0] or tuple(images.shape[1:]) != tuple(input_shape[-3:]):
        fail(f"{path} image count or image shape does not match the model and its captions")
    expected = next((e.get("shape") for e in (req.get("extras") or []) if e.get("name") == "context"), None)
    if expected and tuple(context.shape[1:]) != tuple(expected):
        fail(f"{path} context shape {tuple(context.shape[1:])} does not match the model's {tuple(expected)}")
    if not torch.isfinite(images).all() or not torch.isfinite(context).all():
        fail(f"{path} contains non-finite image or caption values")
    rows = _part(list(range(images.shape[0])), req, part)
    batch = int(req["batchSize"])
    print(f"[neurax] {len(rows)} images with caption embeddings ({part})", flush=True)
    while True:
        epoch = rows if once else [rows[i] for i in _epoch_order(len(rows))]
        stop = len(epoch) if once else len(epoch) - batch + 1
        for i in range(0, max(stop, 0), batch):
            indices = epoch[i : i + batch]
            if not indices:
                continue
            x = images[indices].float().to(device)
            labels = data.get("labels")
            y = {"context": context[indices].float().to(device)}
            if torch.is_tensor(labels):
                y["labels"] = labels[indices].long().to(device)
            yield x, y
        if once:
            return


#: The column a table's target is read from, as the dataset profile reads it.
TARGET_NAMES = ("label", "labels", "target", "class", "category", "y")

#: The images a folder dataset trains on: every format the profiler counts.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _given_part(value):
    """The part a `split` column's value names (`train`, `val`, …), or None."""
    return SPLIT_NAMES.get(str(value or "").strip().lower())


def _require_class_outputs(classes, num_classes):
    """Refuse a labelled dataset the model's classification head cannot represent."""
    if num_classes and len(classes) > num_classes:
        fail(f"the data has {len(classes)} classes but the model's head has {num_classes} outputs — "
             f"set the head to {len(classes)} classes")


def _read_cells(path):
    """A table's column names and its rows as text: CSV, TSV (or any delimiter
    the file uses) and Parquet — what the profiler reads, the run reads."""
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            fail("reading a Parquet table needs pandas and pyarrow: pip install pandas pyarrow")
        frame = pd.read_parquet(path)
        names = [str(c) for c in frame.columns]
        rows = [["" if (v is None or (isinstance(v, float) and v != v)) else str(v) for v in row]
                for row in frame.itertuples(index=False)]
    else:
        import csv

        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                pass
            reader = csv.reader(handle, delimiter=delimiter)
            names = next(reader, [])
            rows = [(row + [""] * len(names))[: len(names)] for row in reader if row]
    if not rows:
        fail(f"{path.name} has no rows")
    return names, rows


def _number(text):
    """A cell as a number, None when empty, or the string "text" when it is not a number."""
    text = text.strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return "text"
    return value if math.isfinite(value) else None


#: The scale a regression target was trained at, to report its error in its own units.
TARGET_SCALE = {}
#: The weight of each class in the loss, when the training part's classes are imbalanced.
CLASS_WEIGHTS = {}
#: The last batch's predicted and true classes, for validation's per-class figures.
LAST_CLASSES = {}


def _class_weights(indices, count):
    """Each class's weight in the loss — the inverse of its share — when the most
    common class of the training part is at least three times the rarest; else
    None, and the loss weighs every sample alike. A model of 92% one class
    learns to answer that class, and its accuracy says 0.92 for nothing."""
    counts = [0] * max(1, count)
    for i in indices:
        if 0 <= i < len(counts):
            counts[i] += 1
    present = [n for n in counts if n]
    CLASS_WEIGHTS.clear()
    if len(present) < 2 or max(present) < 3 * min(present):
        return None
    total = sum(present)
    weights = [total / (len(present) * n) if n else 1.0 for n in counts]
    CLASS_WEIGHTS["values"] = weights
    print(f"[neurax] imbalanced classes: weighed in the loss ({', '.join(f'{w:.2g}' for w in weights)})", flush=True)
    return weights


def balanced_accuracy(predictions, targets):
    """The mean of each class's recall: what a model that always answers the
    common class cannot fake."""
    recalls = []
    for c in sorted(set(int(t) for t in targets.tolist())):
        mask = targets == c
        recalls.append(float((predictions[mask] == c).float().mean()))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _table_batches(req, path, batch, device, torch, part, once, num_classes):
    """A table, prepared as a model reads it, its statistics fitted on the training part only.

    Numeric columns: empty cells take the training median, then the column is
    standardized. Columns of words are categories: each becomes the index of its
    category among the training part's (one column still, as the design's input
    width says), then standardized. A `split` column is the data's own split, not a
    feature. Classes become indices in a stable order (`labels.json`); a regression
    target is trained at unit scale. All of it is written to `preprocessing.json`:
    what turns a raw row into what the model reads, and its output back.
    """
    names, cells = _read_once("cells", path, lambda: _read_cells(path))
    lowered = [n.strip().lower() for n in names]
    split_col = lowered.index("split") if "split" in lowered else None
    candidates = [i for i in range(len(names)) if i != split_col]
    requested = req.get("target")
    if requested:
        # The column the dataset profile named, shown in Decision: the run
        # predicts that one, never one its own rule would have picked.
        if requested not in names:
            fail(f"the target `{requested}` the dataset profile named is not a column of {path.name}: "
                 f"its columns are {', '.join(names[:12])}")
        target = names.index(requested)
    else:
        target = next((lowered.index(t) for t in TARGET_NAMES if t in lowered), candidates[-1])
    features = [i for i in candidates if i != target]
    regression = req.get("task") == "regression"
    cells = [r for r in cells if r[target].strip() != ""]
    if not cells:
        fail(f"{path.name} has no row with a value in its target column `{names[target]}`")
    raw_targets = [r[target].strip() for r in cells]
    given = [_given_part(r[split_col]) for r in cells] if split_col is not None else None
    if regression:
        values = [_number(t) for t in raw_targets]
        if any(not isinstance(v, float) for v in values):
            fail(f"the regression target `{names[target]}` has values that are not numbers")
        labels, classes = None, None
    else:
        classes = sorted(set(raw_targets), key=_label_order)
        _require_class_outputs(classes, req.get("numClasses"))
        labels = raw_targets
    indices = list(range(len(cells)))
    # The split column is where a row goes, not what it is: left out of the
    # key, so the same row given to both parts is recognised as one.
    groups = ["\x1f".join(c for j, c in enumerate(r) if j != split_col) for r in cells]
    train_idx = _part(indices, req, "train", labels=labels, groups=groups, given=given)
    val_idx = _part(indices, req, "validation", labels=labels, groups=groups, given=given)

    plan = {"kind": "table", "source": path.name, "features": [], "fittedOn": "training part"}
    earlier = _warm_plan("table")
    if earlier is not None:
        # Started from an earlier run: its columns, read as it read them.
        wanted = [f["name"] for f in earlier.get("features") or []]
        have = [names[c] for c in features]
        if have != wanted:
            fail(f"this data's columns ({', '.join(have[:8])}) are not those the run it starts from was "
                 f"trained on ({', '.join(wanted[:8])}): a warm start reads the same columns")
        classes = _warm_classes(earlier, classes, regression)
        plan["fittedOn"] = f"the run it starts from ({WARM_START['dir']})"
    columns = []
    for c in features:
        column = [r[c] for r in cells]
        parsed = [_number(v) for v in column]
        missing = sum(1 for v in parsed if v is None)
        if earlier is not None:
            entry = dict(earlier["features"][len(columns)], missing=missing)
            if entry["kind"] == "category":
                code = {k: float(i) for i, k in enumerate(entry["categories"])}
                encoded = [code.get(v.strip(), float(len(entry["categories"]))) for v in column]
            else:
                encoded = [v if isinstance(v, float) else entry["fill"] for v in parsed]
            plan["features"].append(entry)
            columns.append([(v - entry["mean"]) / entry["std"] for v in encoded])
            continue
        if any(v == "text" for v in parsed):
            seen = {}
            for i in train_idx:
                key = column[i].strip()
                if key:
                    seen[key] = seen.get(key, 0) + 1
            categories = sorted(seen, key=lambda k: (-seen[k], k))
            code = {k: float(i) for i, k in enumerate(categories)}
            # A category the training part never had, or an empty cell, is one more code.
            encoded = [code.get(v.strip(), float(len(categories))) for v in column]
            entry = {"name": names[c], "kind": "category", "categories": categories, "missing": missing}
        else:
            known = sorted(parsed[i] for i in train_idx if isinstance(parsed[i], float))
            fill = known[len(known) // 2] if known else 0.0
            encoded = [v if isinstance(v, float) else fill for v in parsed]
            entry = {"name": names[c], "kind": "number", "fill": fill, "missing": missing}
        trained = [encoded[i] for i in train_idx] or encoded
        mean = sum(trained) / len(trained)
        std = math.sqrt(sum((v - mean) ** 2 for v in trained) / len(trained)) or 1.0
        entry.update({"mean": mean, "std": std})
        plan["features"].append(entry)
        columns.append([(v - mean) / std for v in encoded])
    if regression:
        trained = [values[i] for i in train_idx] or values
        mean = sum(trained) / len(trained)
        std = math.sqrt(sum((v - mean) ** 2 for v in trained) / len(trained)) or 1.0
        if earlier is not None and (earlier.get("target") or {}).get("kind") == "value":
            mean, std = earlier["target"]["mean"], earlier["target"]["std"]
        plan["target"] = {"name": names[target], "kind": "value", "mean": mean, "std": std}
        TARGET_SCALE.update(mean=mean, std=std)
        y_all = [(v - mean) / std for v in values]
    else:
        index = {label: i for i, label in enumerate(classes)}
        plan["target"] = {"name": names[target], "kind": "class", "classes": classes}
        y_all = [index[t] for t in raw_targets]
        weights = _class_weights([y_all[i] for i in train_idx], max(num_classes, len(classes)))
        if weights:
            plan["classWeights"] = weights
        if MAIN:
            write_json("labels.json", classes)
    if MAIN:
        write_json("preprocessing.json", plan)
    chosen = val_idx if part == "validation" else train_idx
    if not chosen:
        return
    x = torch.tensor([[col[i] for col in columns] for i in chosen], dtype=torch.float32)
    y = torch.tensor([y_all[i] for i in chosen], dtype=torch.float32 if regression else torch.long)
    print(f"[neurax] {x.shape[0]} rows ({part}), {x.shape[1]} features", flush=True)
    yield from _tensor_batches(x, y, batch, device, torch, once)


def _table_rows(path):
    """The rows of a table or a file of records, as dicts, in every format the
    dataset profile reads: CSV and TSV (delimiter guessed), JSON Lines, a JSON
    array, Parquet. Each text reader used its own subset, and a Parquet file
    the profile had read failed at step 0."""
    import csv

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson", ".json"}:
        return [row for row in _json_records(path) if isinstance(row, dict)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            fail("reading a Parquet file needs pandas and pyarrow: prepare Python + PyTorch to complete the runtime")
        frame = pd.read_parquet(path)
        return [{k: ("" if v is None else v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        # Only the delimiter is guessed. Guessing the quoting too, from the
        # first 4 KB, got real titles wrong (AG News: commas and doubled
        # quotes inside quoted fields) and read 40 "classes" out of 4
        # while the profile, through pandas, read 4. Past the delimiter a
        # CSV is the standard one: double quotes, doubled to escape.
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            delimiter = csv.Sniffer().sniff(sample or ",", delimiters=",;\t|").delimiter
        except csv.Error:
            pass
        return list(csv.DictReader(handle, delimiter=delimiter, quotechar='"', doublequote=True))


def _read_text_table(path, given=None, target=None):
    """`(texts, labels)` from a CSV/TSV or a JSON Lines file of text samples.

    The target is the column the dataset profile names (`label`, `target`, …,
    else the last one); the text is every other field, joined, so a file with
    a title and a body is read as one passage rather than losing half of it.
    """
    rows = [r for r in _table_rows(path) if isinstance(r, dict) and r]
    if not rows:
        fail(f"no rows could be read from {path}")
    names = list(rows[0].keys())
    lowered = {str(n).lower(): n for n in names}
    # A `split` column is the dataset's own split, never part of the text.
    split_column = lowered.get("split")
    names = [n for n in names if n != split_column]
    if target is not None:
        # The column the dataset profile named (see `_table_batches`).
        if target not in names:
            fail(f"the target `{target}` the dataset profile named is not a field of {path.name}: "
                 f"its fields are {', '.join(str(n) for n in names[:12])}")
    else:
        target = next((lowered[c] for c in TARGET_NAMES if c in lowered), names[-1])
    fields = [n for n in names if n != target]
    if not fields:
        fail(f"{path} has only its target column `{target}`: there is no text to train on")
    texts, labels = [], []
    for r in rows:
        label = r.get(target)
        if label is None or str(label).strip() == "":
            continue
        texts.append(" ".join(str(r.get(f) or "") for f in fields))
        labels.append(str(label).strip())
        if given is not None and split_column is not None:
            given.append(_given_part(r.get(split_column)))
    return texts, labels


def _label_order(label):
    """Numeric labels in numeric order (`2` before `10`), the rest by name."""
    try:
        return (0, float(label), "")
    except ValueError:
        return (1, 0.0, label)


#: Words and punctuation, as the word vocabulary splits text.
WORD = r"\w+|[^\w\s]"
#: Encoders fitted in this process, by dataset and vocabulary: fitted once, not at every evaluation.
_ENCODERS = {}
#: Per encoder: many texts to their token ids in one call.
_ENCODE_MANY = {}
#: Per encoder, the BPE tokenizer itself, when there is one.
_BPE = {}
#: The padding id of each encoder: 0 for NEURAX's own, the published
#: tokenizer's for a fine-tuning (RoBERTa pads with 1, GPT-2 with its end token).
_PAD = {}
#: The request of a fine-tuning (`fineTune`), when this run is one.
FINE_TUNE = {}
# The export of an earlier NEURAX run this one starts from (`fineTune.source.kind
# == "neurax-run"`): its weights, and its tokenizer for text.
WARM_START = {}


def _warm_plan(kind):
    """The preprocessing of the export a warm start begins from, when it is of
    this kind: weights learnt on its encoding read only its encoding."""
    if not WARM_START:
        return None
    path = Path(WARM_START["dir"]) / "preprocessing.json"
    try:
        plan = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return plan if plan.get("kind") == kind else None


def _warm_classes(earlier, classes, regression):
    """The earlier run's classes, in its order: its head's outputs are those.
    Another set of classes is another head, refused rather than relabelled."""
    if regression:
        return classes
    before = list((earlier.get("target") or {}).get("classes") or [])
    if not before:
        try:
            before = json.loads((Path(WARM_START["dir"]) / "labels.json").read_text())
        except (OSError, ValueError):
            before = []
    if before and sorted(map(str, before)) != sorted(map(str, classes)):
        fail(f"this data's classes ({', '.join(map(str, classes[:8]))}) are not those the run it starts from "
             f"learnt ({', '.join(map(str, before[:8]))}): a warm start predicts the same classes")
    return before or classes


def _warm_start(model, folder, torch):
    """Load an earlier run's export into this design, strictly: another design
    is refused with the first differences rather than half loaded."""
    import finetune  # noqa: PLC0415

    weights, refused = finetune.warm_start_differences(model, folder, torch)
    if refused:
        fail(refused)
    model.load_state_dict(weights, strict=True)
    print(f"[neurax] started from the weights of {folder}", flush=True)
#: Per dataset, encoder and length: every row's ids, encoded once, shared by
#: the training and the validation part (`_TokenTable`).
_TOKEN_TABLES = {}

#: Texts handed to the tokenizer per call: large enough for its parallelism,
#: small enough that one chunk's Python lists stay a few megabytes.
ENCODE_CHUNK = 8192
#: The most texts a vocabulary is learnt from (a fixed sample of the training part).
FIT_TEXTS = 200_000
#: The texts the share cut to the model's length is measured on.
CUT_SAMPLE = 20_000
#: Per encoder: how many texts it was learnt from, of how many.
FITTED = {}


class _TokenTable:
    """Every text's token ids, cut or padded to `seq_len`, encoded as the
    batches ask for them and kept: the first step waits for one batch, not for
    the whole dataset, and later passes reuse what is already encoded.

    Stored compact (16 bits when the vocabulary fits, else 32): lists of
    Python ints — 28 bytes a token — held a dataset at 26 to 32 times its size
    in memory, and encoding it whole held the first step for minutes."""

    def __init__(self, texts, key, seq_len):
        import numpy as np
        import torch

        self.texts, self.key, self.seq_len = texts, key, seq_len
        vocab = int(key[1])
        small = vocab <= 65_536 and hasattr(torch, "uint16")
        pad = _PAD.get(key, 0)
        self.ids = np.full((len(texts), seq_len), pad, dtype=np.uint16 if small else np.int32)
        self.done = np.zeros(len(texts), dtype=bool)
        self.sized = None
        bpe = _BPE.get(key)
        if bpe is not None:
            from tokenizers import Tokenizer  # noqa: PLC0415

            # A copy cuts and pads; the tokenizer itself, saved for use, does neither.
            self.sized = Tokenizer.from_str(bpe.to_str())
            self.sized.enable_truncation(seq_len)
            self.sized.enable_padding(length=seq_len, pad_id=pad)

    def rows(self, index):
        """The ids of rows `index` (a numpy array), encoding those not yet encoded."""
        import numpy as np

        todo = index[~self.done[index]]
        if todo.size:
            todo = np.unique(todo)
            texts = [self.texts[i] if self.texts[i].strip() else "" for i in todo]
            if self.sized is not None:
                encode = getattr(self.sized, "encode_batch_fast", self.sized.encode_batch)
                self.ids[todo] = np.asarray([e.ids for e in encode(texts)], dtype=self.ids.dtype)
            else:
                for row, seq in zip(todo, _ENCODE_MANY[self.key](texts)):
                    kept = seq[: self.seq_len]
                    self.ids[row, : len(kept)] = kept
            self.done[todo] = True
        return self.ids[index]


def _token_table(texts, key, seq_len):
    """The table of `texts` for this encoder and length, shared by the training
    and the validation part of a run."""
    table = _TOKEN_TABLES.get((key, seq_len))
    if table is None or len(table.texts) != len(texts):
        table = _TOKEN_TABLES[(key, seq_len)] = _TokenTable(texts, key, seq_len)
    return table


def _table_batches_of(table, rows, targets, batch, device, torch, once):
    """Shuffled batches of `rows` of a token table, forever; one ordered pass with `once`."""
    import numpy as np

    rows = np.asarray(rows, dtype=np.int64)
    while True:
        order = np.arange(len(rows)) if once else np.asarray(_epoch_order(len(rows)), dtype=np.int64)
        stop = len(order) if once else len(order) - batch + 1
        for i in range(0, max(stop, 0), batch):
            pick = order[i : i + batch]
            x = torch.from_numpy(table.rows(rows[pick])).long()
            yield x.to(device), targets[torch.from_numpy(pick)].to(device)
        if once:
            return


def _fit_text_encoder(texts, vocab, key):
    """`(encode, kind)`: text to token ids, fitted on `texts` (the training part).

    A subword tokenizer (BPE) at the model's vocabulary size when the
    `tokenizers` package is there; otherwise a vocabulary of the training
    part's most frequent words. 0 is padding and 1 an unknown token in both,
    and the tokenizer is written to `tokenizer.json`, to encode the same way
    when the model is used. Words were hashed into the vocabulary before:
    different words shared an id, and nothing of it was delivered."""
    if key in _ENCODERS:
        return _ENCODERS[key]
    if FINE_TUNE:
        # A published model reads the text as its own tokenizer cuts it: the
        # one it was trained with, not one fitted on the client's data.
        import finetune  # noqa: PLC0415
        from tokenizers import Tokenizer  # noqa: PLC0415

        published = finetune.tokenizer(FINE_TUNE)
        _BPE[key] = Tokenizer.from_str(published.backend_tokenizer.to_str())
        _PAD[key] = int(published.pad_token_id)
        _ENCODE_MANY[key] = lambda texts: [published(t)["input_ids"] for t in texts]
        encoded = (lambda text: published(text)["input_ids"]), "published"
        print("[neurax] text encoded by the published model's own tokenizer", flush=True)
        _ENCODERS[key] = encoded
        return encoded
    if WARM_START and (Path(WARM_START["dir"]) / "tokenizer.json").is_file():
        # Weights trained on another tokenizer's ids mean nothing on these:
        # the earlier run's tokenizer reads this text too.
        folder = Path(WARM_START["dir"])
        data = json.loads((folder / "tokenizer.json").read_text())
        if MAIN:
            (RUN_DIR / "tokenizer.json").write_text((folder / "tokenizer.json").read_text())
        if data.get("kind") == "words":
            one = text_encoder_from(folder)
            encoded = one, "words"
            _ENCODE_MANY[key] = lambda texts, one=one: [one(text) for text in texts]
        else:
            from tokenizers import Tokenizer  # noqa: PLC0415

            tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            encoded = (lambda text: tokenizer.encode(text).ids), "bpe"
            _ENCODE_MANY[key] = lambda texts: [e.ids for e in tokenizer.encode_batch(list(texts))]
            _BPE[key] = tokenizer
        print(f"[neurax] text encoded by the tokenizer of the run it starts from ({folder})", flush=True)
        _ENCODERS[key] = encoded
        return encoded
    import re
    from collections import Counter

    # A vocabulary is learnt from a sample, as large tokenizers are: past a
    # few hundred thousand texts it barely changes, while learning it from a
    # million and a half held over a gigabyte. The sample is the same every run.
    fitted_on = len(texts)
    if len(texts) > FIT_TEXTS:
        texts = [texts[i] for i in sorted(random.Random(0).sample(range(len(texts)), FIT_TEXTS))]
    FITTED[key] = (len(texts), fitted_on)

    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers  # noqa: PLC0415
    except ImportError:
        Tokenizer = None
    if Tokenizer is not None and vocab >= 64:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
        # Spaces kept as "▁" in the pieces, so that ids decode back to the text (Metaspace).
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        tokenizer.decoder = decoders.Metaspace()
        tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["[PAD]", "[UNK]"],
                                                                 show_progress=False))
        if MAIN:
            tokenizer.save(str(RUN_DIR / "tokenizer.json"))
        encoded = (lambda text: tokenizer.encode(text).ids), "bpe"
        # Many at once: the tokenizer's own batch encoder, in Rust and parallel.
        _ENCODE_MANY[key] = lambda texts: [e.ids for e in tokenizer.encode_batch(list(texts))]
        _BPE[key] = tokenizer
    else:
        counts = Counter(w for text in texts for w in re.findall(WORD, text.lower()))
        index = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(max(0, vocab - 2)))}
        if MAIN:
            write_json("tokenizer.json", {"kind": "words", "lowercase": True, "pattern": WORD, "pad": 0, "unknown": 1,
                                          "vocabulary": index})
        encoded = (lambda text: [index.get(w, 1) for w in re.findall(WORD, text.lower())]), "words"
        _ENCODE_MANY[key] = lambda texts, one=encoded[0]: [one(text) for text in texts]
    print(f"[neurax] text encoded by a {'subword (BPE)' if encoded[1] == 'bpe' else 'word'} vocabulary fitted on the training part",
          flush=True)
    _ENCODERS[key] = encoded
    return encoded


def text_encoder_from(folder):
    """The encoder a run fitted, read back from its `tokenizer.json` — as the model is used."""
    import re

    data = json.loads((Path(folder) / "tokenizer.json").read_text())
    if data.get("kind") == "words":
        index = data["vocabulary"]
        return lambda text: [index.get(w, data["unknown"]) for w in re.findall(data["pattern"], text.lower())]
    from tokenizers import Tokenizer  # noqa: PLC0415

    tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
    return lambda text: tokenizer.encode(text).ids


def _fit(ids, length):
    """Token ids cut or padded (with 0) to the model's input length."""
    return ids[:length] + [0] * (length - len(ids))


def _text_plan(kind, vocab, length, truncated, **extra):
    if MAIN:
        write_json("preprocessing.json", {"kind": "text", "tokenizer": kind, "vocabSize": vocab, "sequenceLength": length,
                                          "padding": 0, "unknown": 1, "truncated": truncated, "fittedOn": "training part",
                                          **extra})


def _to_array(img):
    import numpy as np

    return np.asarray(img, dtype="float32")


# ── The run ─────────────────────────────────────────────────────────────────


class _Built(Exception):
    """The model is already built (a published model); `model.py` is not read."""


def main():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # noqa: BLE001
        fail(f"PyTorch is not available to this interpreter: {exc}")
        return

    req = read_json("request.json")
    if not req:
        fail("request.json is missing or unreadable")
        return

    seed = seed_of(req)
    torch.manual_seed(seed)
    random.seed(seed)
    SEED["value"] = seed

    # Several GPUs: one process each, launched by torch.distributed.run, which
    # sets WORLD_SIZE. Only the first writes the run's files.
    global MAIN, SHARD
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist

        local = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
        MAIN = rank == 0
        SHARD = (rank, world)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[neurax] torch {torch.__version__} on {device}" + (f" (process {rank + 1} of {world})" if world > 1 else ""), flush=True)
    if device.type == "cpu" and world == 1:
        # The run is niced below the service (`start_run`); on the CPU it also
        # leaves a core to the studio and the rest of the machine.
        threads = cpu_threads(req, torch)
        torch.set_num_threads(threads)
        print(f"[neurax] training at low priority on {threads} CPU threads", flush=True)

    # The design, as PyTorch. `model.py` sits beside this file: the compiler
    # wrote it from the design it priced, or the assistant did and it passed
    # verification.
    sys.path.insert(0, str(RUN_DIR))
    fine = None
    # Whatever an earlier run in this process was, this one is what its
    # request says: a fine-tuning only when it carries `fineTune`.
    FINE_TUNE.clear()
    WARM_START.clear()
    warm = (req.get("fineTune") or {}).get("source") or {}
    if warm.get("kind") == "neurax-run":
        # An earlier NEURAX run's export: this design, trained on from its weights.
        WARM_START["dir"] = str(warm.get("dir") or "")
    elif req.get("fineTune"):
        # A published model: built by its own libraries from the checked
        # weights, trained through this same loop (see finetune.py).
        FINE_TUNE.update(req)
        try:
            import finetune  # noqa: PLC0415

            model, fine = finetune.build(req, device, torch)
            model_module = None
            if req.get("task", "classification") == "classification" and req.get("inputKind") == "tokens":
                req["vocabSize"] = len(finetune.tokenizer(req))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            fail(str(exc)) if type(exc).__name__ == "Refused" else fail_on_exception("The published model could not be built")
            return
    try:
        if fine is not None:
            raise _Built()
        import model as model_module  # noqa: PLC0415

        cls = getattr(model_module, req["modelClass"], None)
        if cls is None:
            # The generator names the class after the design, so a mismatch is
            # recoverable: take the only nn.Module defined in the file.
            candidates = [
                v
                for v in vars(model_module).values()
                if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module
                and getattr(v, "__module__", "") == model_module.__name__
            ]
            if len(candidates) != 1:
                fail(f"model.py defines no class named {req['modelClass']}")
                return
            cls = candidates[0]
        model = cls().to(device)
        if WARM_START:
            _warm_start(model, Path(WARM_START["dir"]), torch)
    except _Built:
        pass
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail_on_exception("The model's code could not be built")
        return

    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # The precision the analysis priced. Mixed precision computes in 16 bits
    # and keeps fp32 master weights; fp16 also scales the loss. A CPU has no
    # fp16 arithmetic, so there it is bf16 — said, not silently changed.
    asked = str(req.get("precision") or "fp32").lower()
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(asked)
    if amp is torch.float16 and device.type != "cuda":
        amp = torch.bfloat16
        print("[neurax] fp16 needs a GPU; training in bf16 on the CPU", flush=True)
    trained_in = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(amp, "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=amp is torch.float16)

    # Before a single step. This is the cheapest possible moment to discover
    # that a formula is wrong.
    if MAIN:
        write_json(
            "model_built.json",
            {
                "parameters": parameters,
                "trainableParameters": trainable,
                "device": str(device),
                "devices": world,
                "torchVersion": torch.__version__,
                "dtype": str(next(model.parameters()).dtype).replace("torch.", "")
                if parameters
                else req.get("precision", "fp32"),
                "trainingPrecision": trained_in,
                "task": task_of(req),
                **({"fineTune": {"method": fine["method"], "head": fine["head"]}} if fine else {}),
                **({"warmStart": WARM_START["dir"]} if WARM_START else {}),
            },
        )
    print(f"[neurax] {parameters:,} parameters", flush=True)

    # One sample's shape, from the design. Not defaulted: a wrong input shape
    # does not degrade a run, it kills it in the first layer, and guessing
    # would hide which of the two was wrong.
    sample_shape = tuple(int(d) for d in req["inputShape"])
    input_shape = (int(req["batchSize"]), *sample_shape)
    # Prepared ahead while the model computes (see `prefetch`).
    batches = prefetch(load_batches(req, input_shape, device, torch), depth=int(req.get("prefetchBatches") or 2))

    # Parts the design takes pretrained (a frozen text encoder): their
    # published weights, loaded before the first step when they can be.
    if fine is None:
        _load_published_parts(model)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    raw_model = model.module if world > 1 else model

    # Recomputing the repeated layers' activations in the backward pass: the
    # memory the analysis priced when the panel asks for it.
    if req.get("gradientCheckpointing") and fine is None:
        wrapped = enable_checkpointing(torch, raw_model)
        print(
            f"[neurax] gradient checkpointing on {wrapped} repeated layers" if wrapped
            else "[neurax] gradient checkpointing asked for, but the model has no repeated layers to recompute",
            flush=True,
        )

    optimiser = build_optimiser(torch, model, req)
    task = task_of(req)
    if task == "language_modeling" and len(req.get("inputShapes") or [None]) < 2:
        # Predicting each next token with an attention that sees the tokens after it is
        # copying, not learning: the loss falls and the model learns nothing it can use.
        seeing = [name for name, part in raw_model.named_modules() if getattr(part, "causal", None) is False]
        if seeing:
            fail(f"this language model's attention ({', '.join(seeing[:3])}) sees the tokens after each position, "
                 "so it would learn to copy the next token rather than predict it: make the attention causal, "
                 "or train it as a classifier or an encoder")
    extras = list(req.get("extras") or [])
    if extras:
        print("[neurax] the model also reads " + ", ".join(e["name"] for e in extras), flush=True)
    graph_type = getattr(model_module, "Graph", None) if model_module is not None else None
    # Max Steps, when the panel set it, is the run's length — the figure the
    # analysis priced; otherwise the epochs over the data decide.
    max_steps = int(req.get("maxSteps") or 0)
    total_steps = max_steps if max_steps > 0 else max(1, int(req["stepsPerEpoch"]) * int(req["epochs"]))
    lr_at = build_lr_schedule(req, total_steps)
    steps_per_epoch = max(1, int(req["stepsPerEpoch"]))
    checkpoint_every = max(1, int(req["checkpointEverySteps"]))
    # Batches whose gradients one optimizer step sums: the effective batch is
    # the batch size times this, as the analysis priced it.
    accumulate = max(1, int(req.get("gradAccumSteps") or 1))
    # Epochs without a better validation loss before the run ends. 0: never.
    patience = max(0, int(req.get("earlyStoppingPatience") or 0))
    stale_epochs = 0
    keep_checkpoints = int(req.get("keepCheckpoints", 3))

    def forward_loss(x, y):
        """The task's loss on one batch, and how many predictions were right."""
        if task == "graph" and graph_type is not None and isinstance(x, tuple):
            x = graph_type(*x)
        inputs, target, level = prepare(task, x, y, torch)
        kwargs = extra_inputs(extras, inputs, y, level, torch)
        with torch.autocast(device.type, dtype=amp or torch.float32, enabled=amp is not None):
            # Several inputs (a seq2seq pair) are passed apart; a graph is one
            # input, though its `Graph` is a NamedTuple and so a tuple too.
            several = isinstance(inputs, tuple) and not hasattr(inputs, "_fields")
            out = model(*inputs, **kwargs) if several else model(inputs, **kwargs)
        if hasattr(out, "_fields") and hasattr(out, "x"):
            # A graph network returns its graph: the scores are its node features.
            out = out.x
        loss, correct, seen = task_loss(task, out, inputs, target, torch)
        # Losses the design adds (load balancing, router z-loss): every module
        # that computed one this pass keeps it in `aux_loss`.
        for module in raw_model.modules():
            extra = getattr(module, "aux_loss", None)
            if extra is not None and model.training:
                loss = loss + extra
        return loss, correct, seen

    # Resume, when a checkpoint is there to resume from. The step it reached
    # is where this run picks up — not step 1, which would silently retrain
    # everything already done.
    step = 0
    resumed = sorted((RUN_DIR / "checkpoints").glob("step_*.pt"))
    if resumed:
        try:
            # `weights_only=True`, and not for tidiness. A checkpoint is a
            # pickle, and the default unpickler runs whatever the file tells
            # it to — so resuming a run means executing the contents of a
            # `.pt` file found in a directory this process did not choose:
            # the project folder comes from the request, and a downloaded or
            # shared run directory is an ordinary thing to point at. Nothing
            # saved here needs the general unpickler — `save_checkpoint`
            # writes an int, two state dicts of tensors and an RNG tensor,
            # all of which the restricted loader accepts.
            ckpt = torch.load(resumed[-1], map_location=device, weights_only=True)
            raw_model.load_state_dict(ckpt["model"], strict=fine is None)
            optimiser.load_state_dict(ckpt["optimiser"])
            step = int(ckpt.get("step", 0))
            if "rng" in ckpt:
                torch.set_rng_state(ckpt["rng"])
            print(f"[neurax] resumed at step {step}", flush=True)
            # The service rewrites state.json to resume: the last validation
            # comes back from eval.jsonl, or a run resumed at its last step
            # finishes with no accuracy to show.
            try:
                last = [json.loads(line) for line in (RUN_DIR / "eval.jsonl").read_text().splitlines() if line.strip()][-1]
                update_state(force=True, validationLoss=last.get("loss"), validationAccuracy=last.get("accuracy"))
            except (OSError, ValueError, IndexError):
                pass
        except Exception as exc:  # noqa: BLE001
            # Said out loud, with the reason. "Starting fresh" after a
            # thousand steps is the kind of thing someone needs to see, and
            # the reason distinguishes a corrupt file from a checkpoint this
            # PyTorch refuses to load without arbitrary code.
            print(f"[neurax] a checkpoint was present but unreadable ({exc}); starting fresh", flush=True)

    steps_file = (RUN_DIR / "steps.jsonl").open("a", buffering=1) if MAIN else open(os.devnull, "w")
    evaluations = [json.loads(line) for line in (RUN_DIR / "eval.jsonl").read_text().splitlines()] \
        if MAIN and (RUN_DIR / "eval.jsonl").exists() else []
    best = (read_json("export/config.json", {}) or {}).get("validationLoss")
    update_state(status="running", pid=os.getpid(), step=step)

    def evaluate(record=True):
        """The validation part, once, without gradients: its mean loss and,
        for a task that predicts classes, its accuracy. Without `record`, only
        measured: nothing written, nothing exported."""
        nonlocal best
        model.eval()
        total = right = count = 0.0
        predicted, truth = [], []
        # Bounded: a large validation part is sampled, not read whole every epoch.
        limit = int(req.get("evalBatches") or 200)
        with torch.no_grad():
            for seen_batches, (x, y) in enumerate(load_batches(req, input_shape, device, torch, part="validation", once=True)):
                if seen_batches >= limit:
                    break
                LAST_CLASSES.clear()
                loss, correct, seen = forward_loss(x, y)
                total += float(loss) * seen
                right += correct
                count += seen
                if LAST_CLASSES:
                    predicted.append(LAST_CLASSES["predicted"].cpu())
                    truth.append(LAST_CLASSES["target"].cpu())
        model.train()
        if not count:
            return None
        result = {"step": step, "epoch": (step - 1) // steps_per_epoch, "loss": total / count}
        if task in ("classification", "language_modeling", "graph"):
            result["accuracy"] = right / count
        if predicted:
            result["balancedAccuracy"] = balanced_accuracy(torch.cat(predicted), torch.cat(truth))
        if task == "regression" and TARGET_SCALE.get("std"):
            # The loss is at unit scale: the error in the target's own units is what a reader can judge.
            result["rmse"] = math.sqrt(max(result["loss"], 0.0)) * TARGET_SCALE["std"]
        if not record:
            return result
        evaluations.append(result)
        if MAIN:
            with (RUN_DIR / "eval.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            steps_file.flush()
            write_diagnosis(task, evaluations, classes=int(req.get("numClasses") or 0) or None,
                            vocab=int(req.get("vocabSize") or 0) or None)
            update_state(force=True, validationLoss=result["loss"], validationAccuracy=result.get("accuracy"))
            if math.isfinite(result["loss"]) and (best is None or result["loss"] < best):
                best = result["loss"]
                export(torch, raw_model, req, step, result, parameters, "best")
        return result

    if fine is not None and MAIN:
        import finetune  # noqa: PLC0415

        # The starting model, measured before a step: what fine-tuning is
        # measured against. A head created for the data's classes is untrained
        # (chance level), and said to be.
        before = evaluate(record=False) if step == 0 else None
        finetune.write_log(RUN_DIR, tokenizer="published" if fine.get("pad") is not None else None,
                           method=fine["method"], head=fine["head"],
                           **({"before": {**before, "headTrained": False}} if before else {}))

    # Gradients longer than this are scaled down to it before the update: one
    # bad batch moves the weights by a bounded step. 0 turns it off.
    max_grad_norm = float(req.get("maxGradNorm", 1.0) or 0.0)
    # Losses of the steps taken, to tell a run that diverges from one that learns.
    history = []
    not_a_number = 0

    control_checked_at = 0.0
    command = "run"
    # Time spent training and evaluating, pauses excluded, carried across a
    # resume: what the run really took, against what was approved.
    spent = {"seconds": float((read_json("state.json", {}) or {}).get("trainingSeconds") or 0.0) if step else 0.0,
             "stepSeconds": 0.0, "steps": 0}

    try:
        while step < total_steps:
            # Read at most twice a second rather than once a step, for the
            # same reason as the state file: at a thousand steps a second the
            # loop would spend its time opening a file that changes when a
            # human clicks something. Half a second is imperceptible on a
            # stop button and free on the loop.
            if time.monotonic() - control_checked_at >= 0.5:
                command = (read_json("control.json", {}) or {}).get("command", "run")
                control_checked_at = time.monotonic()

            if command == "stop":
                save_checkpoint(torch, raw_model, optimiser, step, exact=True, keep=keep_checkpoints)
                export(torch, raw_model, req, step, None, parameters, "last")
                update_state(status="finished", step=step, trainingSeconds=spent["seconds"])
                record_actual("stopped", step, spent)
                print("[neurax] stopped by request", flush=True)
                return

            if command == "pause":
                # A pause writes a checkpoint once, then idles cheaply. Idling
                # rather than exiting is what makes resume instant and keeps
                # the process's own identity — the studio is still attached to
                # a run, not to a corpse it has to restart.
                if not (RUN_DIR / "checkpoints" / f"step_{step:06d}.pt").exists():
                    save_checkpoint(torch, raw_model, optimiser, step, exact=True, keep=keep_checkpoints)
                update_state(status="paused", step=step)
                time.sleep(0.5)
                # Force the next iteration to re-read: while paused, the file
                # is the only thing that can change, and a throttled read
                # would make resume take up to a second longer than the click.
                control_checked_at = 0.0
                continue

            update_state(status="running", step=step, trainingSeconds=spent["seconds"])

            started = time.perf_counter()
            data_seconds = 0.0
            optimiser.zero_grad(set_to_none=True)
            seen = 0
            losses = []
            for _ in range(accumulate):
                fetched = time.perf_counter()
                try:
                    x, y = next(batches)
                except StopIteration:
                    fail("the training part of the dataset is smaller than one batch: lower the batch size")
                    return
                data_seconds += time.perf_counter() - fetched
                if torch.is_tensor(x) and x.is_floating_point() and not torch.isfinite(x).all():
                    fail("the data has values that are not numbers (an empty cell, or text in a numeric column) "
                         f"in the batch of step {step + 1}: fill or remove them before training")
                part, _, part_seen = forward_loss(x, y)
                # Each batch's share of the step: the sum is the mean over them all. The
                # backward pass always runs, even on a loss that is not a number: in a
                # multi-GPU run every process must join the gradients' exchange, or all wait.
                scaler.scale(part / accumulate).backward()
                losses.append(part.detach())
                seen += part_seen
            loss = torch.stack(losses).mean()
            scaler.unscale_(optimiser)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf")).item()
            # Decided on the gradients, which every process holds the same after the exchange,
            # so that all skip together. In fp16 the scaler skips an overflowing step itself.
            if not torch.isfinite(loss) or (not math.isfinite(grad_norm) and not scaler.is_enabled()):
                # A step on values that are not numbers would write NaN into every
                # weight: it is not taken. Several in a row mean the run cannot learn.
                optimiser.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                not_a_number += 1
                # Said in the log, not in steps.jsonl: the studio's curves read every line there as a step taken.
                print(f"[neurax] step {step + 1} not taken: the loss is not a number", flush=True)
                if not_a_number >= 3 or step == 0:
                    fail(f"the loss is not a number (NaN or infinite) at step {step + 1}"
                         f"{f', {not_a_number} steps in a row' if not_a_number > 1 else ''}: nothing can be learned from it. "
                         "Likely causes: values that are not numbers in the data, a learning rate too high "
                         f"({optimiser.param_groups[0]['lr']:.3g}), or fp16 overflowing — check the data, "
                         "lower the learning rate, or train in bf16/fp32")
                continue
            not_a_number = 0
            for group in optimiser.param_groups:
                group["lr"] = lr_at(step)
            scaler.step(optimiser)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = time.perf_counter() - started
            step += 1
            spent["seconds"] += step_seconds
            spent["stepSeconds"] += step_seconds
            spent["steps"] += 1

            steps_file.write(
                json.dumps(
                    {
                        "step": step,
                        "epoch": (step - 1) // steps_per_epoch,
                        "loss": float(loss.item()),
                        "learningRate": float(optimiser.param_groups[0]["lr"]),
                        "gradNorm": float(grad_norm),
                        "stepTimeMs": step_seconds * 1000.0,
                        "dataTimeMs": data_seconds * 1000.0,
                        "samplesPerSec": float(seen * world) / max(step_seconds, 1e-9),
                        **memory_fields(torch, device),
                    }
                )
                + "\n"
            )

            history.append(float(loss.item()))
            rise = diverged(history)
            if rise:
                start, now_, window = rise
                fail(f"the loss diverged: {start:.4g} " + ("at the first step" if window == 1 else f"on average over the first {window} steps")
                     + f", {now_:.4g} over the last 10 (step {step})."
                     f" The run is not learning — lower the learning rate "
                     f"(now {optimiser.param_groups[0]['lr']:.3g}), keep gradient clipping on, and check the "
                     "scale of the data and of the target")

            if step % checkpoint_every == 0:
                save_checkpoint(torch, raw_model, optimiser, step, exact=True, keep=keep_checkpoints)
            if step % steps_per_epoch == 0 or step == total_steps:
                before = best
                evaluating = time.perf_counter()
                result = evaluate()
                spent["seconds"] += time.perf_counter() - evaluating
                if patience and result is not None:
                    stale_epochs = 0 if before is None or result["loss"] < before else stale_epochs + 1
                    if stale_epochs >= patience:
                        save_checkpoint(torch, raw_model, optimiser, step, exact=True, keep=keep_checkpoints)
                        export(torch, raw_model, req, step, None, parameters, "last")
                        note = f"stopped early: no better validation loss for {stale_epochs} epoch{'s' if stale_epochs != 1 else ''}"
                        update_state(status="finished", step=step, note=note, trainingSeconds=spent["seconds"])
                        record_actual("finished", step, spent)
                        print(f"[neurax] {note}, at step {step}", flush=True)
                        return

        save_checkpoint(torch, raw_model, optimiser, step, exact=True, keep=keep_checkpoints)
        export(torch, raw_model, req, step, None, parameters, "last")
        update_state(status="finished", step=step, trainingSeconds=spent["seconds"])
        record_actual("finished", step, spent)
        print(f"[neurax] finished at step {step}", flush=True)

    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        fail_on_exception("Training stopped on an error")
    finally:
        steps_file.close()


def diverged(history, window=10):
    """`(first, last, window)` when the losses of a run have diverged, else None.

    Diverged: the last `window` losses average three times the first `window`,
    and every one of them is above every early one — a rise that noise around
    a small loss (0.01 to 0.04 and back) does not make. Or they average ten
    times the very first loss, every one of them above it: a run that blew up
    in its first steps has its first window high as well, and the first rule
    alone compares the explosion with itself."""
    if len(history) < 2 * window:
        return None
    first, recent = history[0], history[-window:]
    now = sum(recent) / window
    if first > 0 and now > 10 * first and min(recent) > first:
        return first, now, 1
    if len(history) < 3 * window:
        return None
    start = sum(history[:window]) / window
    if start > 0 and now > 3 * start and min(recent) > max(history[:window]):
        return start, now, window
    return None


def interpret(task, steps, evals, classes=None, vocab=None):
    """What a run's losses say, as an engineer reads them: a list of findings,
    each `{"kind", "step", "message"}` with its figures.

    The training loss is cross-entropy (nats) for classes and tokens, mean
    squared error at unit scale for a regression; an untrained model starts
    near ln(classes) — ln(vocabulary) for a language model — and that is what
    the first loss is held against."""
    found = []
    losses = [s["loss"] for s in steps if isinstance(s.get("loss"), (int, float)) and math.isfinite(s["loss"])]

    def say(kind, step, message, **figures):
        # The figures travel with the sentence, so that a reader in another language can say it.
        found.append({"kind": kind, "step": step, "message": message, "figures": figures})

    if not losses:
        return found
    outputs = vocab if task == "language_modeling" else classes
    if task in ("classification", "graph", "language_modeling") and outputs and outputs > 1:
        expected = math.log(outputs)
        if losses[0] > 1.5 * expected + 0.5:
            say("start", 1, f"The first loss is {losses[0]:.4g}, where an untrained model with {outputs} outputs scores "
                            f"about {expected:.3g} (ln {outputs}): its outputs start far from even — the last layer's "
                            "initialization or the scale of its inputs.", first=losses[0], outputs=outputs, expected=expected)
    tenth = max(1, len(losses) // 10)
    start, end = sum(losses[:tenth]) / tenth, sum(losses[-tenth:]) / tenth
    if len(losses) >= 20:
        drop = (start - end) / start if start > 0 else 0.0
        if drop > 0.01:
            say("learned", len(losses), f"The training loss fell from {start:.4g} to {end:.4g} ({drop:.0%}).",
                start=start, end=end, drop=drop)
        else:
            say("not_learning", len(losses), f"The training loss did not fall: {start:.4g} at the start, {end:.4g} at "
                                             "the end. Try a higher learning rate, a check of the data, or a larger model.",
                start=start, end=end)
    fifth = len(losses) // 5
    if fifth >= 10:
        before, last = sum(losses[-2 * fifth:-fifth]) / fifth, sum(losses[-fifth:]) / fifth
        if before > 0 and abs(before - last) / before < 0.005 and start > last * 1.05:
            say("plateau", len(losses), f"The training loss stopped falling over the last fifth of the run ({before:.4g} "
                                        f"→ {last:.4g}): more steps at this learning rate would not help; a lower one, "
                                        "a decay schedule or a larger model might.", before=before, last=last)
    if evals:
        best = min(evals, key=lambda e: e["loss"])
        after = [e for e in evals if e["step"] > best["step"]]
        if len(after) >= 2 and all(e["loss"] > best["loss"] * 1.05 for e in after[-2:]) and end < start:
            say("overfitting", evals[-1]["step"],
                f"The validation loss was lowest at step {best['step']} ({best['loss']:.4g}) and rose to "
                f"{evals[-1]['loss']:.4g} while the training loss kept falling: the model learns the training part by "
                f"heart. best.pt holds the model of step {best['step']}; more data, regularization or an earlier stop help.",
                best_step=best["step"], best=best["loss"], last=evals[-1]["loss"])
        final = evals[-1]
        score = final.get("balancedAccuracy", final.get("accuracy"))
        if task in ("classification", "graph") and final.get("accuracy") is not None:
            balanced = final.get("balancedAccuracy")
            say("validated", final["step"], f"On validation: accuracy {final['accuracy']:.3g}"
                + (f", balanced accuracy {balanced:.3g}" if balanced is not None else "")
                + (f" — guessing among {classes} classes scores {1 / classes:.3g}." if classes and classes > 1 else "."),
                accuracy=final["accuracy"], balanced=balanced, chance=(1 / classes if classes and classes > 1 else None))
        if task in ("classification", "graph") and classes and classes > 1 and score is not None and len(evals) >= 2:
            if score <= 1.0 / classes + 0.02:
                say("chance", final["step"], f"Validation scores {score:.3g}, what guessing among {classes} classes scores "
                                             f"({1 / classes:.3g}): the model has not learned to tell them apart.",
                    score=score, classes=classes, chance=1 / classes)
        if task == "language_modeling" and outputs:
            say("perplexity", final["step"], f"Validation perplexity {math.exp(final['loss']):.0f} — the model hesitates "
                                             f"as between that many tokens, out of a vocabulary of {outputs}.",
                perplexity=math.exp(final["loss"]), vocabulary=outputs)
        if task == "regression" and final.get("rmse") is not None:
            say("error", final["step"], f"On validation the prediction is off by {final['rmse']:,.0f} on average "
                                        "(root mean squared error, in the target's own units).", rmse=final["rmse"])
    return found


def write_diagnosis(task, evals, classes=None, vocab=None):
    """`diagnosis.json`: the run's findings, read from steps.jsonl and the validations."""
    if not MAIN:
        return []
    try:
        steps = [json.loads(line) for line in (RUN_DIR / "steps.jsonl").read_text().splitlines() if line.strip()]
    except OSError:
        steps = []
    found = interpret(task, steps, evals, classes=classes, vocab=vocab)
    write_json("diagnosis.json", {"findings": found})
    return found


def record_actual(status, step, spent):
    """What the run really took, written beside what was approved.

    The manifest the studio wrote carries the approved figures; this adds the
    run's own, so an auditor reads both side by side and the next estimate on
    this machine can learn from the difference. Cost follows time: the
    approved cost, scaled by the time really taken. Nothing is written for a
    run that has no manifest.
    """
    if not MAIN:
        return
    manifest = read_json("manifest.json", None)
    if not isinstance(manifest, dict):
        return
    approved = manifest.get("approved") or {}
    seconds = spent["seconds"]
    hours = seconds / 3600.0
    cost = None
    if isinstance(approved.get("trainingCostUsd"), (int, float)) and isinstance(approved.get("trainingHours"), (int, float)) and approved["trainingHours"] > 0:
        cost = approved["trainingCostUsd"] * hours / approved["trainingHours"]
    budget = approved.get("budgetUsd")
    manifest["actual"] = {
        "status": status,
        "finishedAt": datetime.now(timezone.utc).isoformat(),
        "steps": step,
        "trainingSeconds": seconds,
        "stepSeconds": spent["stepSeconds"] / spent["steps"] if spent["steps"] else None,
        "costUsd": cost,
        "overBudget": bool(cost is not None and isinstance(budget, (int, float)) and cost > budget),
    }
    write_json("manifest.json", manifest)


def seed_of(req):
    """The seed the run starts from: the panel's, or 0 so a run is repeatable."""
    try:
        return int(req.get("seed") or 0)
    except (TypeError, ValueError):
        return 0


def enable_checkpointing(torch, model):
    """Recompute each repeated layer's activations in the backward pass.

    The layers of an `nn.ModuleList` are the repeated blocks a design stacks,
    and where its activations live. Their `forward` is wrapped in place rather
    than the modules replaced, so the weights keep their names and the exported
    ones load into `model.py` unchanged. Returns how many layers it wrapped.
    """
    from torch.utils.checkpoint import checkpoint

    wrapped = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        for layer in module:
            original = layer.forward

            def forward(*args, _layer=layer, _original=original, **kwargs):
                if _layer.training and torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)

            layer.forward = forward
            wrapped += 1
    return wrapped


def task_of(req):
    """The task the design's output states; a classifier when it states none."""
    task = str(req.get("task") or "classification")
    if req.get("inputKind") == "graph":
        return "graph"
    return "classification" if task in ("auto", "embedding") else task


def prepare(task, x, y, torch):
    """What the model reads, what its output is compared with, and the noise
    level it was given (or None).

    A denoiser reads the image with noise at a random level of a cosine
    schedule and predicts the noise, as DDPM trains it; the level, in [0, 1],
    is what a model that reads `t` is told. An autoencoder reads its sample
    and reconstructs it. Every other task reads its sample and predicts its
    target."""
    if task == "image":
        noise = torch.randn_like(x)
        level = torch.rand(x.shape[0], device=x.device)
        kept = torch.cos(level * math.pi / 2).pow(2).view(-1, *([1] * (x.dim() - 1)))
        return kept.sqrt() * x + (1 - kept).sqrt() * noise, noise, level
    if task == "reconstruction":
        return x, x, None
    return x, y, None


SAID_EMPTY = set()


def extra_inputs(extras, x, y, level, torch):
    """What the model reads besides its data, by name, as keyword arguments.

    `t` is the noise level `prepare` drew. `labels` are the dataset's class
    labels when it has them. Caption embeddings (`context`) are supplied by
    a conditioned image dataset. An action or a foundation model's token
    (`semantic`) is still absent when the dataset does not carry it; the
    run says this once."""
    kwargs = {}
    for extra in extras:
        name = extra.get("name")
        if name == "t":
            if level is not None:
                kwargs["t"] = level
            continue
        if name == "context":
            if isinstance(y, dict) and torch.is_tensor(y.get("context")):
                kwargs["context"] = y["context"]
            else:
                fail("this model reads context but this batch has no caption embeddings")
            continue
        if name == "labels" and isinstance(y, dict) and torch.is_tensor(y.get("labels")):
            kwargs["labels"] = y["labels"]
            continue
        if name == "labels" and torch.is_tensor(x) and torch.is_tensor(y) and not y.is_floating_point() and y.dim() == 1 and y.shape[0] == x.shape[0]:
            kwargs["labels"] = y
            continue
        if name not in SAID_EMPTY:
            SAID_EMPTY.add(name)
            what = {"context": "captions", "labels": "class labels", "action": "actions", "semantic": "foundation-model tokens"}.get(name, name)
            print(f"[neurax] the model reads {name} and the data has no {what}: it trains on the empty one (zeros)", flush=True)
    return kwargs


def task_loss(task, out, inputs, target, torch):
    """`(loss, correct predictions, samples)` for one batch."""
    import torch.nn.functional as F

    out = out.float()
    if task == "language_modeling":
        if isinstance(inputs, tuple):
            logits = out.reshape(-1, out.shape[-1])
            wanted = target.reshape(-1)
            valid = wanted != 0
            if not valid.any():
                fail("the target text in this batch has no tokens")
            logits, wanted = logits[valid], wanted[valid]
            return F.cross_entropy(logits, wanted), float((logits.argmax(-1) == wanted).sum()), wanted.numel()
        # Each position predicts the next token.
        logits = out[:, :-1].reshape(-1, out.shape[-1])
        wanted = inputs[:, 1:].reshape(-1)
        valid = wanted != 0
        if not valid.any():
            fail("the target text in this batch has no tokens")
        logits, wanted = logits[valid], wanted[valid]
        return F.cross_entropy(logits, wanted), float((logits.argmax(-1) == wanted).sum()), wanted.numel()
    if task == "graph":
        labels, mask = target
        logits = out[mask]
        wanted = labels[mask]
        return F.cross_entropy(logits, wanted), float((logits.argmax(-1) == wanted).sum()), int(mask.sum())
    if task in ("image", "reconstruction"):
        return F.mse_loss(out, target.float()), 0.0, out.shape[0]
    if task == "regression":
        prediction = out.reshape(out.shape[0], -1)[:, 0]
        return F.mse_loss(prediction, target.float()), 0.0, out.shape[0]
    y = target
    if out.ndim > 2:
        # A sequence model returns [batch, time, classes]; the loss
        # wants [batch*time, classes].
        out = out.reshape(-1, out.shape[-1])
        y = y.repeat_interleave(out.shape[0] // y.shape[0])
    weights = CLASS_WEIGHTS.get("values")
    weight = torch.tensor(weights, device=out.device, dtype=out.dtype) if weights and len(weights) == out.shape[-1] else None
    predicted = out.argmax(-1)
    LAST_CLASSES.update(predicted=predicted.detach(), target=y.detach())
    return F.cross_entropy(out, y, weight=weight), float((predicted == y).sum()), y.shape[0]


def export(torch, model, req, step, validation, parameters, which):
    """The trained model as something to use: in `export/`, the weights alone
    (`last.pt`, and `best.pt` at the best validation loss), `model.py` to build
    them into, and `config.json` saying what they are. The checkpoints carry
    the optimizer as well and exist to resume; these exist to load."""
    if not MAIN:
        return
    folder = RUN_DIR / "export"
    folder.mkdir(exist_ok=True)
    if FINE_TUNE:
        import finetune  # noqa: PLC0415

        finetune.export(model, req, folder, which)
    else:
        weights = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        torch.save(weights, folder / f"{which}.pt")
        try:
            from safetensors.torch import save_file  # noqa: PLC0415

            save_file({k: v.contiguous() for k, v in weights.items()}, str(folder / f"{which}.safetensors"))
        except Exception:  # noqa: BLE001
            # Optional: `.pt` is always written, safetensors when it is installed.
            pass
        (folder / "model.py").write_text((RUN_DIR / "model.py").read_text())
    # Which rows were held out (split.json), how a raw sample becomes what the model
    # reads (preprocessing.json), and what each output means (labels.json).
    for name in ("split.json", "preprocessing.json", "labels.json", "tokenizer.json", "predict.py", "diagnosis.json"):
        if (RUN_DIR / name).exists():
            (folder / name).write_text((RUN_DIR / name).read_text())
    config = read_json("export/config.json", {}) or {}
    config.update(
        {
            "modelClass": req["modelClass"],
            "parameters": parameters,
            "inputShape": req["inputShape"],
            "inputKind": req.get("inputKind"),
            "vocabSize": req.get("vocabSize"),
            "task": task_of(req),
            "numClasses": req.get("numClasses"),
            f"{which}Step": step,
            "savedAt": now(),
        }
    )
    if FINE_TUNE:
        import finetune  # noqa: PLC0415

        config["fineTune"] = finetune.describe(req)
    if req.get("datasetPath"):
        config["dataset"] = req["datasetPath"]
    if req.get("validationFraction"):
        config["validationFraction"] = req["validationFraction"]
    if validation is not None:
        config["validationLoss"] = validation["loss"]
        if "accuracy" in validation:
            config["validationAccuracy"] = validation["accuracy"]
    write_json("export/config.json", config)
    if FINE_TUNE:
        (folder / "README.md").write_text(readme(config, "the fine-tuned published model", ["model/", "adapter/"]))
        return
    # The class the weights load into: the one the request names when model.py
    # defines it, or the one the run actually built (see `main`).
    source = (folder / "model.py").read_text()
    name = req["modelClass"]
    if f"class {name}" not in source:
        name = type(getattr(model, "module", model)).__name__
    (folder / "README.md").write_text(readme(config, name, sorted(p.name for p in folder.glob("*.pt"))))


def readme(config, class_name, weights):
    """What the exported model is, what it was trained on, how well it did on
    the part held out, and how to load it — from the run's own record. What was
    not measured is said to be, never filled in."""
    main = "best.pt" if "best.pt" in weights else weights[0]
    lines = [
        f"# {class_name}",
        "",
        f"Trained by NEURAX: {config.get('parameters', 0):,} parameters, task {config.get('task')}, "
        f"input {config.get('inputKind') or 'tensor'} of shape {config.get('inputShape')}.",
        "",
        "## Files",
        "",
    ]
    for w in weights:
        which = w[:-3]
        step = config.get(f"{which}Step")
        what = "the weights at the best validation loss" if which == "best" else "the weights at the end of the run"
        lines.append(f"- `{w}` — {what}" + (f", step {step}" if step is not None else "") + ".")
    lines += ["- `model.py` — the model the weights load into.", "- `config.json` — this run's record.", ""]
    lines += ["## Data", ""]
    dataset = config.get("dataset")
    lines.append(f"Trained on `{dataset}`." if dataset else "The dataset is not recorded.")
    held = read_json("export/split.json")
    fraction = config.get("validationFraction")
    if held and held.get("validation"):
        how = ("as the data's own split" if held.get("givenByTheData")
               else "each class in its proportion" if held.get("stratified") else "at random")
        lines.append(f"{held['validation']} samples were held out of training to validate on ({how}); "
                     f"{held['train']} were trained on. Which ones: `split.json`.")
        lines += [f"- {w}" for w in held.get("warnings") or []]
    elif fraction:
        lines.append(f"{round(float(fraction) * 100):g}% of the data was held out of training, to validate on.")
    lines += ["", "## Validation", ""]
    if config.get("validationLoss") is None:
        lines.append("Not measured: no part of the data was held out, or no validation ran.")
    else:
        lines.append(f"- Loss: {config['validationLoss']:.4g}")
        if config.get("validationAccuracy") is not None:
            lines.append(f"- Accuracy: {config['validationAccuracy']:.4g}")
    diagnosis = read_json("diagnosis.json")
    if diagnosis and diagnosis.get("findings"):
        lines += ["", "## How it trained", ""] + [f"- {f['message']}" for f in diagnosis["findings"]]
    task = config.get("task")
    if (RUN_DIR / "predict.py").exists():
        example = {
            "language_modeling": 'model.generate("The beginning of a text", max_new_tokens=40, temperature=0.8, top_k=40)',
            "regression": "model.predict({...})  # a row of the table, by column name → {'value': … in the target's units}",
        }.get(task, {"tokens": 'model.predict("a text")  # →', "image": 'model.predict("photo.jpg")  # →'}.get(
            config.get("inputKind"), "model.predict({...})  # a row of the table, by column name →") + " {'label', 'probability', 'scores'}")
        lines += ["", "## Use it", "",
                  "`predict.py` prepares a raw sample as training did (`preprocessing.json`, `tokenizer.json`) and "
                  "decodes the output (`labels.json`):", "", "```python", "from predict import Model", "",
                  "model = Model()", example, "```"]
    lines += [
        "",
        "## Load it",
        "",
        "```python",
        "import torch",
        f"from model import {class_name}",
        "",
        f"model = {class_name}()",
        f'model.load_state_dict(torch.load("{main}", weights_only=True))',
        "model.eval()",
        "```",
        "",
    ]
    return "\n".join(lines)


def memory_fields(torch, device):
    """Memory and telemetry, where the device reports them.

    A CPU run has no VRAM figures, and inventing them — reporting process RSS
    as though it were allocator state — would put a number in the Accuracy
    view that means something else entirely. Absent is the honest answer.
    """
    if device.type != "cuda":
        return {"vramAllocatedBytes": 0, "vramReservedBytes": 0}
    fields = {
        "vramAllocatedBytes": int(torch.cuda.memory_allocated()),
        "vramReservedBytes": int(torch.cuda.memory_reserved()),
    }
    try:
        fields["gpuUtilisationPct"] = float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001
        pass
    return fields


def prune_checkpoints(keep):
    """
    Keep the last `keep` checkpoints, delete the rest.

    Without this a run accumulates one checkpoint per interval forever. A
    100 000-step run checkpointing every 500 steps leaves 200 files; at the
    742 MB a mid-size model weighs, that is 148 GB from a single run — the
    disk fills, and the run dies from its own bookkeeping.

    Only the most recent ones are useful: resuming uses the last, and the one
    before it is insurance against the last being written when the power went.
    `keep` is part of the request, so a user who wants every checkpoint can
    have them; the default just refuses to be the reason a machine runs out of
    disk.
    """
    if keep <= 0:
        return
    files = sorted((RUN_DIR / "checkpoints").glob("step_*.pt"))
    for stale in files[:-keep]:
        try:
            stale.unlink()
            stale.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            # A checkpoint that cannot be deleted is not worth ending a run
            # over — the next prune will try again.
            pass


def build_optimiser(torch, model, req):
    """The optimizer the request asks for.

    This was AdamW at its library defaults whatever the studio sent, so
    choosing SGD or asking for weight decay changed the screen and nothing
    else. An optimizer this PyTorch does not have is refused here, before the
    first batch, rather than silently replaced by a different one: a run
    reporting AdamW's numbers under Lion's name is worse than a run that did
    not start.
    """
    name = str(req.get("optimizer") or "adamw").strip().lower()
    lr = float(req["learningRate"])
    decay = float(req.get("weightDecay", 0.1) or 0.0)
    # Weight decay pulls weight matrices toward zero; a bias or a norm's scale
    # pulled there only loses what it measures. They train without it.
    trained = [p for p in model.parameters() if p.requires_grad]
    params = [{"params": [p for p in trained if p.ndim >= 2], "weight_decay": decay},
              {"params": [p for p in trained if p.ndim < 2], "weight_decay": 0.0}]

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=decay)
    if name in ("sgd", "sgd_momentum"):
        # The studio's "SGD" is SGD with momentum, as its own label says.
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=decay)
    if name in ("lion", "adafactor"):
        # Shipped by newer PyTorch under these names; absent from older ones.
        factory = getattr(torch.optim, name.capitalize() if name == "lion" else "Adafactor", None)
        if factory is None:
            raise SystemExit(
                f"this PyTorch ({torch.__version__}) has no {name} optimizer — "
                "choose AdamW, Adam or SGD in the Parameters panel"
            )
        return factory(params, lr=lr, weight_decay=decay)
    raise SystemExit(f"unknown optimizer {name!r} — choose adamw, adam, sgd, lion or adafactor")


def build_lr_schedule(req, total_steps):
    """What the learning rate is at each step: warmup, then the schedule.

    Returns a function of the step count, applied to the optimizer before
    every update. The panel has offered a warmup and four schedules all along
    and the rate was flat for every run.
    """
    import math

    base = float(req["learningRate"])
    kind = str(req.get("lrScheduler") or "cosine").strip().lower()
    warmup = max(0, int(req.get("warmupSteps") or 0))
    # A warmup is a ramp at the start of a run, not most of it: at most a
    # tenth of the steps (the studio's plan applies the same cap, and says so).
    warmup = min(warmup, max(0, total_steps // 10))

    def at(step):
        if warmup and step < warmup:
            return base * (step + 1) / warmup
        if kind == "constant":
            return base
        remaining = max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / remaining))
        if kind == "cosine":
            return base * 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "linear":
            return base * (1.0 - progress)
        if kind == "inverse_sqrt":
            # Held at `base` through warmup, then decayed by 1/sqrt(step).
            return base * math.sqrt(max(1, warmup) / max(1, step + 1))
        return base

    return at


def save_checkpoint(torch, model, optimiser, step, exact, keep=3):
    """
    Write a checkpoint, and say honestly what kind it is.

    Weights and optimizer state resume the *training*. Only a checkpoint that
    also saved the RNG state resumes the *same run* — rerunning it produces
    the same batches in the same order. A tool that promises a reproducible
    record has to distinguish the two rather than quietly offer the weaker
    one, so the claim is written beside the file and the studio displays it.
    """
    if not MAIN:
        return
    path = RUN_DIR / "checkpoints" / f"step_{step:06d}.pt"
    if FINE_TUNE:
        import finetune  # noqa: PLC0415

        # What is trained, not the published weights already on the client's disk.
        state = finetune.trained_state(model)
    else:
        state = model.state_dict()
    torch.save(
        {
            "step": step,
            "model": state,
            "optimiser": optimiser.state_dict(),
            "rng": torch.get_rng_state(),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps({"step": step, "exactResume": bool(exact), "savedAt": now()}, indent=2)
    )
    prune_checkpoints(keep)


if __name__ == "__main__":
    main()
