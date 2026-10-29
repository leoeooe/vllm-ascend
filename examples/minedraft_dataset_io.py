# SPDX-License-Identifier: Apache-2.0
"""Text-only, first-turn dataset loading for MineDraft NPU benchmarks.

This module imports neither vLLM nor torch. Network access happens only when
``download=True`` and no local source exists. Downloaded sources are cached as
JSON/JSONL with provenance sidecars; local files always work without datasets.
"""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET_NAMES = ("fastchat", "gsm8k", "sharegpt", "spec_bench")
DEFAULT_FILES = {
    "arena": ("arena.jsonl", "arena.json"),
    "mtbench": ("mt_bench.jsonl", "mt_bench.json"),
    "gsm8k": ("gsm8k_test.jsonl", "gsm8k.jsonl", "gsm8k.json"),
    "sharegpt": ("ShareGPT.json", "sharegpt.json", "sharegpt.jsonl"),
    "spec_bench": ("spec_bench.jsonl", "spec_bench.json"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_spec(name, fastchat_source, revision):
    revision = revision or "main"
    ref = quote(revision, safe="")
    if name == "fastchat" and fastchat_source == "arena":
        return {
            "kind": "huggingface_dataset",
            "repository": "lmsys/chatbot_arena_conversations",
            "url": "https://huggingface.co/datasets/lmsys/"
                   "chatbot_arena_conversations",
            "config": None, "split": "train", "revision_requested": revision,
        }
    if name == "gsm8k":
        return {
            "kind": "huggingface_dataset", "repository": "openai/gsm8k",
            "url": "https://huggingface.co/datasets/openai/gsm8k",
            "config": "main", "split": "test", "revision_requested": revision,
        }
    if name == "fastchat":
        url = (f"https://raw.githubusercontent.com/lm-sys/FastChat/{ref}/"
               "fastchat/llm_judge/data/mt_bench/question.jsonl")
    elif name == "sharegpt":
        url = ("https://huggingface.co/datasets/anon8231489123/"
               f"ShareGPT_Vicuna_unfiltered/resolve/{ref}/"
               "ShareGPT_V3_unfiltered_cleaned_split.json")
    else:
        url = (f"https://raw.githubusercontent.com/hemingkx/Spec-Bench/{ref}/"
               "data/spec_bench/question.jsonl")
    return {"kind": "remote_file", "url": url,
            "revision_requested": revision}


def _download_source(destination, source, name, fastchat_source):
    """Publish only a complete source file; retain its original source schema."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            if source["kind"] == "huggingface_dataset":
                try:
                    from datasets import load_dataset
                except ImportError as exc:
                    raise RuntimeError(
                        "Downloading Arena/GSM8K requires the datasets package; "
                        "install it or supply a local JSON/JSONL file") from exc
                try:
                    dataset = load_dataset(
                        source["repository"], name=source["config"],
                        split=source["split"],
                        revision=source["revision_requested"])
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot load {source['repository']}. Check network and "
                        "Hugging Face access; Arena requires accepting its "
                        "dataset terms. A local JSON/JSONL file is also supported."
                    ) from exc
                source["huggingface_fingerprint"] = getattr(
                    dataset, "_fingerprint", None)
                for row in dataset:
                    output.write((json.dumps(row, ensure_ascii=False)
                                  + "\n").encode("utf-8"))
            else:
                request = Request(source["url"],
                                  headers={"User-Agent": "MineDraft-benchmark"})
                with urlopen(request, timeout=60) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
        if not temporary.stat().st_size:
            raise ValueError(f"Downloaded dataset is empty: {source['url']}")
        # No raw prompts or authentication tokens are stored in the sidecar.
        provenance = {"dataset": name, "fastchat_source": fastchat_source,
                      "source": source, "sha256": sha256_file(temporary)}
        os.replace(temporary, destination)
        destination.with_name(destination.name + ".source.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_file(name, path, data_dir, download, fastchat_source, revision):
    key = fastchat_source if name == "fastchat" else name
    source = None
    if path is not None:
        selected = Path(path).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Dataset file does not exist: {selected}")
    else:
        directory = Path(data_dir).expanduser().resolve()
        selected = next((directory / item for item in DEFAULT_FILES[key]
                         if (directory / item).is_file()), None)
        if selected is None:
            if not download:
                expected = ", ".join(DEFAULT_FILES[key])
                raise FileNotFoundError(
                    f"No {name} source in {directory}; expected {expected}. "
                    "Supply a local file or explicitly enable --download-data")
            selected = directory / DEFAULT_FILES[key][0]
            source = _source_spec(name, fastchat_source, revision)
            _download_source(selected, source, name, fastchat_source)

    checksum = sha256_file(selected)
    sidecar = selected.with_name(selected.name + ".source.json")
    if sidecar.is_file():
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        if provenance.get("dataset") != name:
            raise ValueError(f"Dataset identity disagrees with {sidecar}")
        if (name == "fastchat"
                and provenance.get("fastchat_source") != fastchat_source):
            raise ValueError(f"FastChat source disagrees with {sidecar}")
        if provenance.get("sha256") != checksum:
            raise ValueError(f"Dataset changed since its provenance was saved: "
                             f"{selected}")
        source = provenance["source"]
        if revision and source.get("revision_requested") != revision:
            raise ValueError("Cached source has a different revision; use a "
                             "separate data directory for the requested revision")
    elif revision:
        raise ValueError("Cannot verify the requested revision for a local file "
                         "without a source sidecar; omit revision and record its "
                         "SHA256, or download into a separate data directory")
    return selected, {
        "path": str(selected), "sha256": checksum,
        "origin": source or {"kind": "local_file", "revision_requested": None},
    }


def _read_rows(path, counters):
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    counters["blank_lines"] += 1
                    continue
                counters["source_rows"] += 1
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError:
                    counters["malformed_json_rows"] += 1
        return
    with path.open("r", encoding="utf-8-sig") as stream:
        rows = json.load(stream)
    if isinstance(rows, dict):
        # Common JSON exports wrap the selected split in a named list.
        candidates = [key for key in ("test", "data", "records")
                      if isinstance(rows.get(key), list)]
        if len(candidates) == 1:
            rows = rows[candidates[0]]
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array or JSONL records: {path}")
    for row_number, row in enumerate(rows, 1):
        counters["source_rows"] += 1
        yield row_number, row


def _has_media(row):
    return any(row.get(key) for key in (
        "image", "images", "image_url", "video", "audio", "multi_modal_data"))


def _conversation_pair(conversation):
    if not isinstance(conversation, list) or not conversation:
        return None, None, "invalid_conversation"
    first = conversation[0]
    if not isinstance(first, dict):
        return None, None, "invalid_conversation"
    # Do not silently discard a system instruction or a leading assistant turn.
    if first.get("role", first.get("from", "user")) not in ("user", "human"):
        return None, None, "unsupported_leading_role"
    if _has_media(first):
        return None, None, "multimodal"
    prompt = first.get("content", first.get("value"))
    reference = None
    if len(conversation) > 1 and isinstance(conversation[1], dict):
        second = conversation[1]
        if second.get("role", second.get("from", "assistant")) in (
                "assistant", "gpt"):
            reference = second.get("content", second.get("value"))
    return prompt, reference, None


def _candidate_rows(name, row, row_number, fastchat_source):
    if not isinstance(row, dict):
        yield "", None, None, None, "non_object_row"
        return
    if _has_media(row):
        yield "", None, None, None, "multimodal"
        return
    raw_id = row.get("id", row.get("question_id", row_number))
    category = row.get("category")
    if not isinstance(category, str):
        category = None
    if name == "fastchat" and fastchat_source == "arena" and (
            "conversation_a" in row or "conversation_b" in row):
        # Match MineDraft's convert_datasets.py: retain both Arena sides.
        for side in ("a", "b"):
            prompt, reference, error = _conversation_pair(
                row.get("conversation_" + side))
            yield f"{raw_id}:{side}", prompt, reference, category, error
    elif "conversations" in row or "messages" in row:
        prompt, reference, error = _conversation_pair(
            row.get("conversations", row.get("messages")))
        yield str(raw_id), prompt, reference, category, error
    elif name == "gsm8k":
        yield str(raw_id), row.get("question"), row.get("answer"), category, None
    elif name == "spec_bench" or (name == "fastchat"
                                  and fastchat_source == "mtbench"):
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            yield str(raw_id), None, None, category, "invalid_turns"
        else:
            yield str(raw_id), turns[0], None, category, None
    else:
        yield str(raw_id), None, None, category, "unsupported_schema"


def load_dataset_records(name, *, path=None, data_dir="minedraft-data",
                         download=False, fastchat_source="arena", revision=None):
    """Return ``(records, metadata)`` without sampling, shuffling or tokenizing.

    Each record has id, prompt, reference and category. IDs include source row
    numbers to distinguish repeated original IDs, including Arena A/B sides.
    Natural duplicate prompts remain in the source; rows are never cycled to
    fabricate a requested sample count. All conversation datasets use the first
    user turn only. GSM8K downloads use the main/test split. MT-Bench loading is
    prompt-only throughput evaluation, not the official multi-turn judge.
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"Unsupported dataset: {name}; choose {DATASET_NAMES}")
    if fastchat_source not in ("arena", "mtbench"):
        raise ValueError("fastchat_source must be arena or mtbench")
    selected, source = _resolve_file(
        name, path, data_dir, download, fastchat_source, revision)
    counters = Counter()
    skipped = Counter()
    records = []
    for row_number, row in _read_rows(selected, counters):
        for raw_id, prompt, reference, category, error in _candidate_rows(
                name, row, row_number, fastchat_source):
            counters["candidate_records"] += 1
            if error:
                skipped[error] += 1
                continue
            if not isinstance(prompt, str) or not prompt.strip():
                skipped["empty_or_non_text_prompt"] += 1
                continue
            if any(marker in prompt for marker in ("<image>", "<video>",
                                                    "<audio>")):
                skipped["multimodal"] += 1
                continue
            records.append({
                "id": f"{name}:{row_number}:{raw_id}", "prompt": prompt,
                "reference": reference if isinstance(reference, str) else None,
                "category": category,
            })
    skipped["malformed_json"] += counters["malformed_json_rows"]
    metadata = {
        "name": name,
        "fastchat_source": fastchat_source if name == "fastchat" else None,
        "source": source,
        "source_rows": counters["source_rows"],
        "candidate_records": counters["candidate_records"],
        "accepted_records": len(records),
        "blank_lines": counters["blank_lines"],
        "skipped_by_reason": dict(skipped),
        "turn_policy": "first_user_turn_only; no assistant text in prompt",
        "arena_sides": "a_and_b" if name == "fastchat"
                       and fastchat_source == "arena" else None,
        "gsm8k_download_split": "main/test" if name == "gsm8k" else None,
    }
    if not records:
        raise ValueError(f"No usable text prompts in {selected}; skipped={dict(skipped)}")
    return records, metadata
