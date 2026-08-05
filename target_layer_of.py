#!/usr/bin/env python3
"""Report the target layer of a published Jacobian lens without downloading it.

A fitted lens does not record which layer it was fitted toward — the upstream
library's `save()` writes no such field. But the target is recoverable from the
container, because the number of stacked Jacobians is a function of it. This
reads only the bytes needed to count them: a few kilobytes out of a file that
may be several gigabytes.

    python target_layer_of.py neuronpedia/jacobian-lens \
        qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt --n-layers 64

Two container formats are handled:

  *.safetensors  header is JSON at byte 0, length-prefixed. Exact shapes, and
                 any parameters the producer chose to embed, come straight out
                 of it. No inference needed.

  *.pt           a zip of pickled storages. The central directory gives every
                 entry's uncompressed size; the Jacobian stack is the dominant
                 one, and dividing by d_model^2 * itemsize gives the count.

**The count alone does not identify the target**, because producers disagree on
whether the stack includes the target itself. See `interpret()` below and the
README section "The near-collision". Pass --n-layers to get an interpretation;
without it you get the raw count and nothing more.

No dependencies beyond the standard library and `curl` on PATH.
"""

import argparse
import json
import re
import struct
import subprocess
import sys

ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4}


def fetch(url, start=None, end=None, timeout=120):
    """Range-request bytes. Returns b'' on failure rather than raising."""
    cmd = ["curl", "-sSL", "--max-time", str(timeout)]
    if start is not None:
        cmd += ["-r", f"{start}-{end if end is not None else ''}"]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True)
    return p.stdout


def content_length(url):
    out = subprocess.run(
        ["curl", "-sSLI", "--max-time", "60", url], capture_output=True
    ).stdout.decode("latin-1", "replace")
    sizes = [int(m) for m in re.findall(r"(?im)^(?:content-length|x-linked-size):\s*(\d+)", out)]
    return max(sizes) if sizes else None


def read_safetensors(url):
    """Exact tensor shapes and producer metadata from the JSON header."""
    raw = fetch(url, 0, 7)
    if len(raw) < 8:
        return None, "could not read the first 8 bytes"
    n = struct.unpack("<Q", raw[:8])[0]
    if not 0 < n < 100_000_000:
        return None, f"implausible header length {n} (an LFS pointer, or not safetensors)"
    body = fetch(url, 8, 8 + n - 1)
    try:
        hdr = json.loads(body.decode("utf-8"))
    except Exception as exc:
        return None, f"header did not parse as JSON: {exc}"

    meta = hdr.pop("__metadata__", {})
    tensors = {k: v for k, v in hdr.items()}
    stack = max(tensors.items(), key=lambda kv: _numel(kv[1]["shape"]), default=(None, None))
    name, spec = stack
    if name is None:
        return None, "header contained no tensors"
    shape = spec["shape"]
    return {
        "format": "safetensors",
        "tensor": name,
        "shape": shape,
        "dtype": spec["dtype"],
        "n_jacobians": shape[0] if len(shape) == 3 else None,
        "d_model": shape[1] if len(shape) == 3 else None,
        "metadata": meta,
    }, None


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def read_torch_zip(url, d_model=None, itemsize=2):
    """Count stacked Jacobians from the zip central directory of a .pt file."""
    total = content_length(url)
    if not total:
        return None, "server did not report a content length"

    tail = fetch(url, max(0, total - 131072), total - 1)
    idx = tail.rfind(b"PK\x05\x06")
    if idx < 0:
        return None, "no zip end-of-central-directory in the last 128 KB"

    n_entries = struct.unpack("<H", tail[idx + 10 : idx + 12])[0]
    cd_size = struct.unpack("<I", tail[idx + 12 : idx + 16])[0]
    cd_off = struct.unpack("<I", tail[idx + 16 : idx + 20])[0]

    cd = tail[-(total - cd_off) :] if cd_off >= total - len(tail) else fetch(url, cd_off, cd_off + cd_size - 1)

    entries, pos = [], 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        usize = struct.unpack("<I", cd[pos + 24 : pos + 28])[0]
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28 : pos + 34])
        name = cd[pos + 46 : pos + 46 + nlen].decode("latin-1", "replace")
        entries.append((name, usize))
        pos += 46 + nlen + elen + clen

    if not entries:
        return None, "central directory parsed to zero entries"

    data = [e for e in entries if "/data/" in e[0]] or entries
    biggest = max(e[1] for e in data)
    payload = sum(e[1] for e in data)

    out = {
        "format": "torch-zip",
        "zip_entries": n_entries,
        "storage_entries": len(data),
        "payload_bytes": payload,
        "largest_entry_bytes": biggest,
        "n_jacobians": None,
        "d_model": d_model,
    }
    if d_model:
        per = d_model * d_model * itemsize
        n_from_payload = payload / per
        out["n_jacobians"] = int(round(n_from_payload))
        out["exact"] = abs(n_from_payload - round(n_from_payload)) < 1e-6
        out["residual_bytes"] = payload - int(round(n_from_payload)) * per
    return out, None


def interpret(n_jacobians, n_layers):
    """Both conventions in the wild, and why the count alone cannot decide."""
    if not (n_jacobians and n_layers):
        return []
    out = []
    # anthropics/jacobian-lens: source_layers = range(target); target not stored.
    out.append(
        (
            "range(target), target NOT stored  [anthropics/jacobian-lens]",
            n_jacobians,
            "final" if n_jacobians == n_layers - 1 else
            "penultimate" if n_jacobians == n_layers - 2 else "other",
        )
    )
    # [target_block + 1, d, d] with J[target] = I; target IS stored.
    out.append(
        (
            "target+1 slots, J[target]=I stored  [agu18dec global-workspace]",
            n_jacobians - 1,
            "final" if n_jacobians - 1 == n_layers - 1 else
            "penultimate" if n_jacobians - 1 == n_layers - 2 else "other",
        )
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo id, e.g. neuronpedia/jacobian-lens")
    ap.add_argument("path", help="path to the lens file inside the repo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--n-layers", type=int, help="the subject model's num_hidden_layers")
    ap.add_argument("--d-model", type=int, help="hidden size; required to count inside a .pt")
    ap.add_argument("--itemsize", type=int, default=2, help="bytes per element in the .pt (default 2, fp16/bf16)")
    a = ap.parse_args()

    url = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/{a.path}"
    print(f"# {a.repo}/{a.path}\n# {url}\n")

    if a.path.endswith(".safetensors"):
        info, err = read_safetensors(url)
    else:
        info, err = read_torch_zip(url, a.d_model, a.itemsize)

    if err:
        print(f"  FAILED: {err}")
        return 1

    for k, v in info.items():
        if k == "metadata":
            continue
        print(f"  {k:22s} {v}")

    meta = info.get("metadata") or {}
    if meta:
        print("\n  producer metadata embedded in the file:")
        for k in sorted(meta):
            print(f"    {k:18s} {meta[k]}")
        if "target_block" in meta:
            print(f"\n  -> target_block is recorded directly: {meta['target_block']}. No inference needed.")

    readings = interpret(info.get("n_jacobians"), a.n_layers)
    if readings:
        print(f"\n  {info['n_jacobians']} stacked Jacobians on a {a.n_layers}-layer model reads as:")
        for label, target_idx, verdict in readings:
            print(f"    under {label}")
            print(f"      -> target layer {target_idx}  ({verdict})")
        print(
            "\n  The count alone does not decide between these. Check whether the last\n"
            "  slot is the identity: it is under the second convention, not the first."
        )
    elif not a.n_layers:
        print("\n  (pass --n-layers to interpret the count)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
