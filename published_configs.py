#!/usr/bin/env python3
"""Survey the target layer recorded by every config in the Neuronpedia lens family.

Answers three questions from bytes rather than from documentation:

  1. What target layer does each published lens record?
  2. Does any recorded fit command pass --target_layer explicitly?
  3. Which model directories ship no config at all?

Question 2 is the one that matters. A recorded `target_layer: null` tells you
the value; whether the flag was ever passed tells you whether it was chosen.

    python published_configs.py

Standard library plus `curl`. Fetches ~40 small YAML files.
"""

import json
import re
import subprocess
import sys
from collections import Counter

REPO = "neuronpedia/jacobian-lens"
API = f"https://huggingface.co/api/models/{REPO}?full=true"
RAW = f"https://huggingface.co/{REPO}/raw/main/"


def curl(url, timeout=60):
    return subprocess.run(
        ["curl", "-sSL", "--max-time", str(timeout), url], capture_output=True
    ).stdout.decode("utf-8", "replace")


def scalar(text, key, section=None):
    """Minimal YAML scalar lookup — enough for these flat files, no dependency."""
    body = text
    if section:
        m = re.search(rf"(?m)^{re.escape(section)}:\s*$", text)
        if not m:
            return None
        rest = text[m.end():]
        stop = re.search(r"(?m)^\S", rest)
        body = rest[: stop.start()] if stop else rest
    m = re.search(rf"(?m)^\s*{re.escape(key)}:\s*(.+?)\s*$", body)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def main():
    try:
        listing = json.loads(curl(API))
    except Exception as exc:
        print(f"could not list the repo: {exc}", file=sys.stderr)
        return 1

    files = [s["rfilename"] for s in listing.get("siblings", [])]
    model_dirs = sorted({f.split("/")[0] for f in files if "/" in f and not f.startswith(".")})
    cfgs = sorted(f for f in files if f.endswith("config.yaml"))
    with_cfg = {c.split("/")[0] for c in cfgs}
    without = [d for d in model_dirs if d not in with_cfg]

    print(f"model directories : {len(model_dirs)}")
    print(f"config.yaml files : {len(cfgs)}")
    if without:
        print(f"\nDIRECTORIES WITH NO CONFIG ({len(without)}):")
        for d in without:
            extra = [f for f in files if f.startswith(d + "/") and not f.endswith(".DS_Store")]
            print(f"  {d}")
            for e in extra:
                print(f"      {e}")
        print("\n  For these, nothing about the fit is documented. The parameters are")
        print("  recoverable only by measuring the artifact — see target_layer_of.py.")

    print(f"\n{'model':34s} {'target_layer':>13s} {'--target_layer passed?':>23s}")
    print("-" * 74)
    targets, flagged = Counter(), 0
    for c in cfgs:
        text = curl(RAW + c)
        model = c.split("/")[0]
        tl = scalar(text, "target_layer", "fit")
        has_flag = "--target_layer" in (scalar(text, "command") or "")
        flagged += has_flag
        targets[tl] += 1
        print(f"{model:34s} {str(tl):>13s} {str(has_flag):>23s}")

    print("-" * 74)
    print(f"\ntarget_layer values across {len(cfgs)} configs: {dict(targets)}")
    print(f"configs whose recorded command passes --target_layer explicitly: {flagged}")
    if targets.get("null") == len(cfgs) and flagged == 0:
        print(
            "\n  All null, and the flag is never passed. So the target was not chosen;\n"
            "  it was left to the library default, which resolves to the final layer\n"
            "  (jlens/fitting.py:79). Same outcome, different meaning: an omission,\n"
            "  not a decision."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
