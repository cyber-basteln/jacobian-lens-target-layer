# Which layer is your Jacobian lens fitted to?

It depends — and it likely depends on which document you read first. The library accompanying the
Global Workspace paper defaults to the final layer, and most published lenses follow it. The paper however
also mentions and prints its runs that used the penultimate one, and why that might be preferable,
and the best-known external replication follows the penultimate advice. Which is why two people can
end up using different lenses not knowing they are doing so, both ultimately citing the same paper.

We found this while building a J-lens fitting pipeline of our own. It is not a critique of anyone's
results, and as far as we can tell it changes very little — but it is real, it is undocumented, and
nobody appears to have written it down. So: here it is, with the checks to verify every claim.

---

## The divergence

| who | target layer | how we know |
|---|---|---|
| the paper's main text (L189) | **final** | *"a single d_model × d_model matrix per layer that maps from a source layer ℓ to the final layer L"* |
| the paper's ablations section (L1456, L1464) | penultimate *(both final and penultimate measured, penultimate seemingly preferable)* | *"The default lens on Sonnet 4.5, used throughout the paper … with z taken at the penultimate layer"*, and the reason given: *"We experimented with computing partial derivatives of the final-layer residual stream or the penultimate-layer residual stream … We observed that including the last layer can sometimes increase the number of noisy artifacts in lens-readouts. This may be because the final block is heavily specialized for calibrating next token predictions and contains less semantic content."* |
| the paper's pseudocode (L1526) | **final** | `# z[t] : residual stream at the target layer L (by default final)` |
| the paper's Figure 57 | **both are scored** — and final is the row labelled `(default)` | its ablation grid carries `all tokens (default)` and `all tokens, penultimate` as *separate rows*, so penultimate is a variation *against* the default |
| `anthropics/jacobian-lens` | **final** | `fitting.py:79` — `target = n_layers - 1 if target_layer is None else target_layer` |
| the neuronpedia published lens family, 37 configs | **final** | `target_layer: null` in 37/37, and `--target_layer` appears in 0/37 recorded commands |
| the 38th lens, `qwen3.6-27b` — fitted by the paper's authors, not by the host | **final** | no config ships; `CREDIT.md` credits Anthropic Interpretability. Measured from the artifact: 63 Jacobians on a 64-layer model |
| Nanda's replication | penultimate | *"by taking Jacobians to the penultimate layer on twenty-five prompts from the Pile"* |
| the meta-tokens post, by the same team | penultimate | same lens, same wording — *not* independent confirmation |

**One replication team** read the prose and fitted penultimate — the review, the meta-tokens post,
and that team's own lens uploads are all the same effort, not three confirmations of each other.
Everything downstream of the library's default is final, thirty-seven times.

**Figure 57 settles which is the paper's actual default.** Penultimate appears there as a named
variation *against* the default, so the default is final and the L1456 prose sentence is simply an
error — not one horn of a genuine ambiguity. That figure is a raster image, which is why reading the
paper as text does not resolve it.

---

## Nobody made a mistake

This is worth stating plainly, because the table above looks like an accusation and isn't one.

**Neuronpedia's model card is accurate.** It says the lenses were *"trained using Anthropic's Jacobian
Lens library"* — which is exactly and only what happened. It does not claim to reproduce the paper.
Every config records the full command, including a `target_layer: null` that anyone could have read.
In this whole picture, theirs is the description that overclaims least.

**The target was never chosen.** The recorded commands pass every parameter the operators thought
about — corpus, prompt count, sequence length, dtype, the entire stopping rule — and `--target_layer`
is absent from all 37. It fell through to a library default. That is what defaults are for.

That claim covers the 37 only. The 38th lens ships no config, so whether *its* target was chosen or
inherited cannot be established either way — all that is known is what it measures to.

**The replicators read the paper and implemented what it said.** Setting a parameter explicitly is
precisely how you never discover its default: if you pass `target_layer=-2`, the line that would have
told you otherwise never executes. Two of the three groups did not use the library at all — they wrote
their own pipelines from the paper's description, so its defaults were never in view.

The defect exists only in the seams between these parties. There is no step at which someone should
have caught it, which is why it is still here.

---

## What it costs

The paper measured this itself, on Sonnet 4.5, in Figures 57 and 58:

| metric (aggregation) | final *(default)* | penultimate |
|---|---|---|
| normalized pass@k AUC, mean | 0.76 | **0.79** |
| normalized pass@k AUC, median | 0.76 | 0.76 |
| causal-ablation KL, mean | 1.07 | 1.11 |
| causal-ablation KL, median | 1.14 | 1.12 |

Penultimate is mildly better on mean-aggregated pass@k and a wash on everything else. The gap sits
almost entirely in two evaluations — multilingual (0.66 → 0.71) and typo (0.76 → 0.83) — and vanishes
under median aggregation, which is what a difference carried by a minority of items looks like.

The paper's stated reason for preferring penultimate: the final block is *"heavily specialized for
calibrating next token predictions,"* and including it *"can sometimes increase the number of noisy
artifacts in lens-readouts."*

**So: small, on a frontier model.** Whether it is small on a 270M model is not something anyone has
measured, and we would not assume it transfers in either direction.

---

## The near-collision

A fitted lens records no target layer — the library's `save()` writes `J`, `n_prompts`,
`source_layers` and `d_model`, and no target field. For a **default** fit that is a distinction
without a difference: `source_layers` defaults to `range(target)`, so `max(source_layers) + 1` is the
target, and reading it is easier than what follows. It only stops working when a custom source list
was passed, or when the producer is not using this library at all.

Counting stacked Jacobians works in every case, but **the count alone is not enough**, because
producers disagree about whether the stack includes the target itself.

Run the checker on two lenses for the same model, one final and one penultimate. Both are public:

- **final** — [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens),
  `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`. Hosted by Neuronpedia;
  `CREDIT.md` credits Anthropic Interpretability. Fitted with the upstream library.
- **penultimate** —
  [`agu18dec/qwen3.6-27b-pile-jacobians`](https://huggingface.co/agu18dec/qwen3.6-27b-pile-jacobians),
  `n25-skip4-penultimate/jacobians.safetensors`, by a co-author of the meta-tokens post and part of
  the replication team above. Fitted with their own pipeline, not the upstream library, which is why
  the conventions differ.

```
$ python target_layer_of.py neuronpedia/jacobian-lens \
    qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt \
    --n-layers 64 --d-model 5120

  n_jacobians            63
  payload_bytes          3303014400
  residual_bytes         0

$ python target_layer_of.py agu18dec/qwen3.6-27b-pile-jacobians \
    n25-skip4-penultimate/jacobians.safetensors --n-layers 64

  shape                  [63, 5120, 5120]
  target_block           62
```

**Both report 63.** One targets layer 63, the other targets layer 62. Under
`source_layers = range(target)` the stack excludes the target; under `[target_block + 1, d, d]` it
includes it as an identity anchor. Same count, same index range, opposite answers.

Two ways to tell them apart: read the producer's embedded metadata if there is any, or check whether
the last slot is the identity matrix — it is under the second convention and is not under the first.

This comparison is only possible to show in a repository because the second repository
publishes **both** variants side by side — `n25-skip0-final/` and `n25-skip4-penultimate/` — and
records every construction parameter in the safetensors header (`target_block`, `skip_first`,
`n_prompts`, `dataset_id`, `t_max`, `git_commit`) rather than losing them at save time. It seems the only public matched pair, and the only producer here whose parameters survive inside the file. 

---

## What this is not

- **Not a claim that anyone's results are wrong.** The effect on the paper's own metrics is small, and
  no result in any of the cited work has been re-run here.
- **Not a claim that final is the wrong choice.** It is defensible, and the library documents it —
  `fit()` states the default in its API docs, not just in code. It is simply not what one sentence of
  the paper says, and noticing that the two disagree takes reading both.
- **Not a Neuronpedia problem.** Their documentation is accurate. See above.
- **Not measured at small scale.** Everything quantitative here is the paper's Sonnet 4.5 ablation.

---

## Run the checks

No dependencies beyond the Python standard library and `curl`.

```bash
python target_layer_of.py <hf-repo> <path-in-repo> --n-layers N [--d-model D]
python published_configs.py
```

`target_layer_of.py` reads only the container header — a few kilobytes out of a file that may be
several gigabytes. It handles `.safetensors` (exact shapes and any embedded producer metadata) and
`.pt` (counted from the zip central directory).

`published_configs.py` fetches every config in the Neuronpedia family, reports the target layer each
records, whether the recorded command ever passes `--target_layer`, and which model directories ship
no config at all.

---

## How this was checked

Every claim above traces to bytes fetched and parsed directly: the paper's raw HTML and its figure
PNGs, the library at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`, the 37 published config
YAMLs, the lens containers themselves by HTTP range request, and both mirrors of each replication
writeup. And by a human who fell down a rabbit hole unintentionally.

### If you check these claims with agents

Worth knowing before you start, because two audit passes here got the first two of these wrong:

**Pin the paper.** The line numbers above (`L1456`, `L1526`) are positions in one retrieval of a live
page, so they are only meaningful against the same bytes. As fetched: md5
`0db83fcfdf6fe4675c3e4506b9f2ab90`, 417,920 bytes. If yours differs, the page has changed and every
line number here is void — check the quoted text, not the number.

**Figure 57 is an image.** A fleet built out of `curl`, `grep` and text extraction will report the
figures as unreadable and then agree with itself that the question is unresolvable in the paper. Ours
did, twice, and the agreement looked like corroboration until the agent reports were read and
verified by human eyes.

**Don't let a summary stand in for a source.** Don't let them use summarizers. General advice for free.

---

## Sources

- Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams,
  Kantamneni, Ameisen, Batson & Lindsey, *Verbalizable Representations Form a Global Workspace in
  Language Models*, Transformer Circuits Thread, 2026 ·
  <https://transformer-circuits.pub/2026/workspace/index.html>
- `anthropics/jacobian-lens` (Apache-2.0) · <https://github.com/anthropics/jacobian-lens>
- `neuronpedia/jacobian-lens` · <https://huggingface.co/neuronpedia/jacobian-lens>
- External commentary, incl. the independent replication ·
  <https://www.anthropic.com/research/global-workspace>
- `agu18dec/qwen3.6-27b-pile-jacobians` · <https://huggingface.co/agu18dec/qwen3.6-27b-pile-jacobians>
