"""
133_capture_field_dump_multiclass.py   [K-class one-pass capture: residuals + PER-CLASS gradients]
==================================================================================================
Multiclass analog of 119. The 2-class capture stored ONE gradient per layer:
d(logit_beta - logit_alpha)/dh. For K classes there is no single "margin", and the
usage axis u can be defined several ways (mean_others / max_others / softmax). To let
us reconstruct u under ANY formula post-hoc WITHOUT re-running the model, this script
stores the K PER-CLASS gradients d(logit_c)/dh at every layer. Any usage formula is then
a linear combination of these stored gradients, computed in the atlas script.

Captured (decision = answer position; tap(L) = input of block L+1, final = input of norm):
  per layer L in 0..n_layers-1:
      res_L{L:02d}.npy        (n, d)      float32   residual at answer position
      grad_class{c}_L{L:02d}.npy (n, d)   float32   d(logit of class c)/d h^(L), for c in 0..K-1
  meta.npz:
      y                  (n,)    int       0..K-1 correct class index
      class_tokens       (K,)    int       answer token id for each class
      class_names        (K,)    str       e.g. ['electron','neutron','photon','proton']
      class_logits       (n, K)  float     the K class-logit values per prompt (for max_others etc.)
      baseline_top1      (n,)    int       argmax token id of the clean run
      baseline_correct   (n,)    int       1 if top1 == the correct class token
      wU_class           (d, K)  float     unembedding vector of each class token (build gamma downstream)
      d, n_layers, K, model_name, prompts_sha
  families.json          wording_family per prompt (split reconstruction downstream)

Balancing: by default keeps 4 classes (electron, neutron, photon, proton) subsampled to
--per_class (default 80) each => 320 prompts, balanced. positron/muon excluded by default
(too few for clean Fisher); pass --classes to override, --per_class 0 to keep all.

Two regimes: run once with --corpus <raw.jsonl> and once with --corpus <scaffold.jsonl>
(few-shot version), writing to two out dirs. Scaffold exemplars must NOT reveal the query
answer (same control as alpha/beta) — caller's responsibility in corpus construction.

Cost: K backward passes per prompt (one per class logit). 320 prompts x 4 classes = 1280
backward passes per regime; minutes-to-~hour on one A100. Disk: ~K x 0.4 GB per regime.

SELF-TEST (no torch / no repo):  python 133_capture_field_dump_multiclass.py --self_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("capture133")


# =====================================================================
# Pure-numpy helpers (exercised by --self_test)
# =====================================================================
def unit_raw(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def finite_diff_grad(f, h, eps=1e-5):
    g = np.zeros_like(h)
    for j in range(len(h)):
        e = np.zeros_like(h); e[j] = eps
        g[j] = (f(h + e) - f(h - e)) / (2 * eps)
    return g


def sha16(path):
    hh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hh.update(chunk)
    return hh.hexdigest()[:16]


def balance_indices(labels, per_class, classes, rng):
    """Return indices keeping only `classes`, subsampled to `per_class` each (0 = keep all)."""
    by = defaultdict(list)
    for i, c in enumerate(labels):
        if c in classes:
            by[c].append(i)
    keep = []
    for c in classes:
        idx = by.get(c, [])
        if per_class and len(idx) > per_class:
            idx = list(rng.choice(idx, size=per_class, replace=False))
        keep.extend(idx)
    return sorted(keep)


def usage_from_class_grads(class_grads, correct_idx, all_logits, formula="mean_others"):
    """Reconstruct the per-prompt usage gradient from stored per-class gradients.
    class_grads: (K, d) the K per-class gradients for ONE prompt at ONE layer.
    correct_idx: int, the correct class index for this prompt.
    all_logits:  (K,) the K class-logit values for this prompt (for max_others).
    Returns the usage gradient (d,) under the chosen formula. This is the function the
    atlas uses; kept here so the capture and the analysis share one definition."""
    K = class_grads.shape[0]
    gc = class_grads[correct_idx]
    others = [j for j in range(K) if j != correct_idx]
    if formula == "mean_others":
        return gc - class_grads[others].mean(0)
    if formula == "max_others":
        j = others[int(np.argmax(all_logits[others]))]
        return gc - class_grads[j]
    if formula == "softmax":
        # gradient of log p_correct = grad logit_correct - sum_j p_j grad logit_j
        p = np.exp(all_logits - all_logits.max()); p = p / p.sum()
        return gc - (p[:, None] * class_grads).sum(0)
    raise ValueError(f"unknown formula {formula}")


# =====================================================================
# Self-test
# =====================================================================
def self_test():
    rng = np.random.default_rng(0)

    # balancing: 6 classes uneven -> 4 classes at 80 each
    labels = ([0] * 80 + [1] * 135 + [2] * 152 + [3] * 80 + [4] * 20 + [5] * 20)
    rng.shuffle(labels)
    keep = balance_indices(labels, per_class=80, classes=[0, 1, 2, 3], rng=np.random.default_rng(1))
    kept = [labels[i] for i in keep]
    from collections import Counter
    cc = Counter(kept)
    assert set(cc) == {0, 1, 2, 3} and all(v == 80 for v in cc.values()), f"balance failed: {cc}"

    # usage formulas: build toy per-class gradients aligned to known directions
    d, K = 16, 4
    dirs = [unit_raw(rng.standard_normal(d)) for _ in range(K)]
    class_grads = np.stack(dirs)                      # (K,d): grad of class c = its direction
    logits = np.array([0.2, 2.0, 0.5, -1.0])          # class 1 is the top competitor
    correct = 0
    u_mean = usage_from_class_grads(class_grads, correct, logits, "mean_others")
    # mean_others: gc - mean(others). Check it equals dir0 - mean(dir1,2,3)
    expect = dirs[0] - np.stack([dirs[1], dirs[2], dirs[3]]).mean(0)
    assert np.allclose(u_mean, expect), "mean_others formula mismatch"
    u_max = usage_from_class_grads(class_grads, correct, logits, "max_others")
    assert np.allclose(u_max, dirs[0] - dirs[1]), "max_others must subtract the top-logit competitor (class 1)"
    u_soft = usage_from_class_grads(class_grads, correct, logits, "softmax")
    p = np.exp(logits - logits.max()); p /= p.sum()
    assert np.allclose(u_soft, dirs[0] - (p[:, None] * class_grads).sum(0)), "softmax formula mismatch"
    # the three formulas should generally give DIFFERENT directions (the robustness point)
    assert abs(unit_raw(u_mean) @ unit_raw(u_max)) < 0.999, "formulas should differ (else no robustness test)"

    # per-class gradient calculus: a stored class-grad supports first-order logit prediction
    a = dirs[2]
    f = lambda h: float(h @ a)                          # toy: logit of class 2 = h . dir2
    h0 = rng.standard_normal(d)
    g = finite_diff_grad(f, h0)
    assert abs(unit_raw(g) @ a) > 0.999, "finite-diff grad recovers the class direction"
    delta = 0.3 * rng.standard_normal(d)
    assert abs((g @ delta) - (f(h0 + delta) - f(h0))) < 1e-4, "class-grad supports first-order prediction"

    print("[self_test] OK — balancing 6->4 at 80, three u-formulas distinct & correct, "
          "per-class gradient calculus. pass.")


# =====================================================================
# Real run
# =====================================================================
def _chain(o, p):
    for x in p.split("."):
        o = getattr(o, x)
    return o


def first_token_id(tok, answer):
    """answer like ' electron' -> first sub-token id (leading space preserved)."""
    ids = tok.encode(answer, add_special_tokens=False)
    return int(ids[0])


def run_real(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw_prompts = [json.loads(l) for l in open(args.corpus)]
    # class label per prompt from correct_answer (stripped, lowercased)
    def cls_of(p):
        return p["correct_answer"].strip().lower()
    classes = [c.strip().lower() for c in args.classes]
    labels_all = [cls_of(p) for p in raw_prompts]
    keep = balance_indices(labels_all, args.per_class, classes, rng)
    prompts = [raw_prompts[i] for i in keep]
    nP = len(prompts)
    # map class name -> index in the SAME order as args.classes
    cidx = {c: k for k, c in enumerate(classes)}
    y = np.array([cidx[cls_of(p)] for p in prompts], np.int64)
    K = len(classes)
    from collections import Counter
    logger.info("balanced corpus: %d prompts, K=%d classes %s, per-class counts %s",
                nP, K, classes, dict(Counter(int(v) for v in y)))

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to(args.device).eval()
    blocks = _chain(model, "model.layers"); n_layers = len(blocks); norm_mod = _chain(model, "model.norm")
    d = model.config.hidden_size
    last = n_layers - 1
    layers = list(range(n_layers))

    # class answer token ids: prefer a leading-space variant ' electron'
    class_tokens = np.array([first_token_id(tok, " " + c) for c in classes], np.int64)
    W_U = _chain(model, "lm_head").weight.detach().float().cpu().numpy()
    wU_class = np.stack([W_U[t] for t in class_tokens], axis=1).astype(np.float32)  # (d, K)
    logger.info("class tokens: %s", {classes[k]: int(class_tokens[k]) for k in range(K)})

    for p_ in model.parameters():
        p_.requires_grad_(True)

    def tap(L):
        return blocks[L + 1] if L < last else norm_mod

    res = {L: np.zeros((nP, d), np.float32) for L in layers}
    # per-class gradients: grad[c][L] is (nP, d)
    grad = {c: {L: np.zeros((nP, d), np.float32) for L in layers} for c in range(K)}
    class_logits = np.zeros((nP, K), np.float64)
    baseline_top1 = np.zeros(nP, np.int64)
    baseline_correct = np.zeros(nP, np.int64)
    families = []

    for i, p in enumerate(prompts):
        inp = tok([p["prompt"]], return_tensors="pt").to(args.device)
        # one forward, capture residuals once; then K backward passes (retain_graph)
        keep_acts = {}; handles = []
        for L in layers:
            def mk(L=L):
                def pre(m, a):
                    a[0].retain_grad(); keep_acts[L] = a[0]; return None
                return pre
            handles.append(tap(L).register_forward_pre_hook(mk(), with_kwargs=False))
        try:
            o = model(**inp, use_cache=False)
            row = o.logits[0, -1, :]
            rowf = row.detach().float()
            for k in range(K):
                class_logits[i, k] = float(rowf[int(class_tokens[k])])
            baseline_top1[i] = int(rowf.argmax().item())
            baseline_correct[i] = int(baseline_top1[i] == int(class_tokens[int(y[i])]))
            # residuals (same for all backward passes) — read now
            for L in layers:
                res[L][i] = keep_acts[L].detach()[0, -1, :].float().cpu().numpy()
            # K backward passes, one per class logit, retaining graph between them
            for k in range(K):
                model.zero_grad(set_to_none=True)
                for L in layers:
                    if keep_acts[L].grad is not None:
                        keep_acts[L].grad = None
                row[int(class_tokens[k])].backward(retain_graph=(k < K - 1))
                for L in layers:
                    g = keep_acts[L].grad
                    grad[k][L][i] = (g[0, -1, :].float().cpu().numpy() if g is not None else 0.0)
        finally:
            for h in handles:
                h.remove()
        model.zero_grad(set_to_none=True)
        families.append(p.get("wording_family", p.get("surface_family", f"__nofam_{i}")))
        if (i + 1) % 25 == 0:
            logger.info("  capture %d/%d (x%d backward each)", i + 1, nP, K)

    # ---------- consistency checks ----------
    # each class gradient at the final tap must be nonzero (hook wiring) and the correct-class
    # gradient should, on average, align with that class's unembedding direction (modulo final norm)
    for k in range(K):
        gn = np.linalg.norm(grad[k][last], axis=1)
        assert float(gn.min()) > 0, f"zero final-tap gradient for class {k} — hook wiring broken"
    # build mean usage (mean_others) at final tap and check it points toward the
    # correct-minus-mean unembedding contrast (sanity that grads mean what we say)
    u_final = np.zeros((nP, d), np.float64)
    for i in range(nP):
        cg = np.stack([grad[k][last][i] for k in range(K)]).astype(np.float64)
        u_final[i] = usage_from_class_grads(cg, int(y[i]), class_logits[i], "mean_others")
    gamma_contrast = np.zeros(d, np.float64)
    for i in range(nP):
        others = [j for j in range(K) if j != int(y[i])]
        gamma_contrast += wU_class[:, int(y[i])] - wU_class[:, others].mean(1)
    cf = float(unit_raw(u_final.mean(0)) @ unit_raw(gamma_contrast))
    logger.info("sanity: cos(mean usage_final, mean class-contrast unembed) = %+.3f (high => valid)", cf)
    acc = float(baseline_correct.mean())
    logger.info("clean top-1 accuracy on balanced set: %.3f", acc)

    # ---------- dump ----------
    for L in layers:
        np.save(out / f"res_L{L:02d}.npy", res[L])
        for k in range(K):
            np.save(out / f"grad_class{k}_L{L:02d}.npy", grad[k][L])
    np.savez(out / "meta.npz",
             y=y, class_tokens=class_tokens, class_names=np.array(classes),
             class_logits=class_logits, baseline_top1=baseline_top1,
             baseline_correct=baseline_correct, wU_class=wU_class,
             d=d, n_layers=n_layers, K=K, per_class=args.per_class,
             clean_accuracy=acc, cos_usage_final_contrast=cf,
             model_name=args.model_name,
             prompts_sha=f"{Path(args.corpus).name}:{sha16(args.corpus)}")
    json.dump(families, open(out / "families.json", "w"))
    logger.info("dumped %d layers x (1 res + %d class-grads) to %s", n_layers, K, out)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--corpus", default="data/prompts/physics_internal_candidate_selection_v2_train.jsonl")
    p.add_argument("--out_dir", default="data/analysis/runD_v2/particles4/field_dump")
    p.add_argument("--classes", nargs="+", default=["electron", "neutron", "photon", "proton"],
                   help="class names (order = label index); default 4 balanced classes")
    p.add_argument("--per_class", type=int, default=80, help="subsample each class to this many (0=keep all)")
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        self_test(); return
    run_real(args)


if __name__ == "__main__":
    main()
