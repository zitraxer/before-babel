import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json, gc, random
from itertools import product
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# How do small instruct models decode an in-context symbol code: by looking up
# the surface form they were shown, or by composing the rule position by
# position? Three tiers separate the two. lookup shows every mapping and asks
# for a seen object, so surface lookup alone is enough. comp 2x2 holds out one
# object whose code is built from two symbols that each appear in the shown
# examples but never together, so a model that matches on shared symbols is
# actively pulled onto a wrong shown candidate, and overlap shows up as below
# chance. comp 2x3 holds out four objects with more of the rule covered, which
# is where real composition can appear. Every tier has a shuffle control:
# scramble the example codes so no rule exists, and a model that was really
# using the code drops to chance. The reasoning condition lets the model think
# before it answers, to see whether composition can be coaxed out. Answers are
# graded by the probability the model puts on each option letter, so there is
# nothing to parse.
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "Falcon3-1B": "tiiuae/Falcon3-1B-Instruct",
    "OLMo-2-1B": "allenai/OLMo-2-0425-1B-Instruct",
    "Granite-3.1-2B": "ibm-granite/granite-3.1-2b-instruct",
    "StableLM-2-1.6B": "stabilityai/stablelm-2-1_6b-chat",
    "Gemma-2-2B": "google/gemma-2-2b-it",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
}
SYM = ["ka", "mi", "to", "ru", "so", "ne"]
LETTERS = ["A", "B", "C", "D"]
N_SEEDS = 8
PATH = "results/code_reading.json"

# name, attributes per object, values per attribute, reasoning condition too
TIERS = [
    ("lookup 1x4", 1, 4, False),
    ("comp 2x2", 2, 2, True),
    ("comp 2x3", 2, 3, True),
]


def encode(obj, perms):
    return " ".join(SYM[perms[i][v]] for i, v in enumerate(obj))


def blocks_for(tier, objs, rng):
    objs = objs[:]
    rng.shuffle(objs)
    if tier.startswith("lookup"):
        return [(objs, objs, objs)]
    if tier == "comp 2x2":
        return [([o for o in objs if o != held], [held], objs) for held in objs]
    train, test = objs[:5], objs[5:9]  # four held-out, candidates are all held-out
    return [(train, test, test)]


def build_prompt(examples, msg, cands, reason):
    L = ["You are decoding a symbol code. Each object is a tuple of small numbers.",
         "Here are objects and the codes that stand for them:"]
    for o, m in examples:
        L.append(f'  {o} -> "{m}"')
    L.append(f'\nWhich object does the code "{msg}" stand for?')
    for i, o in enumerate(cands):
        L.append(f"  {LETTERS[i]}. {o}")
    if reason:
        L.append("\nReason briefly about each position, then end with your answer.")
    else:
        L.append("\nAnswer with a single letter.")
    return "\n".join(L)


def letter_ids(tok):
    ids = []
    for ch in LETTERS:
        toks = []
        for s in (ch, " " + ch):
            t = tok.encode(s, add_special_tokens=False)
            if t:
                toks.append(t[0])
        ids.append(toks)
    return ids


def score_letters(model, lids, ids, n):
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    logp = torch.log_softmax(logits.float(), -1)
    return int(np.argmax([max(logp[i].item() for i in lids[k]) for k in range(n)]))


def choose(tok, model, lids, text, n, reason):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors="pt").to(DEV).input_ids
    if not reason:
        return score_letters(model, lids, ids, n)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=90, do_sample=False, pad_token_id=tok.eos_token_id)
    tail = tok("\nAnswer:", add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    return score_letters(model, lids, torch.cat([gen, tail], dim=1), n)


def run_condition(tok, model, lids, tier, n_attrs, n_values, reason):
    objs = [tuple(t) for t in product(range(n_values), repeat=n_attrs)]
    real, shuf = [], []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        perms = [list(range(n_values)) for _ in range(n_attrs)]
        for p in perms:
            rng.shuffle(p)
        for scramble in (False, True):
            correct = total = 0
            for train, test, pool in blocks_for(tier, objs, rng):
                if scramble:
                    ex = [(o, " ".join(rng.choice(SYM[:n_values]) for _ in range(n_attrs))) for o in train]
                else:
                    ex = [(o, encode(o, perms)) for o in train]
                for o in test:
                    cands = pool[:]
                    rng.shuffle(cands)
                    pick = choose(tok, model, lids, build_prompt(ex, encode(o, perms), cands, reason), len(cands), reason)
                    correct += (cands[pick] == o)
                    total += 1
            (shuf if scramble else real).append(correct / total)
    return {"real": float(np.mean(real)), "shuffled": float(np.mean(shuf)), "chance": 1 / len(LETTERS)}


def main():
    os.makedirs("results", exist_ok=True)
    out = json.load(open(PATH)) if os.path.exists(PATH) else {}
    for short, repo in MODELS.items():
        if short in out:
            print(f"[{short}] done, skipping", flush=True)
            continue
        print(f"[{short}] loading", flush=True)
        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
        lids = letter_ids(tok)
        res = {"plain": {}, "reason": {}}
        for tier, na, nv, do_reason in TIERS:
            res["plain"][tier] = run_condition(tok, model, lids, tier, na, nv, False)
            r = res["plain"][tier]
            print(f"[{short}] {tier:11s} plain  real {r['real']:.2f}  shuf {r['shuffled']:.2f}", flush=True)
            if do_reason:
                res["reason"][tier] = run_condition(tok, model, lids, tier, na, nv, True)
                r = res["reason"][tier]
                print(f"[{short}] {tier:11s} reason real {r['real']:.2f}  shuf {r['shuffled']:.2f}", flush=True)
        out[short] = res
        json.dump(out, open(PATH, "w"), indent=2)
        del model
        gc.collect()
        if DEV == "mps":
            torch.mps.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
