import os
os.environ["HF_HUB_OFFLINE"] = "1"

import random
from itertools import product
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Confirmatory check after stage1 came back at chance. Two worries with stage1:
# the answer was parsed out of free text (Llama scored 0.00, almost certainly a
# parse failure, not a real result), and the task may simply have been too hard.
# Here the answer is graded by the probability the model puts on each option
# letter, so there is nothing to parse, and the task runs in three tiers from a
# plain in-context lookup up to the kind of compositional generalization stage1
# asked for. Each tier keeps the shuffle control: scramble the example codes so
# no rule exists, and a model that was really using the code should fall to
# chance. If even the easy tiers show real == shuffled, the limit is real.
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-2-2B": "google/gemma-2-2b-it",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
}
SYM = ["ka", "mi", "to", "ru", "so", "ne"]
LETTERS = ["A", "B", "C", "D", "E", "F"]
N_SEEDS = 10

# name, attributes per object, values per attribute
TIERS = [
    ("lookup 1x4", 1, 4),  # show every mapping, decode a seen object, no generalization
    ("comp 2x2", 2, 2),    # one symbol per position, hold out one of four objects
    ("comp 2x3", 2, 3),    # train on six of nine, decode three held-out objects
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
    train, test = objs[:6], objs[6:9]
    return [(train, test, test)]


def build_prompt(examples, msg, cands):
    L = ["You are decoding a symbol code. Each object is a tuple of small numbers.",
         "Here are objects and the codes that stand for them:"]
    for o, m in examples:
        L.append(f'  {o} -> "{m}"')
    L.append(f'\nWhich object does the code "{msg}" stand for?')
    for i, o in enumerate(cands):
        L.append(f"  {LETTERS[i]}. {o}")
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


def choose(tok, model, lids, text, n):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    inp = tok(s, return_tensors="pt").to(DEV)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1]
    logp = torch.log_softmax(logits.float(), -1)
    scores = [max(logp[i].item() for i in lids[k]) for k in range(n)]
    return int(np.argmax(scores))


print(f"{'model':16s}{'tier':12s}{'real':>7s}{'shuffled':>10s}{'chance':>8s}")
for short, repo in MODELS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    lids = letter_ids(tok)
    for tier, n_attrs, n_values in TIERS:
        objs = [tuple(t) for t in product(range(n_values), repeat=n_attrs)]
        real, shuf = [], []
        chance = None
        for seed in range(N_SEEDS):
            rng = random.Random(seed)
            perms = [list(range(n_values)) for _ in range(n_attrs)]
            for p in perms:
                rng.shuffle(p)
            for scramble in (False, True):
                correct = total = 0
                for train, test, cands_pool in blocks_for(tier, objs, rng):
                    if scramble:
                        ex = [(o, " ".join(rng.choice(SYM[:n_values]) for _ in range(n_attrs))) for o in train]
                    else:
                        ex = [(o, encode(o, perms)) for o in train]
                    for o in test:
                        cands = cands_pool[:]
                        rng.shuffle(cands)
                        pick = choose(tok, model, lids, build_prompt(ex, encode(o, perms), cands), len(cands))
                        correct += (cands[pick] == o)
                        total += 1
                        chance = 1 / len(cands)
                (shuf if scramble else real).append(correct / total)
        print(f"{short:16s}{tier:12s}{np.mean(real):>7.2f}{np.mean(shuf):>10.2f}{chance:>8.2f}", flush=True)
    del model
    import gc
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
