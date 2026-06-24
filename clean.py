import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json, gc, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Clean version. Three fixes over the first pass.
#  1. Coverage. The training examples always show every value in every position
#     at least once, so every held-out code is decodable by composition. A model
#     that fails cannot blame missing information.
#  2. No familiarity confound. Composition is tested with held-out objects as the
#     query and held-out objects as every candidate, so preferring an object the
#     model already saw cannot track the right answer. Retrieval is tested with a
#     seen query and seen candidates, the symmetric control.
#  3. The overlap manipulation. Holding the composition demand fixed, vary whether
#     a distractor shares one symbol with the query code. comp_none gives three
#     distractors that share no symbol. comp_overlap swaps one in that shares a
#     single position with the query, a tempting surface match. A model that
#     composes checks both positions and is not fooled; a model that matches on
#     surface overlap gets pulled onto the overlapping distractor. The gap between
#     the two conditions is the isolated signature of decode-by-overlap.
# Answers are graded by the probability the model puts on each option letter, and
# every condition has a shuffle control: scramble the example codes so no rule
# exists, and a model genuinely using the code drops to chance.
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
K = 5
POOL_A = ["ka", "mi", "to", "ru", "so"]
POOL_B = ["va", "su", "ne", "lo", "da"]
LETTERS = ["A", "B", "C", "D"]
N_SEEDS = int(os.environ.get("CR_SEEDS", "16"))
QPS = int(os.environ.get("CR_QPS", "8"))  # queries per seed per condition
PATH = os.environ.get("CR_PATH", "results/clean.json")
OBJECTS = [(a, b) for a in range(K) for b in range(K)]
CONDS = ["retrieval", "comp_none", "comp_overlap"]


def code(o, pa, pb):
    return f"{POOL_A[pa[o[0]]]} {POOL_B[pb[o[1]]]}"


def make_split(rng):
    # Train of ten objects that covers every value in both positions.
    while True:
        objs = OBJECTS[:]
        rng.shuffle(objs)
        train = objs[:10]
        if {o[0] for o in train} == set(range(K)) and {o[1] for o in train} == set(range(K)):
            return train, objs[10:]


def candidates(cond, o, train, held, rng):
    if cond == "retrieval":
        pool = [x for x in train if x != o]
        rng.shuffle(pool)
        dist = pool[:3]
    else:
        others = [x for x in held if x != o]
        none = [x for x in others if x[0] != o[0] and x[1] != o[1]]
        one = [x for x in others if (x[0] == o[0]) ^ (x[1] == o[1])]
        rng.shuffle(none)
        rng.shuffle(one)
        if cond == "comp_none":
            if len(none) < 3:
                return None
            dist = none[:3]
        else:
            if len(one) < 1 or len(none) < 2:
                return None
            dist = one[:1] + none[:2]
    cands = [o] + dist
    rng.shuffle(cands)
    return cands


def build_prompt(examples, msg, cands):
    L = ["You are decoding a symbol code. Each object is a pair (a, b).",
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
    return int(np.argmax([max(logp[i].item() for i in lids[k]) for k in range(n)]))


def run_condition(tok, model, lids, cond):
    real, shuf = [], []
    for seed in range(N_SEEDS):
        rng = random.Random(seed * 100 + 7)
        pa, pb = list(range(K)), list(range(K))
        rng.shuffle(pa)
        rng.shuffle(pb)
        train, held = make_split(rng)
        source = train if cond == "retrieval" else held
        rng.shuffle(source)
        queries = [o for o in source][:QPS]
        for scramble in (False, True):
            if scramble:
                ex = [(o, f"{rng.choice(POOL_A)} {rng.choice(POOL_B)}") for o in train]
            else:
                ex = [(o, code(o, pa, pb)) for o in train]
            correct = total = 0
            for o in queries:
                cands = candidates(cond, o, train, held, rng)
                if cands is None:
                    continue
                pick = choose(tok, model, lids, build_prompt(ex, code(o, pa, pb), cands), len(cands))
                correct += (cands[pick] == o)
                total += 1
            if total:
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
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
            lids = letter_ids(tok)
            res = {}
            for cond in CONDS:
                res[cond] = run_condition(tok, model, lids, cond)
                r = res[cond]
                print(f"[{short}] {cond:13s} real {r['real']:.2f}  shuf {r['shuffled']:.2f}", flush=True)
            out[short] = res
            json.dump(out, open(PATH, "w"), indent=2)
        except Exception as e:
            print(f"[{short}] FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            for v in ("model", "tok"):
                if v in dir():
                    pass
            gc.collect()
            if DEV == "mps":
                torch.mps.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
