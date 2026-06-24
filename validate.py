import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json, gc, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Validation. The clean run shows the larger models score above chance on a
# four-way choice, but that alone does not prove they read both symbols. Three
# stronger checks.
#  swap: a causal test. Present an object, its position-1 twin (same b, different
#    a) and its position-2 twin (same a, different b). All three share a symbol
#    pairwise, so surface overlap cannot tell them apart. Ask for each of the
#    three codes. Require all three correct (swap_all3). A model that genuinely
#    composes flips its answer in exactly the position whose symbol changed; a
#    surface matcher cannot.
#  hard: every distractor is half-right (shares one position with the answer), so
#    only decoding both symbols picks the target.
#  failure mode: when wrong on the swap test, did the model pick a half-right twin
#    (an overlap error) or the unrelated filler (noise)? This separates an
#    overlap matcher from a guesser.
# Each has a shuffle control. Graded by option-letter probability, no parsing.
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
N_SEEDS = int(os.environ.get("CR_SEEDS", "12"))
QPS = int(os.environ.get("CR_QPS", "6"))
PATH = os.environ.get("CR_PATH", "results/validate.json")
OBJECTS = [(a, b) for a in range(K) for b in range(K)]


def code(o, pa, pb):
    return f"{POOL_A[pa[o[0]]]} {POOL_B[pb[o[1]]]}"


def make_split(rng):
    while True:
        objs = OBJECTS[:]
        rng.shuffle(objs)
        train = objs[:10]
        if {o[0] for o in train} == set(range(K)) and {o[1] for o in train} == set(range(K)):
            return train, objs[10:]


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
        toks = [tok.encode(s, add_special_tokens=False)[0] for s in (ch, " " + ch) if tok.encode(s, add_special_tokens=False)]
        ids.append(toks)
    return ids


def choose(tok, model, lids, text, n):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    inp = tok(s, return_tensors="pt").to(DEV)
    with torch.no_grad():
        logits = model(**inp).logits[0, -1]
    logp = torch.log_softmax(logits.float(), -1)
    return int(np.argmax([max(logp[i].item() for i in lids[k]) for k in range(n)]))


def held(o, kind, pool, rng):
    if kind == "p1":
        c = [x for x in pool if x[1] == o[1] and x[0] != o[0]]
    elif kind == "p2":
        c = [x for x in pool if x[0] == o[0] and x[1] != o[1]]
    else:
        c = [x for x in pool if x[0] != o[0] and x[1] != o[1]]
    rng.shuffle(c)
    return c


def run(tok, model, lids):
    swap_all3, swap_single, swap_shuf = [], [], []
    hard_real, hard_shuf = [], []
    choice = {"correct": 0, "neighbor": 0, "filler": 0}  # failure-mode tally on swap (real)
    for seed in range(N_SEEDS):
        rng = random.Random(seed * 100 + 13)
        pa, pb = list(range(K)), list(range(K))
        rng.shuffle(pa)
        rng.shuffle(pb)
        train, pool = make_split(rng)
        ex_real = [(o, code(o, pa, pb)) for o in train]
        ex_shuf = [(o, f"{rng.choice(POOL_A)} {rng.choice(POOL_B)}") for o in train]
        queries = [o for o in pool]
        rng.shuffle(queries)
        for o in queries[:QPS]:
            tw1, tw2, fl = held(o, "p1", pool, rng), held(o, "p2", pool, rng), held(o, "both", pool, rng)
            # swap test needs both twins and a filler, all held-out
            if tw1 and tw2 and fl:
                o1, o2, f = tw1[0], tw2[0], fl[0]
                cset = [o, o1, o2, f]
                ok = 0
                for ex, store in ((ex_real, "real"), (ex_shuf, "shuf")):
                    n_ok = 0
                    for target in (o, o1, o2):
                        cands = cset[:]
                        rng.shuffle(cands)
                        pick = cands[choose(tok, model, lids, build_prompt(ex, code(target, pa, pb), cands), 4)]
                        hit = pick == target
                        n_ok += hit
                        if store == "real":
                            choice["correct" if hit else ("neighbor" if pick in (o, o1, o2) else "filler")] += 1
                    if ex is ex_real:
                        ok = n_ok
                        swap_single.append(n_ok / 3)
                    else:
                        swap_shuf.append(n_ok / 3)
                swap_all3.append(1.0 if ok == 3 else 0.0)
            # hard test: target plus three half-right distractors
            halves = held(o, "p1", pool, rng) + held(o, "p2", pool, rng)
            rng.shuffle(halves)
            if len(halves) >= 3:
                for ex, bucket in ((ex_real, hard_real), (ex_shuf, hard_shuf)):
                    cands = [o] + halves[:3]
                    rng.shuffle(cands)
                    pick = cands[choose(tok, model, lids, build_prompt(ex, code(o, pa, pb), cands), 4)]
                    bucket.append(1.0 if pick == o else 0.0)
    tot = max(1, sum(choice.values()))
    return {
        "swap_all3": float(np.mean(swap_all3)), "swap_single": float(np.mean(swap_single)),
        "swap_shuf": float(np.mean(swap_shuf)), "hard_real": float(np.mean(hard_real)),
        "hard_shuf": float(np.mean(hard_shuf)), "chance": 1 / len(LETTERS),
        "choice": {k: v / tot for k, v in choice.items()},
    }


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
            r = run(tok, model, letter_ids(tok))
            out[short] = r
            json.dump(out, open(PATH, "w"), indent=2)
            print(f"[{short}] swap_all3 {r['swap_all3']:.2f}  swap_single {r['swap_single']:.2f}/{r['swap_shuf']:.2f}  hard {r['hard_real']:.2f}/{r['hard_shuf']:.2f}", flush=True)
        except Exception as e:
            print(f"[{short}] FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            gc.collect()
            if DEV == "mps":
                torch.mps.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
