import os
os.environ["HF_HUB_OFFLINE"] = "1"

import re, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Stage 1, the feasibility check. Objects are abstract pairs (a, b) with a and b
# in 0..3, so there is no English word for them. A compositional code maps each
# value to a symbol, one symbol per position. We show a model some examples and
# test whether it can decode the code for held-out objects. The control: scramble
# the examples so there is no rule to learn. If accuracy stays up under scrambling,
# the model was not using the code and the whole idea is theater.
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Gemma-2-2B": "google/gemma-2-2b-it",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
}
SYM = ["ka", "mi", "to", "ru"]
VALUES = 4
OBJECTS = [(a, b) for a in range(VALUES) for b in range(VALUES)]
N_SEEDS, N_TEST, N_CAND = 6, 8, 4


def encode(o, px, py):
    return f"{SYM[px[o[0]]]} {SYM[py[o[1]]]}"


def prompt(examples, msg, cands):
    L = ["Decode a symbol code. Each object is a pair (a, b) with a and b each from 0 to 3.",
         "Examples of objects and their codes:"]
    for o, m in examples:
        L.append(f'  object {o}: code "{m}"')
    L.append(f'\nThe code "{msg}" refers to one of these objects:')
    for i, o in enumerate(cands):
        L.append(f"  {i}: {o}")
    L.append("\nReply with only the number of the correct object.")
    return "\n".join(L)


def ask(tok, model, text):
    inp = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True),
              return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
    m = re.search(r"\d", tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True))
    return int(m.group()) if m else -1


print(f"{'model':16s}{'real':>7s}{'shuffled':>10s}{'chance':>8s}")
for short, repo in MODELS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    real, shuf = [], []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        px, py = list(range(VALUES)), list(range(VALUES))
        rng.shuffle(px); rng.shuffle(py)
        objs = OBJECTS[:]; rng.shuffle(objs)
        train, test = objs[:8], objs[8:8 + N_TEST]
        for scramble in (False, True):
            if scramble:
                ex = [(o, f"{rng.choice(SYM)} {rng.choice(SYM)}") for o in train]
            else:
                ex = [(o, encode(o, px, py)) for o in train]
            correct = 0
            for o in test:
                msg = encode(o, px, py)
                cands = [o] + rng.sample([t for t in test if t != o], N_CAND - 1)
                rng.shuffle(cands)
                a = ask(tok, model, prompt(ex, msg, cands))
                correct += (0 <= a < len(cands) and cands[a] == o)
            (shuf if scramble else real).append(correct / len(test))
    print(f"{short:16s}{np.mean(real):>7.2f}{np.mean(shuf):>10.2f}{1/N_CAND:>8.2f}", flush=True)
    del model
    import gc; gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()
