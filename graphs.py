import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIZE = {
    "SmolLM2-360M": 0.36, "Qwen2.5-0.5B": 0.49, "OLMo-2-1B": 1.0,
    "Llama-3.2-1B": 1.24, "Falcon3-1B": 1.5, "Qwen2.5-1.5B": 1.54,
    "StableLM-2-1.6B": 1.64, "SmolLM2-1.7B": 1.71, "Granite-3.1-2B": 2.53,
    "Gemma-2-2B": 2.61, "Llama-3.2-3B": 3.21, "Phi-3.5-mini": 3.82,
}
GREEN, GREY, PURPLE, RUST, GOLD = "#0F6E56", "#B4B2A9", "#534AB7", "#A6452B", "#C98A00"
data = json.load(open("results/clean.json"))
models = [m for m in sorted(SIZE, key=SIZE.get) if m in data]
chance = data[models[0]]["retrieval"]["chance"]


def g(m, cond, key):
    return data[m].get(cond, {}).get(key, np.nan)


# 1. The capability plane. Each model is a point: how much it retrieves (x) versus
# how much it composes (y), both as signal above the shuffle control. The whole
# claim is in the geometry: a low-left cluster that does neither, a lower-right
# group that retrieves but cannot compose, and an upper-right group that does both.
rx = [g(m, "retrieval", "real") - g(m, "retrieval", "shuffled") for m in models]
cy = [g(m, "comp_none", "real") - g(m, "comp_none", "shuffled") for m in models]
sz = [SIZE[m] for m in models]
plt.figure(figsize=(9, 7.5))
plt.axhline(0, color="k", lw=0.8)
plt.axvline(0, color="k", lw=0.8)
plt.axhspan(-0.1, 0.1, color="#00000008")
sc = plt.scatter(rx, cy, c=sz, cmap="viridis", s=160, edgecolors="k", zorder=3)
for m, x, y in zip(models, rx, cy):
    plt.annotate(m, (x, y), fontsize=8, xytext=(7, 4), textcoords="offset points")
plt.ylim(-0.1, 0.6)
plt.xlim(-0.08, 0.75)
plt.text(0.30, 0.92, "retrieves and composes", color=GREEN, fontsize=12, transform=plt.gca().transAxes)
plt.text(0.62, 0.20, "retrieves only", color=RUST, fontsize=12, transform=plt.gca().transAxes)
plt.text(0.03, 0.20, "neither", color=GREY, fontsize=12, transform=plt.gca().transAxes)
plt.colorbar(sc, label="approx size (B params)")
plt.xlabel("retrieval signal (real minus shuffle)")
plt.ylabel("composition signal (real minus shuffle)")
plt.title("Capability plane: retrieving a shown code vs composing a held-out one")
plt.tight_layout()
plt.savefig("figures/plane.png", dpi=130)

# 2. The two thresholds against scale, retrieval onset before composition onset.
ret = [g(m, "retrieval", "real") for m in models]
comp = [g(m, "comp_none", "real") for m in models]
plt.figure(figsize=(9, 5.5))
plt.axhspan(0, chance, color="#00000008")
plt.axhline(chance, color="k", ls="--", lw=1)
plt.text(sz[-1], chance + 0.01, "chance", ha="right", fontsize=9)
plt.plot(sz, ret, "o-", color=GREY, label="retrieval")
plt.plot(sz, comp, "o-", color=GREEN, label="composition (no overlap)")
for m, x, y in zip(models, sz, comp):
    plt.annotate(m, (x, y), fontsize=7, xytext=(3, -10), textcoords="offset points")
plt.xlabel("approximate size (billions of parameters)")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.title("Retrieval comes online before composition, and composition is not guaranteed by size")
plt.legend()
plt.tight_layout()
plt.savefig("figures/thresholds.png", dpi=130)

# 3 and 4 need the validation file.
if os.path.exists("results/validate.json"):
    v = json.load(open("results/validate.json"))
    vm = [m for m in sorted(SIZE, key=SIZE.get) if m in v]
    vch = v[vm[0]]["chance"]

    # 3. The causal swap check at two levels of strictness, all twelve models.
    # single = decode one of the three minimal-pair codes correctly (graded, shows
    # the ladder). all3 = decode all three in a row (strict, only real composition
    # survives). The two together show every model and where each sits.
    single = [v[m]["swap_single"] for m in vm]
    a3 = [v[m]["swap_all3"] for m in vm]
    x = np.arange(len(vm))
    w = 0.38
    plt.figure(figsize=(11, 5.5))
    plt.bar(x - w / 2, single, w, color=GREEN, label="swap, one code at a time")
    plt.bar(x + w / 2, a3, w, color=PURPLE, label="swap, all three in a row")
    plt.axhline(vch, color="k", ls="--", lw=1)
    plt.text(len(vm) - 0.5, vch + 0.012, "chance, one at a time", ha="right", fontsize=9)
    plt.axhline((1 / 4) ** 3, color="#999", ls=":", lw=1)
    plt.text(0, (1 / 4) ** 3 + 0.012, "chance, all three", fontsize=8)
    plt.xticks(x, vm, rotation=45, ha="right", fontsize=8)
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.title("Causal swap test: one code at a time is a ladder, all three at once only Phi clears")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/validation.png", dpi=130)

    # 4. Failure mode: when wrong on the swap test, a half-right neighbor (overlap
    # error) or an unrelated filler (noise)? Stacked per model.
    cor = [v[m]["choice"]["correct"] for m in vm]
    nei = [v[m]["choice"]["neighbor"] for m in vm]
    fil = [v[m]["choice"]["filler"] for m in vm]
    plt.figure(figsize=(11, 5.5))
    plt.bar(vm, cor, color=GREEN, label="correct")
    plt.bar(vm, nei, bottom=cor, color=RUST, label="half-right neighbor (overlap error)")
    plt.bar(vm, fil, bottom=np.array(cor) + np.array(nei), color=GREY, label="unrelated filler (noise)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("fraction of swap-test choices")
    plt.title("How models choose on the swap test: composition, overlap error, or noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/failuremode.png", dpi=130)
    print("saved figures/plane.png thresholds.png validation.png failuremode.png")
else:
    print("saved figures/plane.png thresholds.png  (run validate.py for validation + failure-mode figures)")
