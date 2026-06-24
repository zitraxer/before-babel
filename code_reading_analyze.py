import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Approximate parameter counts in billions, only used to order the models and to
# ask whether composition appears with scale. Ranks matter more than exact values.
SIZE = {
    "SmolLM2-360M": 0.36, "Qwen2.5-0.5B": 0.49, "OLMo-2-1B": 1.0,
    "Llama-3.2-1B": 1.24, "Falcon3-1B": 1.5, "Qwen2.5-1.5B": 1.54,
    "StableLM-2-1.6B": 1.64, "SmolLM2-1.7B": 1.71, "Granite-3.1-2B": 2.53,
    "Gemma-2-2B": 2.61, "Llama-3.2-3B": 3.21, "Phi-3.5-mini": 3.82,
}
TIERS = ["lookup 1x4", "comp 2x2", "comp 2x3"]
GREEN, GREY, PURPLE, RUST = "#0F6E56", "#B4B2A9", "#534AB7", "#A6452B"

data = json.load(open("results/code_reading.json"))
models = [m for m in sorted(SIZE, key=SIZE.get) if m in data]
chance = data[models[0]]["plain"]["lookup 1x4"]["chance"]


def cell(m, cond, tier, key):
    return data[m].get(cond, {}).get(tier, {}).get(key, np.nan)


# 1. The trap. comp 2x2 real accuracy per model, chance line. Below the line is
# the signature of matching on shared symbols instead of composing.
vals = [cell(m, "plain", "comp 2x2", "real") for m in models]
plt.figure(figsize=(9, 5))
colors = [RUST if v < chance else GREEN for v in vals]
plt.barh(models, vals, color=colors)
plt.axvline(chance, color="k", ls="--", lw=1)
plt.text(chance + 0.005, -0.6, "chance", fontsize=9)
plt.xlabel("accuracy on held-out object (comp 2x2)")
plt.title("The composition trap: below chance means the model matches shared symbols, not the rule")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("figures/trap.png", dpi=130)

# 2. Signal above control. real minus shuffled per model per tier. Positive means
# genuine use of the code, near zero means the examples did not help, negative is
# the trap pulling the model below where guessing would leave it.
M = np.array([[cell(m, "plain", t, "real") - cell(m, "plain", t, "shuffled") for t in TIERS] for m in models])
plt.figure(figsize=(7, 7))
lim = np.nanmax(np.abs(M))
im = plt.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
plt.xticks(range(len(TIERS)), TIERS)
plt.yticks(range(len(models)), models)
for i in range(len(models)):
    for j in range(len(TIERS)):
        if not np.isnan(M[i, j]):
            plt.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(M[i, j]) > lim * 0.6 else "black")
plt.colorbar(im, label="real minus shuffled (genuine signal from the code)")
plt.title("Lookup works everywhere, composition mostly does not\n(blue = below its own control, the overlap trap)")
plt.tight_layout()
plt.savefig("figures/signal_heatmap.png", dpi=130)

# 3. Does reasoning rescue composition. plain versus reason real on comp 2x3.
pr = [cell(m, "plain", "comp 2x3", "real") for m in models]
re = [cell(m, "reason", "comp 2x3", "real") for m in models]
x = np.arange(len(models))
plt.figure(figsize=(10, 5))
for i in range(len(models)):
    plt.plot([x[i], x[i]], [pr[i], re[i]], color=GREY, lw=1, zorder=1)
plt.scatter(x, pr, color=GREY, label="answer directly", zorder=2)
plt.scatter(x, re, color=PURPLE, label="reason first", zorder=2)
plt.axhline(chance, color="k", ls="--", lw=1)
plt.xticks(x, models, rotation=45, ha="right", fontsize=8)
plt.ylabel("accuracy on held-out objects (comp 2x3)")
plt.title("Does thinking before answering help composition?")
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/reasoning.png", dpi=130)

# 4. Composition versus scale. comp 2x3 signal against model size.
sz = [SIZE[m] for m in models]
sig = [cell(m, "plain", "comp 2x3", "real") - cell(m, "plain", "comp 2x3", "shuffled") for m in models]
plt.figure(figsize=(8, 5.5))
plt.axhline(0, color="k", lw=0.8)
plt.scatter(sz, sig, color=GREEN, s=60, zorder=3)
for m, a, b in zip(models, sz, sig):
    plt.annotate(m, (a, b), fontsize=7, xytext=(4, 4), textcoords="offset points")
plt.xlabel("approximate size (billions of parameters)")
plt.ylabel("composition signal on comp 2x3 (real minus shuffled)")
plt.title("Does compositional code reading switch on with scale?")
plt.tight_layout()
plt.savefig("figures/scale.png", dpi=130)

print(f"models analyzed: {len(models)}   chance: {chance:.2f}\n")
print(f"{'model':16s}{'size':>6s} | " + "  ".join(f"{t:>10s}" for t in TIERS))
for m in models:
    row = "  ".join(f"{cell(m,'plain',t,'real'):.2f}/{cell(m,'plain',t,'shuffled'):.2f}" for t in TIERS)
    print(f"{m:16s}{SIZE[m]:>6.2f} | {row}")
print("\n(real/shuffled per tier; comp 2x2 below chance = overlap trap)")
print("\nreasoning effect on comp 2x3 (plain real -> reason real):")
for m in models:
    print(f"  {m:16s} {cell(m,'plain','comp 2x3','real'):.2f} -> {cell(m,'reason','comp 2x3','real'):.2f}")
print("\nsaved figures/trap.png signal_heatmap.png reasoning.png scale.png")
