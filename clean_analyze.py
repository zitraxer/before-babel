import json
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
GREEN, GREY, PURPLE, RUST = "#0F6E56", "#B4B2A9", "#534AB7", "#A6452B"
data = json.load(open("results/clean.json"))
models = [m for m in sorted(SIZE, key=SIZE.get) if m in data]
chance = data[models[0]]["retrieval"]["chance"]


def g(m, cond, key):
    return data[m].get(cond, {}).get(key, np.nan)


# 1. Retrieval, composition without a surface trap, composition with one. Real
# accuracy per model, chance line. Reads left to right as: can it retrieve, can
# it compose, is composition broken by a shared-symbol distractor.
x = np.arange(len(models))
w = 0.26
plt.figure(figsize=(11, 5.5))
plt.bar(x - w, [g(m, "retrieval", "real") for m in models], w, label="retrieval (seen code)", color=GREY)
plt.bar(x, [g(m, "comp_none", "real") for m in models], w, label="composition, no overlap", color=GREEN)
plt.bar(x + w, [g(m, "comp_overlap", "real") for m in models], w, label="composition, overlapping distractor", color=RUST)
plt.axhline(chance, color="k", ls="--", lw=1)
plt.text(len(models) - 0.5, chance + 0.01, "chance", fontsize=9, ha="right")
plt.xticks(x, models, rotation=45, ha="right", fontsize=8)
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.title("Retrieval vs composition, and whether a shared symbol breaks composition")
plt.legend()
plt.tight_layout()
plt.savefig("figures/clean_main.png", dpi=130)

# 2. The isolated overlap effect: composition drop caused purely by adding one
# distractor that shares a symbol. Positive means the model was pulled off by
# surface overlap, the decode-by-overlap signature, with the composition demand
# held fixed.
drop = [g(m, "comp_none", "real") - g(m, "comp_overlap", "real") for m in models]
plt.figure(figsize=(9, 5))
plt.barh(models, drop, color=[RUST if d > 0.03 else GREY for d in drop])
plt.axvline(0, color="k", lw=0.8)
plt.xlabel("accuracy lost when one distractor shares a symbol (comp_none minus comp_overlap)")
plt.title("Isolated overlap effect: surface match breaking composition")
plt.tight_layout()
plt.savefig("figures/clean_overlap.png", dpi=130)

# 3. Signal above control. real minus shuffled per condition, the genuine use of
# the code with guessing and biases subtracted out.
CONDS = ["retrieval", "comp_none", "comp_overlap"]
M = np.array([[g(m, c, "real") - g(m, c, "shuffled") for c in CONDS] for m in models])
plt.figure(figsize=(7, 7))
lim = max(0.01, np.nanmax(np.abs(M)))
im = plt.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
plt.xticks(range(len(CONDS)), ["retrieval", "comp\nno overlap", "comp\noverlap"])
plt.yticks(range(len(models)), models)
for i in range(len(models)):
    for j in range(len(CONDS)):
        if not np.isnan(M[i, j]):
            plt.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(M[i, j]) > lim * 0.6 else "black")
plt.colorbar(im, label="real minus shuffled")
plt.title("Genuine signal from the code, by condition")
plt.tight_layout()
plt.savefig("figures/clean_signal.png", dpi=130)

# 4. Scale. Retrieval and composition signal against model size.
sz = [SIZE[m] for m in models]
ret = [g(m, "retrieval", "real") - g(m, "retrieval", "shuffled") for m in models]
comp = [g(m, "comp_none", "real") - g(m, "comp_none", "shuffled") for m in models]
plt.figure(figsize=(8, 5.5))
plt.axhline(0, color="k", lw=0.8)
plt.scatter(sz, ret, color=GREY, s=55, label="retrieval signal", zorder=3)
plt.scatter(sz, comp, color=GREEN, s=55, label="composition signal", zorder=3)
for m, a, b in zip(models, sz, comp):
    plt.annotate(m, (a, b), fontsize=7, xytext=(4, 3), textcoords="offset points")
plt.xlabel("approximate size (billions of parameters)")
plt.ylabel("signal above control (real minus shuffled)")
plt.title("Retrieval vs composition against scale")
plt.legend()
plt.tight_layout()
plt.savefig("figures/clean_scale.png", dpi=130)

print(f"models: {len(models)}   chance: {chance:.2f}\n")
print(f"{'model':16s}{'size':>6s} | {'retrieval':>16s}{'comp_none':>16s}{'comp_overlap':>16s}")
for m in models:
    cells = "".join(f"{g(m,c,'real'):>7.2f}/{g(m,c,'shuffled'):.2f}  " for c in CONDS)
    print(f"{m:16s}{SIZE[m]:>6.2f} | {cells}")
print("\n(real/shuffled per condition, chance 0.25)")
print("\noverlap effect (comp_none real minus comp_overlap real, positive = broken by surface match):")
for m in models:
    print(f"  {m:16s} {g(m,'comp_none','real') - g(m,'comp_overlap','real'):+.2f}")
print("\nsaved figures/clean_main.png clean_overlap.png clean_signal.png clean_scale.png")
