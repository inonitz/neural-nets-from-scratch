# Generates the XOR decision-boundary figure for the README. Added at archive-publication time.
import os
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xor_bias_added import XOR

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
Y = np.array([[1], [1], [0], [0]])

np.random.seed(5)   # a seed that converges to a clean XOR solution (this net is init-sensitive)
net = XOR(arch=[2, 2, 1], lr=1, plot_loss=True)
net.add_bias(1)
with contextlib.redirect_stdout(io.StringIO()):     # fit() prints every iteration
    net.fit(X, Y, epochs=80000)

def predict(a, b):
    net.propagate(np.array([a, b]))
    out = float(net.activations[-1][0])
    net.flush()
    return out

# --- decision surface over the unit square ---
g = 120
grid = np.array([[predict(i / (g - 1), j / (g - 1)) for i in range(g)] for j in range(g)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
im = ax1.imshow(grid, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=0, vmax=1)
for (a, b), t in zip(X, Y.ravel()):
    ax1.scatter(a, b, s=260, c=("#b03a2e" if t else "#1f4e79"),
                edgecolors="white", linewidths=2, zorder=3)
    ax1.annotate(f"{t}", (a, b), color="white", ha="center", va="center",
                 fontsize=11, fontweight="bold", zorder=4)
ax1.set_title("Learned XOR decision surface")
ax1.set_xlabel("input A"); ax1.set_ylabel("input B")
fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="network output")

loss = [float(np.ravel(l)[0]) for l in net.loss]
ax2.plot(loss, color="#c0392b", lw=1)
ax2.set_title("Training loss (squared error)")
ax2.set_xlabel("logged step"); ax2.set_ylabel("loss"); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "xor_decision_boundary.png"), dpi=110)

print("outputs:", {tuple(x): round(predict(*x), 3) for x in X})
print("wrote xor_decision_boundary.png")
