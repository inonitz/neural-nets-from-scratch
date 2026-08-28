# Generates the README figures from a trained network. Added at archive-publication time.
# Run after training (NN.pickle present) with MNIST in samples/ (see get_mnist.py).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mnist.loader import MNIST
from Classifier_minibatchGD import classifier, normalize

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

net = classifier()
net.import_nn("NN.pickle")

mndata = MNIST("samples", gz=True)
test_imgs, test_lbls = mndata.load_testing()
X = normalize(test_imgs)
y = np.array(test_lbls)

# --- Figure 1: training curves (loss per epoch + running test accuracy) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(range(1, len(net.loss) + 1), net.loss, color="#c0392b")
ax1.set_title("Training loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy")
ax1.grid(alpha=0.3)

# recompute a clean running-accuracy curve over the 10k test set
correct = 0
running = []
for i in range(len(y)):
    net.propagate(X[i])
    if int(np.argmax(net.activations[-1])) == y[i]:
        correct += 1
    running.append(100 * correct / (i + 1))
    net.flush()
final_acc = running[-1]
ax2.plot(range(1, len(running) + 1), running, color="#2471a3")
ax2.axhline(final_acc, ls="--", color="#7f8c8d", lw=1)
ax2.set_title(f"Running test accuracy -> {final_acc:.2f}%")
ax2.set_xlabel("test samples seen"); ax2.set_ylabel("accuracy (%)")
ax2.set_ylim(80, 100); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "training_curves.png"), dpi=110)
print("wrote training_curves.png  (final accuracy %.2f%%)" % final_acc)

# --- Figure 2: sample predictions (green = correct, red = wrong) ---
rng = np.random.default_rng(7)
picks = rng.choice(len(y), 40, replace=False)
fig, axes = plt.subplots(5, 8, figsize=(11, 7))
for ax, idx in zip(axes.ravel(), picks):
    net.propagate(X[idx])
    pred = int(np.argmax(net.activations[-1]))
    net.flush()
    ok = pred == y[idx]
    ax.imshow(np.array(test_imgs[idx]).reshape(28, 28), cmap="gray")
    ax.set_title(f"{pred}", color="#1e8449" if ok else "#c0392b", fontsize=13)
    ax.axis("off")
fig.suptitle("Predictions on unseen MNIST test digits", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "predictions.png"), dpi=110)
print("wrote predictions.png")
