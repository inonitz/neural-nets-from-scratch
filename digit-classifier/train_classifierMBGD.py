import os
import argparse
import matplotlib
import numpy as np
from mnist.loader import MNIST
from Classifier_minibatchGD import classifier, normalize
from get_mnist import ensure_mnist

PICKLE = "NN.pickle"
DEFAULT_ARCH = [784, 100, 100, 10]     # input(28x28=784) -> two hidden -> output(10 digits)


def build_args():
    p = argparse.ArgumentParser(
        description="Train or evaluate a from-scratch MNIST digit classifier (raw numpy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The dataset: 60,000 training images (used here as 50,000 train + 10,000 validation)\n"
            "and 10,000 test images, each a 28x28 = 784-pixel grayscale digit.\n\n"
            "How training works: one PASS over the 50,000 training images splits them into\n"
            "mini-batches of --batch-size (e.g. 400 -> 125 batches) and updates the weights once per\n"
            "batch. --iterations is how many full passes to run.\n\n"
            "Default (no --train/--resume): just load {p} and evaluate on the 10k test set.\n\n"
            "Examples:\n"
            "  python train_classifierMBGD.py                        # evaluate the saved network\n"
            "  python train_classifierMBGD.py --train                # train a new one from scratch\n"
            "  python train_classifierMBGD.py --train --iterations 20 --lr 0.1\n"
            "  python train_classifierMBGD.py --resume --iterations 5    # train the saved one more"
        ).format(p=PICKLE),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true",
                      help="train a NEW network from scratch, then evaluate (overwrites %s)" % PICKLE)
    mode.add_argument("--resume", action="store_true",
                      help="continue training the existing %s, then evaluate" % PICKLE)
    p.add_argument("--iterations", type=int, default=10, metavar="N",
                   help="full passes over the 50,000 training images (default: 10)")
    p.add_argument("--lr", type=float, default=None, metavar="RATE",
                   help="learning rate. Default 0.1 for --train; --resume keeps the saved rate unless "
                        "you pass this. Higher = faster but oscillates (the original used 1.0, which "
                        "reaches ~97%% but bounces a lot); 0.1 is slower and much smoother.")
    p.add_argument("--batch-size", type=int, default=400, metavar="N", dest="batch_size",
                   help="images per mini-batch (default: 400 -> 125 batches per pass)")
    p.add_argument("--arch", type=int, nargs="+", default=DEFAULT_ARCH, metavar="SIZE",
                   help="layer sizes including input 784 and output 10 (default: 784 100 100 10)")
    return p.parse_args()


def load_data():
    ensure_mnist("samples")            # downloads MNIST (~11MB) into samples/ on first run
    mndata = MNIST("samples", gz=True)
    train_i, train_l = mndata.load_training()
    test_i, test_l = mndata.load_testing()
    X = normalize(train_i[0:50000])
    y = np.array(train_l[0:50000]).reshape(50000, 1)
    Xte = normalize(test_i)
    yte = np.array(test_l).reshape(10000, 1)
    return X, y, Xte, yte


def train(net, X, y, batch_size, iterations):
    batches = len(X) // batch_size     # drop the remainder (< batch_size)
    if batches < 1:
        raise SystemExit("--batch-size %d is larger than the 50,000-image training set." % batch_size)
    print("Training set: {} images. Each pass = {} batches of {} (lr={}). Running {} pass(es)..."
          .format(len(X), batches, batch_size, net.lr, iterations))
    net.fit(X, y, batchsize=batch_size, epochs=batches, iterations=iterations)
    net.export(os.path.splitext(PICKLE)[0])
    print("Saved weights to {}.".format(PICKLE))


def main():
    args = build_args()
    X, y, Xte, yte = load_data()

    if args.train:
        net = classifier(args.arch, lr=(args.lr if args.lr is not None else 0.1))
        train(net, X, y, args.batch_size, args.iterations)
    elif args.resume:
        if not os.path.exists(PICKLE):
            raise SystemExit("No %s to resume from -- run with --train first." % PICKLE)
        net = classifier()
        net.import_nn(PICKLE)
        if args.lr is not None:
            net.lr = args.lr
        train(net, X, y, args.batch_size, args.iterations)
    else:
        if not os.path.exists(PICKLE):
            raise SystemExit(
                "No %s found. Train one first:\n  python train_classifierMBGD.py --train" % PICKLE)
        net = classifier()
        net.import_nn(PICKLE)
        print("Loaded %s -- evaluating only (pass --train or --resume to train)." % PICKLE)

    # evaluate on the 10,000-image test set
    net.test(Xte, yte, epochs=10000, testing_method="testing", show_activations=False)

    # plot windows only make sense on an interactive backend; a headless run skips them
    # (avoiding matplotlib's "FigureCanvasAgg is non-interactive" warning).
    interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "template")
    if interactive:
        if net.loss:
            net.plot_loss()
        net.plot_accuracy()
    else:
        print("(non-interactive matplotlib backend -- skipping plot windows; run make_figures.py to save them)")


if __name__ == "__main__":
    main()

# Reference: accuracy reached in past runs (single classifier, [784, 100, 100, 10]):
#   --batch-size 625 --iterations ~80 --lr 5    --> ~93%   (arch [784, 20, 20, 10])
#   --batch-size 400 --iterations ~125 --lr 0.1 --> ~93-95%
#   --batch-size 400 --iterations many --lr 1   --> ~96.5-97.4% (but the loss oscillates)
