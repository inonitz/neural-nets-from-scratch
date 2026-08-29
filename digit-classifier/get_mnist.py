# Downloads the MNIST dataset (4 gzipped idx files, ~11MB) into samples/.
# The dataset itself is not committed to this archive repo.
# Added at archive-publication time (2026); not part of the original project.
#
# Run it directly (`python get_mnist.py`) or let the trainer call ensure_mnist() for you.
import os
import urllib.request

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def ensure_mnist(dest="samples"):
    """Download any missing MNIST files into `dest`; a no-op if they're already there."""
    os.makedirs(dest, exist_ok=True)
    for name in FILES:
        path = os.path.join(dest, name)
        if os.path.exists(path):
            print(f"{path} already present, skipping")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(MIRROR + name, path)
    return dest


if __name__ == "__main__":
    ensure_mnist()
    print("done.")
