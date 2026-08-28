# Downloads the MNIST dataset (4 gzipped idx files, ~11MB) into samples/.
# The dataset itself is not committed to this archive repo.
# Added at archive-publication time (2026); not part of the original project.
import os
import urllib.request

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

os.makedirs("samples", exist_ok=True)
for name in FILES:
    dest = os.path.join("samples", name)
    if os.path.exists(dest):
        print(f"{dest} already present, skipping")
        continue
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(MIRROR + name, dest)
print("done.")
