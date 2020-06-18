import os
import struct
import urllib.request
import gzip

import numpy as np
import pandas

data = dict(
    train_images="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    train_labels="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    test_images="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    test_labels="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
)


def compress_name(name):
    return f"{name}.gz"


def download(name):
    if os.path.exists(compress_name(name)):
        return
    response = urllib.request.urlopen(data[name])
    with open(compress_name(name), "wb") as f:
        f.write(response.read())


def parse_image(name):
    with gzip.open(compress_name(name), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        num_row, num_col = struct.unpack(">II", f.read(8))
        d = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).reshape((size, num_row * num_col))
        return d


def parse_label(name):
    with gzip.open(compress_name(name), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        d = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(size, 1)
        return d


def prepare_upload_data(name, image_name, label_name):
    image, label = parse_image(image_name), parse_label(label_name)
    array = np.hstack([image, label])
    pd = pandas.DataFrame(array)
    pd.to_csv(name, index_label="id")


def main():
    for name in data:
        download(name)
    prepare_upload_data("mnist.train.csv", "train_images", "train_labels")
    prepare_upload_data("mnist.test.csv", "test_images", "test_labels")


if __name__ == "__main__":
    main()
