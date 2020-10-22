import os
import struct
import urllib.request
import gzip

import numpy as np

data = dict(
    train_images="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    train_labels="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    test_images="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    test_labels="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
)


def compress_file_path(name, cache_dir):
    return os.path.join(cache_dir, f"{name}.gz")


def download(name, cache_dir):
    target_file = compress_file_path(name, cache_dir)
    if os.path.exists(target_file):
        return
    response = urllib.request.urlopen(data[name])
    with open(target_file, "wb") as f:
        f.write(response.read())


def parse_image(name, cache_dir):
    with gzip.open(compress_file_path(name, cache_dir), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        num_row, num_col = struct.unpack(">II", f.read(8))
        d = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).reshape((size, num_row * num_col))
        return d


def parse_label(name, cache_dir):
    with gzip.open(compress_file_path(name, cache_dir), "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        d = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(size, 1)
        return d


def prepare_upload_data(name, image_name, label_name, cache_dir):
    image, label = parse_image(image_name, cache_dir), parse_label(label_name, cache_dir)
    id_col = np.array(range(label.size)).reshape(label.shape)
    array = np.hstack([id_col, image, label])
    header = f"id,{','.join(map(str, range(array.shape[1] - 2)))},label"
    np.savetxt(name, array, delimiter=',', fmt="%d", header=header, comments='')


def main():
    data_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), os.path.pardir, os.path.pardir, "data"))
    cache_dir = os.path.join(data_dir, ".cache")
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for name in data:
        download(name, cache_dir)
    prepare_upload_data(f"{os.path.join(data_dir, 'mnist_train.csv')}", "train_images", "train_labels", cache_dir)
    prepare_upload_data(f"{os.path.join(data_dir, 'mnist_test.csv')}", "test_images", "test_labels", cache_dir)


if __name__ == "__main__":
    main()
