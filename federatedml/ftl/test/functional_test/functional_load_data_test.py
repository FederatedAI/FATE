import numpy as np

from arch.api.session import init
from federatedml.ftl.data_util.common_data_util import load_data, feed_into_dtable
from federatedml.ftl.test.util import assert_array, assert_matrix

if __name__ == '__main__':

    expected_ids = np.array([133, 273, 175, 551, 199, 274])
    expected_y = np.array([1, 1, 1, 1, -1, -1])

    expected_X = np.array([[0.254879, -1.046633, 0.209656, 0.074214, -0.441366, -0.377645],
                           [-1.142928, - 0.781198, - 1.166747, - 0.923578, 0.62823, - 1.021418],
                           [-1.451067, - 1.406518, - 1.456564, - 1.092337, - 0.708765, - 1.168557],
                           [-0.879933, 0.420589, - 0.877527, - 0.780484, - 1.037534, - 0.48388],
                           [0.426758, 0.723479, 0.316885, 0.287273, 1.000835, 0.962702],
                           [0.963102, 1.467675, 0.829202, 0.772457, - 0.038076, - 0.468613]])

    infile = "../../../../examples/data/unittest_data.csv"
    ids, X, y = load_data(infile, 0, (2, 8), 1)

    ids = np.array(ids, dtype=np.int32)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)

    print("ids shape", ids.shape)
    print("X shape", X.shape)
    print("y shape", y.shape)

    assert_array(expected_ids, ids)
    assert_array(expected_y, y)
    assert_matrix(expected_X, X)

    expected_data = {}
    for i, id in enumerate(expected_ids):
        expected_data[id] = {
            "X": expected_X[i],
            "y": expected_y[i]
        }

    init()
    data_table = feed_into_dtable(ids, X, y.reshape(-1, 1), (0, len(ids)), (0, X.shape[-1]))
    for item in data_table.collect():
        id = item[0]
        inst = item[1]
        expected_item = expected_data[id]
        X_i = expected_item["X"]
        y_i = expected_item["y"]

        features = inst.features
        label = inst.label
        assert_array(X_i, features)
        assert y_i == label
