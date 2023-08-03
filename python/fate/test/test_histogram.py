import pickle
import random

import pandas as pd
import torch
from fate.arch import Context
from fate.arch.computing.standalone import CSession
from fate.arch.histogram.histogram import DistributedHistogram, Histogram

ctx = Context()
kit = ctx.cipher.phe.setup(options={"kind": "paillier", "key_length": 1024})
sk, pk, coder, evaluator = kit.sk, kit.pk, kit.coder, kit.evaluator


def test_plain():
    # plain
    hist = Histogram.create(1, [3, 2], {"c0": {"type": "tensor", "stride": 2}})
    print(f"created:\n {hist}")
    hist.i_update(
        [0, 0, 0, 0],
        [[1, 0], [0, 1], [2, 1], [2, 0]],
        [
            {"c0": torch.tensor([0.0, 1.0])},
            {"c0": torch.tensor([1.0, 0.0])},
            {"c0": torch.tensor([0.0, 1.0])},
            {"c0": torch.tensor([0.0, 1.0])},
        ],
    )
    print(f"update: \n: {hist}")
    hist.iadd(hist)
    print(f"merge: \n: {hist}")


def test_tensor():
    # paillier
    hist = Histogram.create(1, [3, 2], {"c0": {"type": "paillier", "stride": 2, "pk": pk, "evaluator": evaluator}})
    print(f"created:\n {hist}")
    hist.i_update(
        [0, 0, 0, 0],
        [[1, 0], [0, 1], [2, 1], [2, 0]],
        [
            {"c0": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0, 1.0])), False)},
            {"c0": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, 0.0])), False)},
            {"c0": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0, 1.0])), False)},
            {"c0": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0, 1.0])), False)},
        ],
    )
    print(f"update: \n: {hist}")
    hist.iadd(hist)
    print(f"merge: \n: {hist}")
    hist = hist.decrypt({"c0": sk})
    print(f"decrypt: \n: {hist}")
    hist = hist.decode({"c0": (coder, torch.float32)})
    print(f"decode: \n {hist}")


def create_mixed_hist():
    hist = Histogram.create(
        1,
        [3, 2],
        {
            "g": {"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
            "h": {"type": "paillier", "stride": 2, "pk": pk, "evaluator": evaluator},
            "1": {"type": "tensor", "stride": 2, "dtype": torch.int64},
        },
    )
    print(f"created:\n {hist}")
    hist.i_update(
        [0, 0, 0, 0],
        [[1, 0], [0, 1], [2, 1], [2, 0]],
        [
            {
                "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                "1": torch.tensor([1, -1]),
            },
            {
                "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0])), False),
                "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0, -0.0])), False),
                "1": torch.tensor([1, -1]),
            },
            {
                "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                "1": torch.tensor([1, -1]),
            },
            {
                "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                "1": torch.tensor([1, -1]),
            },
        ],
    )
    print(f"update: \n: {hist}")
    return hist


def test_mixed():
    # mixed
    hist = create_mixed_hist()
    hist.iadd(hist)
    print(f"merge: \n: {hist}")
    hist = hist.decrypt({"g": sk, "h": sk})
    print(f"decrypt: \n: {hist}")
    hist = hist.decode({"g": (coder, torch.float64), "h": (coder, torch.float64)})
    print(f"decode: \n {hist}")


def test_flatten():
    # flatten
    hist = create_mixed_hist()
    hist = hist.flatten_all_feature_bins()
    print(f"flatten: \n: {hist}")
    hist = hist.decrypt({"g": sk, "h": sk})
    print(f"decrypt: \n: {hist}")
    hist = hist.decode({"g": (coder, torch.float64), "h": (coder, torch.float64)})
    print(f"decode: \n {hist}")


def test_cumsum():
    hist = create_mixed_hist()
    hist.i_cumsum_bins()
    print(f"cumsum: \n: {hist}")
    hist = hist.decrypt({"g": sk, "h": sk})
    print(f"decrypt: \n: {hist}")
    hist = hist.decode({"g": (coder, torch.float64), "h": (coder, torch.float64)})
    print(f"decode: \n {hist}")


def test_sum():
    hist = create_mixed_hist()
    hist = hist.sum_bins()
    print(f"sum: \n: {hist}")
    hist = hist.decrypt({"g": sk, "h": sk})
    print(f"decrypt: \n: {hist}")
    hist = hist.decode({"g": (coder, torch.float64), "h": (coder, torch.float64)})
    print(f"decode: \n {hist}")


def test_serde():
    hist = create_mixed_hist()
    hist_bytes = pickle.dumps(hist)
    hist2 = pickle.loads(hist_bytes)
    print(f"hist2: \n: {hist2}")
    hist2 = hist2.decrypt({"g": sk, "h": sk})
    hist2 = hist2.decode({"g": (coder, torch.float64), "h": (coder, torch.float64)})
    print(f"hist2: \n: {hist2}")


def create_complex_hist(num_nodes, feature_bins, count):
    hist = Histogram.create(
        num_nodes,
        feature_bins,
        {
            "g": {"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
            "h": {"type": "paillier", "stride": 2, "pk": pk, "evaluator": evaluator},
            "1": {"type": "tensor", "stride": 2, "dtype": torch.int64},
        },
    )
    print(f"created:\n {hist}")

    for i in range(count):
        hist.i_update(
            [random.randint(0, num_nodes - 1)],
            [[random.randint(0, bins - 1) for bins in feature_bins]],
            [
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.rand(1)), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.rand(2)), False),
                    "1": torch.tensor([1, -1]),
                },
            ],
        )
    print(f"update: \n: {hist}")
    return hist


def test_split():
    hist = create_complex_hist(2, [3, 2, 4, 5], 100)
    for pid, split in hist.to_splits(3):
        print(f"split: \n: {split._data}")


def test_i_shuffle():
    hist = create_complex_hist(2, [3, 2, 4, 5], 100)
    hist = hist.i_shuffle(0)
    print(f"i_shuffle: \n: {hist}")


def test_distributed_hist():
    print()
    computing = CSession()
    ctx = Context(computing=computing)
    fake_data = [
        (
            [0, 0, 0, 0],
            [[1, 0], [0, 1], [2, 1], [2, 0]],
            [
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                    "1": torch.tensor([0.0, 1.0, -1.0]),
                },
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0, -0.0])), False),
                    "1": torch.tensor([1.0, 0.0, -0.0]),
                },
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                    "1": torch.tensor([0.0, 1.0, -1.0]),
                },
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                    "1": torch.tensor([0.0, 1.0, -1.0]),
                },
            ],
        ),
        (
            [0, 0, 0, 0],
            [[1, 0], [0, 1], [2, 1], [2, 0]],
            [
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                    "1": torch.tensor([0.0, 1.0, -1.0]),
                },
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0, -0.0])), False),
                    "1": torch.tensor([1.0, 0.0, -0.0]),
                },
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                    "1": torch.tensor([0.0, 1.0, -1.0]),
                },
                {
                    "g": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([0.0])), False),
                    "h": pk.encrypt_encoded(coder.encode_f32_vec(torch.tensor([1.0, -1.0])), False),
                    "1": torch.tensor([0.0, 1.0, -1.0]),
                },
            ],
        ),
    ]
    table = ctx.computing.parallelize(fake_data, 2, include_key=False)
    hist = DistributedHistogram(
        node_size=2,
        feature_bin_sizes=[3, 2],
        value_schemas={
            "g": {"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
            "h": {"type": "paillier", "stride": 2, "pk": pk, "evaluator": evaluator},
            "1": {"type": "tensor", "stride": 3, "dtype": torch.float32},
        },
        seed=0,
    )
    shuffled = hist.i_update(table)
    out = shuffled.decrypt(
        sk_map={"g": sk, "h": sk}, coder_map={"g": (coder, torch.float32), "h": (coder, torch.float32)}
    )
    print(out)
    out = out.reshape([3, 2])
    out.i_shuffle(seed=0, reverse=True)
    print(out)


def test_distributed_hist_calling_from_df():
    import multiprocessing
    import random

    import pandas as pd
    from fate.arch.dataframe import DataFrame, PandasReader

    multiprocessing.set_start_method("fork")

    data_list = []
    for i in range(100):
        row = [f"sample_id_{i}", f"match_id_{i}", random.randint(0, 1)] + [random.randint(0, j + 1) for j in range(4)]
        data_list.append(row)

    pd_df = pd.DataFrame(data_list, columns=["sample_id", "match_id", "y", "x0", "x1", "x2", "x3"])
    node_df = pd.DataFrame(
        [[f"sample_id_{i}", f"match_id_{i}", random.randint(0, 3)] for i in range(100)],
        columns=["sample_id", "match_id", "node_id"],
    )

    computing = CSession()
    from fate.arch.federation.standalone import StandaloneFederation

    arbiter = ("arbiter", "10000")
    guest = ("guest", "10000")
    host = ("host", "9999")
    name = "fed"
    ctx = Context(computing=computing, federation=StandaloneFederation(computing, name, guest, [guest, host, arbiter]))

    df_reader = PandasReader(
        sample_id_name="sample_id",
        match_id_name="match_id",
        label_name="y",
        label_type="int32",
        dtype={"x0": "int32", "x1": "int32", "x2": "int32", "x3": "int32"},
    )
    df = df_reader.to_frame(ctx, pd_df)

    pos_reader = PandasReader(sample_id_name="sample_id", match_id_name="match_id", dtype={"node_id": "int32"})
    pos_df = pos_reader.to_frame(ctx, node_df)

    one_df = df.create_frame()
    one_df["one"] = 1

    encryptor = kit.get_tensor_encryptor()
    # decryptor = kit.get_tensor_encryptor()

    targets = dict(one=one_df["one"].as_tensor(), g=encryptor.encrypt_tensor(df.label.as_tensor()))

    hist = DistributedHistogram(
        node_size=4,
        feature_bin_sizes=[2, 3, 4, 5],
        value_schemas={
            "g": {"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
            "one": {"type": "tensor", "stride": 1, "dtype": torch.float32},
        },
        seed=0,
    )

    stat_obj = df.distributed_hist_stat(hist, pos_df, targets)


# test_distributed_hist_calling_from_df()
