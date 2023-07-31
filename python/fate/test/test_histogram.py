import pickle

import torch
from fate.arch import Context
from fate.arch.protocol.histogram import Histogram

ctx = Context()
kit = ctx.cipher.phe.setup(options={"kind": "paillier_vector_based", "key_length": 1024})
sk, pk, coder = kit.sk, kit.pk, kit.coder


def test_plain():
    # plain
    hist = Histogram(1, [3, 2])
    hist.set_value_schema({"c0": {"type": "tensor", "stride": 2}})
    print(f"created:\n {hist}")
    hist.update(
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
    hist.merge(hist)
    print(f"merge: \n: {hist}")


def test_tensor():
    # paillier
    hist = Histogram(1, [3, 2])
    hist.set_value_schema({"c0": {"type": "paillier", "stride": 2, "pk": pk}})
    print(f"created:\n {hist}")
    hist.update(
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
    hist.merge(hist)
    print(f"merge: \n: {hist}")
    hist = hist.decrypt({"c0": sk})
    print(f"decrypt: \n: {hist}")
    hist = hist.decode({"c0": (coder, torch.float32)})
    print(f"decode: \n {hist}")


def create_mixed_hist():
    hist = Histogram(1, [3, 2])
    hist.set_value_schema(
        {
            "g": {"type": "paillier", "stride": 1, "pk": pk},
            "h": {"type": "paillier", "stride": 2, "pk": pk},
            "1": {"type": "tensor", "stride": 2, "dtype": torch.int64},
        }
    )
    print(f"created:\n {hist}")
    hist.update(
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
    hist.merge(hist)
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
    hist.cumsum_bins()
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
