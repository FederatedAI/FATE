import typing

import torch


class Hist:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.data: typing.Dict[int, typing.Dict[int, typing.Any]] = {}

    def update(self, features, labels):
        shape_x, shape_y = features.shape
        for i in range(shape_x):
            for j in range(shape_y):
                v = features[i, j].item()
                if j not in self.data:
                    self.data[j] = {}
                if v not in self.data[j]:
                    self.data[j][v] = labels[i]
                else:
                    self.data[j][v] = torch.add(self.data[j][v], labels[i])

    def merge(self, hist):
        for k in hist.data:
            if k not in self.data:
                self.data[k] = hist.data[k]
            else:
                for kk in hist.data[k]:
                    if kk not in self.data[k]:
                        self.data[k][kk] = hist.data[k][kk]
                    else:
                        self.data[k][kk] = torch.add(self.data[k][kk], hist.data[k][kk])
        return self

    def cumsum(self):
        for k in self.data:
            s = 0
            for kk in sorted(self.data[k].keys()):
                s = torch.add(s, self.data[k][kk])
                self.data[k][kk] = s
        return self

    def __sub__(self, other: "Hist"):
        out = Hist(self.feature_names)
        for j in self.data:
            out.data[j] = {}
            for v in self.data[j]:
                if v not in other.data[j]:
                    out.data[j][v] = self.data[j][v]
                else:
                    out.data[j][v] = torch.sub(self.data[j][v], other.data[j][v])
        return out

    def to_dict(self):
        return {name: self.data[i] for i, name in enumerate(self.feature_names)}

    def decrypt(self, pri):
        for j in self.data:
            for v in self.data[j]:
                self.data[j][v] = pri.decrypt(self.data[j][v])
        return self

    def encrypt(self, pub):
        for j in self.data:
            for v in self.data[j]:
                self.data[j][v] = pub.encrypt(self.data[j][v])
        return self


if __name__ == "__main__":
    import numpy as np
    from fate.arch import Context

    ctx = Context()

    pub, pri = ctx.cipher.phe.keygen(options={"key_length": 1024})
    hist = Hist(["a", "b"])
    features = np.array([[1, 0], [0, 1], [2, 1], [2, 0]])
    labels = torch.tensor(np.array([0, 1, 0, 0]))
    hist.update(features, labels)
    hist.encrypt(pub)
    hist.decrypt(pri)
    print((hist - hist).data)
