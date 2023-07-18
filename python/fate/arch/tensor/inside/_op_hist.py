import typing


class Hist:
    def __init__(self):
        self.data: typing.Dict[int, typing.Dict[int, typing.Any]] = {}

    def update(self, features, labels):
        shape_x, shape_y = features.shape
        for i in range(shape_x):
            for j in range(shape_y):
                v = features[i, j]
                if j not in self.data:
                    self.data[j] = {}
                if v not in self.data[j]:
                    self.data[j][v] = labels[i]
                else:
                    self.data[j][v] += labels[i]

    def merge(self, hist):
        for k in hist.data:
            if k not in self.data:
                self.data[k] = hist.data[k]
            else:
                for kk in hist.data[k]:
                    if kk not in self.data[k]:
                        self.data[k][kk] = hist.data[k][kk]
                    else:
                        self.data[k][kk] += hist.data[k][kk]
        return self

    def cumsum(self):
        for k in self.data:
            s = 0
            for kk in sorted(self.data[k].keys()):
                s += self.data[k][kk]
                self.data[k][kk] = s
        return self

    def __sub__(self, other: "Hist"):
        out = Hist()
        for j in self.data:
            out.data[j] = {}
            for v in self.data[j]:
                if v not in other.data[j]:
                    out.data[j][v] = self.data[j][v]
                else:
                    out.data[j][v] = self.data[j][v] - other.data[j][v]
        return out


if __name__ == "__main__":
    import numpy as np

    hist = Hist()
    features = np.array([[1, 0], [0, 1], [2, 1], [2, 0]])
    labels = np.array([0, 1, 0, 0])
    hist.update(features, labels)
    print((hist - hist).data)
