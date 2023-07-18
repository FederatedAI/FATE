import torch


class Histogram:
    """
    Hist is a class to store the histogram of features and labels.

    Examples:
        >>> features = torch.tensor([[1, 0], [0, 1], [2, 1], [2, 0]])
        >>> labels = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 1]], dtype=torch.float32)
        >>> hist = Histogram(binning_shape=(2,), src_shape=(2,), bin_nums=torch.LongTensor([3, 2]), dtype=torch.float32)
        >>> hist.update(features, labels)
        >>> histogram = hist.to_masked_histogram()
        >>> histogram = histogram.cumsum()
        >>> histogram.data.reshape(3, -1)
        tensor([[1., 0., 0., 2.],
                [1., 1., 1., 3.],
                [1., 3., 1., 3.]])
    """

    def __init__(self, binning_shape, src_shape, bin_nums, dtype=torch.long):
        self._binning_shape = binning_shape
        self._src_shape = src_shape
        self._bin_nums = bin_nums
        self._dtype = dtype

        self._binning_shape_size = len(self._binning_shape)
        self._src_shape_size = len(self._src_shape)
        self._max_bin_num = max(self._bin_nums)

        # initialize with zeros
        self.data = torch.zeros(self._max_bin_num, *binning_shape, *src_shape, dtype=self._dtype)
        self.mask = torch.ones(self._max_bin_num, *binning_shape, *src_shape, dtype=torch.bool)

    def update(self, index: torch.Tensor, src: torch.Tensor):
        assert index.shape[0] == src.shape[0]
        assert index.shape[1:] == self._binning_shape
        assert src.shape[1:] == self._src_shape
        self.data = torch.histogram_f(self.data, index, src)

    def to_masked_histogram(self):
        return MaskedHistogram.create(self.data, self.mask)


class MaskedHistogram:
    """
    MaskedHistogram is a class to store the masked histogram of features and labels.

    Since the binning number of each feature is different, we use a mask to indicate the valid bins.
    This may be useful when we want to efficiently operate on the histogram especially when the data is encrypted.
    """

    def __init__(self, data):
        self.data = data

    @classmethod
    def create(cls, data, mask):
        # TODO: enable masked histogram
        # ignore the mask for now
        return MaskedHistogram(data)

    def merge(self, masked_histogram: "MaskedHistogram"):
        return MaskedHistogram(torch.add(self.data, masked_histogram.data))

    def cumsum(self):
        return MaskedHistogram(self.data.cumsum(dim=0))


if __name__ == "__main__":
    from fate.arch import Context
    from fate.arch.computing.standalone import CSession

    ctx = Context(computing=CSession())
    kit = ctx.cipher.phe.setup(key_length=1024)
    features = torch.tensor([[1, 0], [0, 1], [2, 1], [2, 0]])
    labels = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 1]], dtype=torch.float32)
    labels = kit.encryptor.encrypt(labels)
    hist = Histogram(binning_shape=(2,), src_shape=(2,), bin_nums=torch.LongTensor([3, 2]), dtype=torch.float32)
    hist.update(features, labels)
    histogram = hist.to_masked_histogram()
    histogram = histogram.cumsum()
    print(histogram.data.reshape(3, -1))
