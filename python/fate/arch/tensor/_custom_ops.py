import torch


def encrypt_f(tensor, encryptor):
    if isinstance(tensor, torch.Tensor):
        return encryptor.encrypt(tensor.detach())
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(encrypt_f, (type(tensor),), (tensor, encryptor), None)
    raise NotImplementedError("")


def decrypt_f(tensor, decryptor):
    if isinstance(tensor, torch.Tensor.detach):
        return decryptor.encrypt(tensor.detach())
    else:
        # torch tensor-like
        if hasattr(tensor, "__torch_function__"):
            return tensor.__torch_function__(decrypt_f, (type(tensor),), (tensor, decryptor), None)
    raise NotImplementedError("")


def rmatmul_f(input, other):
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor):
        return torch.matmul(other, input)
    else:
        # torch tensor-like
        if isinstance(input, torch.Tensor):
            return torch.matmul(other, input)

        else:
            if hasattr(input, "__torch_function__"):
                return input.__torch_function__(rmatmul_f, (type(input), type(other)), (input, other), None)
    raise NotImplementedError("")


def to_local_f(input):
    if isinstance(input, torch.Tensor):
        return input

    else:
        # torch tensor-like
        if hasattr(input, "__torch_function__"):
            return input.__torch_function__(to_local_f, (type(input),), (input,), None)
    raise NotImplementedError("")


def slice_f(input, arg):
    if isinstance(input, torch.Tensor):
        return input[arg]

    else:
        # torch tensor-like
        if hasattr(input, "__torch_function__"):
            out = input.__torch_function__(slice_f, (type(input),), (input, arg), None)
            if out == NotImplemented:
                raise NotImplementedError(f"slice_f: {input}")
            return out

    raise NotImplementedError(f"slice_f: {input}")


def histogram_f(histogram, index, src):
    """
    Update the histogram with the given index and src.

    to be more specific, for each i, j, k, we do:
        histogram[index[i, j], j, k] += src[i, k]
    where
        i is the index of sample, ranging from 0 to src_shape[0] - 1
        j is the indexes of index, could be a list of indexes
        k is the indexes of values, could be a list of indexes
    we implement this by expanding the index and src to the same shape, and then do scatter_add.
    the expanded shape is:
        index: (src_shape[0], *index_shape[1:], *src_shape[1:])
        src: (src_shape[0], *index_shape[1:], *src_shape[1:])
        histogram: (max_bin_num, *index_shape[1:], *src_shape[1:])
    """

    # sue scatter_add to update histogram
    if isinstance(histogram, torch.Tensor) and isinstance(index, torch.Tensor) and isinstance(src, torch.Tensor):
        assert index.shape[0] == src.shape[0]
        index_skip_first_dim = index.shape[1:]
        src_skip_first_dim = src.shape[1:]
        sample_num = index.shape[0]
        index = index.view(sample_num, *index_skip_first_dim, *([1] * len(src_skip_first_dim))).expand(
            -1, *index_skip_first_dim, *src_skip_first_dim
        )
        src = src.view(sample_num, *([1] * len(index_skip_first_dim)), *src_skip_first_dim).expand(
            -1, *index_skip_first_dim, *src_skip_first_dim
        )
        return torch.scatter_add(histogram, 0, index, src)
    elif hasattr(src, "__torch_function__"):
        out = src.__torch_function__(
            histogram_f, (type(histogram), type(index), type(src)), (histogram, index, src), None
        )
        if out != NotImplemented:
            return out

        # torch tensor-like, without scatter_add
        else:
            for i in range(index.shape[0]):
                for j in range(index.shape[1]):
                    for k in range(src.shape[1]):
                        histogram[index[i, j], j, k] += src[i, k]
            return histogram
    raise NotImplementedError("")


# hook custom ops to torch
torch.encrypt_f = encrypt_f
torch.decrypt_f = decrypt_f
torch.rmatmul_f = rmatmul_f
torch.to_local_f = to_local_f
torch.slice_f = slice_f
torch.histogram_f = histogram_f
