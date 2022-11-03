from ._base import Shape, _CPUStorage


def slice(storage: _CPUStorage, key):
    output_data = storage.data[key]
    return _CPUStorage(storage.dtype, Shape(output_data.shape), output_data)


custom_ops = dict(slice=slice)
