import torch
import torch as t
import copy
from torch.nn import Module


class OpBase(object):

    def __init__(self):
        self.param_dict = {}

    def to_dict(self):
        ret = copy.deepcopy(self.param_dict)
        ret['op'] = type(self).__name__
        return ret


class Astype(Module, OpBase):

    def __init__(self, cast_type: str):
        OpBase.__init__(self)
        Module.__init__(self)
        assert cast_type in ['float', 'int', 'bool', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'float16']
        self.param_dict['cast_type'] = cast_type
        self.cast_type = cast_type
        self.cast_type_map = {
            'float': t.float,
            'int': t.int,
            'bool': t.bool,
            'float32': t.float32,
            'float64': t.float64,
            'float16': t.float16,
            'int8': t.int8,
            'int16': t.int16,
            'int32': t.int32,
            'int64': t.int64,
        }

    def forward(self, tensor: t.Tensor, **kwargs):
        return tensor.type(self.cast_type_map[self.cast_type])


class Flatten(Module, OpBase):

    def __init__(self, start_dim=0, end_dim=-1):
        OpBase.__init__(self)
        Module.__init__(self)
        self.param_dict['start_dim'] = start_dim
        self.param_dict['end_dim'] = end_dim

    def forward(self, tensor):
        return tensor.flatten(**self.param_dict)


class Reshape(Module, OpBase):

    def __init__(self, shape):
        OpBase.__init__(self)
        Module.__init__(self)
        assert isinstance(shape, tuple) or isinstance(shape, list)
        self.shape = shape
        self.param_dict['shape'] = list(shape)

    def forward(self, tensor: t.Tensor):
        return tensor.reshape(shape=self.shape)


class Index(Module, OpBase):

    def __init__(self, index):
        OpBase.__init__(self)
        Module.__init__(self)
        assert isinstance(index, int)
        self.param_dict['index'] = index

    def forward(self, content):
        return content[self.param_dict['index']]


class Select(Module, OpBase):

    def __init__(self, dim, idx):
        OpBase.__init__(self)
        Module.__init__(self)
        self.param_dict = {'dim': dim, 'index': idx}

    def forward(self, tensor):
        return tensor.select(self.param_dict['dim'], self.param_dict['index'])


class SelectRange(Module, OpBase):

    def __init__(self, dim, start, end):
        OpBase.__init__(self)
        Module.__init__(self)
        self.param_dict = {'dim': dim, 'start': start, 'end': end}

    def forward(self, tensor):
        return tensor.select(self.param_dict['dim'], -1)[self.param_dict['start']: self.param_dict['end']]


class Sum(Module, OpBase):

    def __init__(self, dim):
        OpBase.__init__(self)
        Module.__init__(self)
        assert isinstance(dim, int)
        self.param_dict['dim'] = dim

    def forward(self, tensor):
        return tensor.sum(dim=self.param_dict['dim'])


class Squeeze(Module, OpBase):

    def __init__(self, **kwargs):
        OpBase.__init__(self)
        Module.__init__(self)

    def forward(self, tensor: t.Tensor):
        return tensor.squeeze()


class Unsqueeze(Sum, OpBase):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__(dim)

    def forward(self, tensor: t.Tensor):
        return tensor.unsqueeze(self.param_dict['dim'])
