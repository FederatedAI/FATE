from torch import nn
from fate.components.components.nn.fate_torch.base import FateTorch


class Bilinear(nn.modules.linear.Bilinear, FateTorch):
        
    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in1_features'] = in1_features
        self.param_dict['in2_features'] = in2_features
        self.param_dict['out_features'] = out_features
        self.param_dict.update(kwargs)
        nn.modules.linear.Bilinear.__init__(self, **self.param_dict)
    
    
class Identity(nn.modules.linear.Identity, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.linear.Identity.__init__(self, **self.param_dict)
    
    
class LazyLinear(nn.modules.linear.LazyLinear, FateTorch):
        
    def __init__(self, out_features, bias=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_features'] = out_features
        self.param_dict.update(kwargs)
        nn.modules.linear.LazyLinear.__init__(self, **self.param_dict)
    
    
class Linear(nn.modules.linear.Linear, FateTorch):
        
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_features'] = in_features
        self.param_dict['out_features'] = out_features
        self.param_dict.update(kwargs)
        nn.modules.linear.Linear.__init__(self, **self.param_dict)
    
    
class NonDynamicallyQuantizableLinear(nn.modules.linear.NonDynamicallyQuantizableLinear, FateTorch):
        
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_features'] = in_features
        self.param_dict['out_features'] = out_features
        self.param_dict.update(kwargs)
        nn.modules.linear.NonDynamicallyQuantizableLinear.__init__(self, **self.param_dict)
    
    
class GRU(nn.modules.rnn.GRU, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.rnn.GRU.__init__(self, **self.param_dict)
    
    
class GRUCell(nn.modules.rnn.GRUCell, FateTorch):
        
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['input_size'] = input_size
        self.param_dict['hidden_size'] = hidden_size
        self.param_dict.update(kwargs)
        nn.modules.rnn.GRUCell.__init__(self, **self.param_dict)
    
    
class LSTM(nn.modules.rnn.LSTM, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.rnn.LSTM.__init__(self, **self.param_dict)
    
    
class LSTMCell(nn.modules.rnn.LSTMCell, FateTorch):
        
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['input_size'] = input_size
        self.param_dict['hidden_size'] = hidden_size
        self.param_dict.update(kwargs)
        nn.modules.rnn.LSTMCell.__init__(self, **self.param_dict)
    
    
class RNN(nn.modules.rnn.RNN, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.rnn.RNN.__init__(self, **self.param_dict)
    
    
class RNNBase(nn.modules.rnn.RNNBase, FateTorch):
        
    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['num_layers'] = num_layers
        self.param_dict['bias'] = bias
        self.param_dict['batch_first'] = batch_first
        self.param_dict['dropout'] = dropout
        self.param_dict['bidirectional'] = bidirectional
        self.param_dict['proj_size'] = proj_size
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['mode'] = mode
        self.param_dict['input_size'] = input_size
        self.param_dict['hidden_size'] = hidden_size
        self.param_dict.update(kwargs)
        nn.modules.rnn.RNNBase.__init__(self, **self.param_dict)
    
    
class RNNCell(nn.modules.rnn.RNNCell, FateTorch):
        
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['bias'] = bias
        self.param_dict['nonlinearity'] = nonlinearity
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['input_size'] = input_size
        self.param_dict['hidden_size'] = hidden_size
        self.param_dict.update(kwargs)
        nn.modules.rnn.RNNCell.__init__(self, **self.param_dict)
    
    
class RNNCellBase(nn.modules.rnn.RNNCellBase, FateTorch):
        
    def __init__(self, input_size, hidden_size, bias, num_chunks, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['input_size'] = input_size
        self.param_dict['hidden_size'] = hidden_size
        self.param_dict['bias'] = bias
        self.param_dict['num_chunks'] = num_chunks
        self.param_dict.update(kwargs)
        nn.modules.rnn.RNNCellBase.__init__(self, **self.param_dict)
    
    
class Embedding(nn.modules.sparse.Embedding, FateTorch):
        
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding_idx'] = padding_idx
        self.param_dict['max_norm'] = max_norm
        self.param_dict['norm_type'] = norm_type
        self.param_dict['scale_grad_by_freq'] = scale_grad_by_freq
        self.param_dict['sparse'] = sparse
        self.param_dict['_weight'] = _weight
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_embeddings'] = num_embeddings
        self.param_dict['embedding_dim'] = embedding_dim
        self.param_dict.update(kwargs)
        nn.modules.sparse.Embedding.__init__(self, **self.param_dict)
    
    
class EmbeddingBag(nn.modules.sparse.EmbeddingBag, FateTorch):
        
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, mode='mean', sparse=False, _weight=None, include_last_offset=False, padding_idx=None, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['max_norm'] = max_norm
        self.param_dict['norm_type'] = norm_type
        self.param_dict['scale_grad_by_freq'] = scale_grad_by_freq
        self.param_dict['mode'] = mode
        self.param_dict['sparse'] = sparse
        self.param_dict['_weight'] = _weight
        self.param_dict['include_last_offset'] = include_last_offset
        self.param_dict['padding_idx'] = padding_idx
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_embeddings'] = num_embeddings
        self.param_dict['embedding_dim'] = embedding_dim
        self.param_dict.update(kwargs)
        nn.modules.sparse.EmbeddingBag.__init__(self, **self.param_dict)
    
    
class AlphaDropout(nn.modules.dropout.AlphaDropout, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout.AlphaDropout.__init__(self, **self.param_dict)
    
    
class Dropout(nn.modules.dropout.Dropout, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout.Dropout.__init__(self, **self.param_dict)
    
    
class Dropout1d(nn.modules.dropout.Dropout1d, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout.Dropout1d.__init__(self, **self.param_dict)
    
    
class Dropout2d(nn.modules.dropout.Dropout2d, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout.Dropout2d.__init__(self, **self.param_dict)
    
    
class Dropout3d(nn.modules.dropout.Dropout3d, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout.Dropout3d.__init__(self, **self.param_dict)
    
    
class FeatureAlphaDropout(nn.modules.dropout.FeatureAlphaDropout, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout.FeatureAlphaDropout.__init__(self, **self.param_dict)
    
    
class _DropoutNd(nn.modules.dropout._DropoutNd, FateTorch):
        
    def __init__(self, p=0.5, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.dropout._DropoutNd.__init__(self, **self.param_dict)
    
    
class CELU(nn.modules.activation.CELU, FateTorch):
        
    def __init__(self, alpha=1.0, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['alpha'] = alpha
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.CELU.__init__(self, **self.param_dict)
    
    
class ELU(nn.modules.activation.ELU, FateTorch):
        
    def __init__(self, alpha=1.0, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['alpha'] = alpha
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.ELU.__init__(self, **self.param_dict)
    
    
class GELU(nn.modules.activation.GELU, FateTorch):
        
    def __init__(self, approximate='none', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['approximate'] = approximate
        self.param_dict.update(kwargs)
        nn.modules.activation.GELU.__init__(self, **self.param_dict)
    
    
class GLU(nn.modules.activation.GLU, FateTorch):
        
    def __init__(self, dim=-1, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dim'] = dim
        self.param_dict.update(kwargs)
        nn.modules.activation.GLU.__init__(self, **self.param_dict)
    
    
class Hardshrink(nn.modules.activation.Hardshrink, FateTorch):
        
    def __init__(self, lambd=0.5, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['lambd'] = lambd
        self.param_dict.update(kwargs)
        nn.modules.activation.Hardshrink.__init__(self, **self.param_dict)
    
    
class Hardsigmoid(nn.modules.activation.Hardsigmoid, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.Hardsigmoid.__init__(self, **self.param_dict)
    
    
class Hardswish(nn.modules.activation.Hardswish, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.Hardswish.__init__(self, **self.param_dict)
    
    
class Hardtanh(nn.modules.activation.Hardtanh, FateTorch):
        
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['min_val'] = min_val
        self.param_dict['max_val'] = max_val
        self.param_dict['inplace'] = inplace
        self.param_dict['min_value'] = min_value
        self.param_dict['max_value'] = max_value
        self.param_dict.update(kwargs)
        nn.modules.activation.Hardtanh.__init__(self, **self.param_dict)
    
    
class LeakyReLU(nn.modules.activation.LeakyReLU, FateTorch):
        
    def __init__(self, negative_slope=0.01, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['negative_slope'] = negative_slope
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.LeakyReLU.__init__(self, **self.param_dict)
    
    
class LogSigmoid(nn.modules.activation.LogSigmoid, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.activation.LogSigmoid.__init__(self, **self.param_dict)
    
    
class LogSoftmax(nn.modules.activation.LogSoftmax, FateTorch):
        
    def __init__(self, dim=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dim'] = dim
        self.param_dict.update(kwargs)
        nn.modules.activation.LogSoftmax.__init__(self, **self.param_dict)
    
    
class Mish(nn.modules.activation.Mish, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.Mish.__init__(self, **self.param_dict)
    
    
class MultiheadAttention(nn.modules.activation.MultiheadAttention, FateTorch):
        
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dropout'] = dropout
        self.param_dict['bias'] = bias
        self.param_dict['add_bias_kv'] = add_bias_kv
        self.param_dict['add_zero_attn'] = add_zero_attn
        self.param_dict['kdim'] = kdim
        self.param_dict['vdim'] = vdim
        self.param_dict['batch_first'] = batch_first
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['embed_dim'] = embed_dim
        self.param_dict['num_heads'] = num_heads
        self.param_dict.update(kwargs)
        nn.modules.activation.MultiheadAttention.__init__(self, **self.param_dict)
    
    
class PReLU(nn.modules.activation.PReLU, FateTorch):
        
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['num_parameters'] = num_parameters
        self.param_dict['init'] = init
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict.update(kwargs)
        nn.modules.activation.PReLU.__init__(self, **self.param_dict)
    
    
class RReLU(nn.modules.activation.RReLU, FateTorch):
        
    def __init__(self, lower=0.125, upper=0.3333333333333333, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['lower'] = lower
        self.param_dict['upper'] = upper
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.RReLU.__init__(self, **self.param_dict)
    
    
class ReLU(nn.modules.activation.ReLU, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.ReLU.__init__(self, **self.param_dict)
    
    
class ReLU6(nn.modules.activation.ReLU6, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.ReLU6.__init__(self, **self.param_dict)
    
    
class SELU(nn.modules.activation.SELU, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.SELU.__init__(self, **self.param_dict)
    
    
class SiLU(nn.modules.activation.SiLU, FateTorch):
        
    def __init__(self, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict.update(kwargs)
        nn.modules.activation.SiLU.__init__(self, **self.param_dict)
    
    
class Sigmoid(nn.modules.activation.Sigmoid, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.activation.Sigmoid.__init__(self, **self.param_dict)
    
    
class Softmax(nn.modules.activation.Softmax, FateTorch):
        
    def __init__(self, dim=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dim'] = dim
        self.param_dict.update(kwargs)
        nn.modules.activation.Softmax.__init__(self, **self.param_dict)
    
    
class Softmax2d(nn.modules.activation.Softmax2d, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.activation.Softmax2d.__init__(self, **self.param_dict)
    
    
class Softmin(nn.modules.activation.Softmin, FateTorch):
        
    def __init__(self, dim=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dim'] = dim
        self.param_dict.update(kwargs)
        nn.modules.activation.Softmin.__init__(self, **self.param_dict)
    
    
class Softplus(nn.modules.activation.Softplus, FateTorch):
        
    def __init__(self, beta=1, threshold=20, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['beta'] = beta
        self.param_dict['threshold'] = threshold
        self.param_dict.update(kwargs)
        nn.modules.activation.Softplus.__init__(self, **self.param_dict)
    
    
class Softshrink(nn.modules.activation.Softshrink, FateTorch):
        
    def __init__(self, lambd=0.5, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['lambd'] = lambd
        self.param_dict.update(kwargs)
        nn.modules.activation.Softshrink.__init__(self, **self.param_dict)
    
    
class Softsign(nn.modules.activation.Softsign, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.activation.Softsign.__init__(self, **self.param_dict)
    
    
class Tanh(nn.modules.activation.Tanh, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.activation.Tanh.__init__(self, **self.param_dict)
    
    
class Tanhshrink(nn.modules.activation.Tanhshrink, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.activation.Tanhshrink.__init__(self, **self.param_dict)
    
    
class Threshold(nn.modules.activation.Threshold, FateTorch):
        
    def __init__(self, threshold, value, inplace=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['inplace'] = inplace
        self.param_dict['threshold'] = threshold
        self.param_dict['value'] = value
        self.param_dict.update(kwargs)
        nn.modules.activation.Threshold.__init__(self, **self.param_dict)
    
    
class Conv1d(nn.modules.conv.Conv1d, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.Conv1d.__init__(self, **self.param_dict)
    
    
class Conv2d(nn.modules.conv.Conv2d, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.Conv2d.__init__(self, **self.param_dict)
    
    
class Conv3d(nn.modules.conv.Conv3d, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.Conv3d.__init__(self, **self.param_dict)
    
    
class ConvTranspose1d(nn.modules.conv.ConvTranspose1d, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['dilation'] = dilation
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.ConvTranspose1d.__init__(self, **self.param_dict)
    
    
class ConvTranspose2d(nn.modules.conv.ConvTranspose2d, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['dilation'] = dilation
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.ConvTranspose2d.__init__(self, **self.param_dict)
    
    
class ConvTranspose3d(nn.modules.conv.ConvTranspose3d, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['dilation'] = dilation
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.ConvTranspose3d.__init__(self, **self.param_dict)
    
    
class LazyConv1d(nn.modules.conv.LazyConv1d, FateTorch):
        
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.LazyConv1d.__init__(self, **self.param_dict)
    
    
class LazyConv2d(nn.modules.conv.LazyConv2d, FateTorch):
        
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.LazyConv2d.__init__(self, **self.param_dict)
    
    
class LazyConv3d(nn.modules.conv.LazyConv3d, FateTorch):
        
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.LazyConv3d.__init__(self, **self.param_dict)
    
    
class LazyConvTranspose1d(nn.modules.conv.LazyConvTranspose1d, FateTorch):
        
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['dilation'] = dilation
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.LazyConvTranspose1d.__init__(self, **self.param_dict)
    
    
class LazyConvTranspose2d(nn.modules.conv.LazyConvTranspose2d, FateTorch):
        
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['dilation'] = dilation
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.LazyConvTranspose2d.__init__(self, **self.param_dict)
    
    
class LazyConvTranspose3d(nn.modules.conv.LazyConvTranspose3d, FateTorch):
        
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['dilation'] = dilation
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.conv.LazyConvTranspose3d.__init__(self, **self.param_dict)
    
    
class _ConvNd(nn.modules.conv._ConvNd, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['transposed'] = transposed
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict.update(kwargs)
        nn.modules.conv._ConvNd.__init__(self, **self.param_dict)
    
    
class _ConvTransposeMixin(nn.modules.conv._ConvTransposeMixin, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.conv._ConvTransposeMixin.__init__(self, **self.param_dict)
    
    
class _ConvTransposeNd(nn.modules.conv._ConvTransposeNd, FateTorch):
        
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['in_channels'] = in_channels
        self.param_dict['out_channels'] = out_channels
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['transposed'] = transposed
        self.param_dict['output_padding'] = output_padding
        self.param_dict['groups'] = groups
        self.param_dict['bias'] = bias
        self.param_dict['padding_mode'] = padding_mode
        self.param_dict.update(kwargs)
        nn.modules.conv._ConvTransposeNd.__init__(self, **self.param_dict)
    
    
class _LazyConvXdMixin(nn.modules.conv._LazyConvXdMixin, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.conv._LazyConvXdMixin.__init__(self, **self.param_dict)
    
    
class Transformer(nn.modules.transformer.Transformer, FateTorch):
        
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['d_model'] = d_model
        self.param_dict['nhead'] = nhead
        self.param_dict['num_encoder_layers'] = num_encoder_layers
        self.param_dict['num_decoder_layers'] = num_decoder_layers
        self.param_dict['dim_feedforward'] = dim_feedforward
        self.param_dict['dropout'] = dropout
        self.param_dict['custom_encoder'] = custom_encoder
        self.param_dict['custom_decoder'] = custom_decoder
        self.param_dict['layer_norm_eps'] = layer_norm_eps
        self.param_dict['batch_first'] = batch_first
        self.param_dict['norm_first'] = norm_first
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict.update(kwargs)
        nn.modules.transformer.Transformer.__init__(self, **self.param_dict)
    
    
class TransformerDecoder(nn.modules.transformer.TransformerDecoder, FateTorch):
        
    def __init__(self, decoder_layer, num_layers, norm=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['norm'] = norm
        self.param_dict['decoder_layer'] = decoder_layer
        self.param_dict['num_layers'] = num_layers
        self.param_dict.update(kwargs)
        nn.modules.transformer.TransformerDecoder.__init__(self, **self.param_dict)
    
    
class TransformerDecoderLayer(nn.modules.transformer.TransformerDecoderLayer, FateTorch):
        
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dim_feedforward'] = dim_feedforward
        self.param_dict['dropout'] = dropout
        self.param_dict['layer_norm_eps'] = layer_norm_eps
        self.param_dict['batch_first'] = batch_first
        self.param_dict['norm_first'] = norm_first
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['d_model'] = d_model
        self.param_dict['nhead'] = nhead
        self.param_dict.update(kwargs)
        nn.modules.transformer.TransformerDecoderLayer.__init__(self, **self.param_dict)
    
    
class TransformerEncoder(nn.modules.transformer.TransformerEncoder, FateTorch):
        
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['norm'] = norm
        self.param_dict['enable_nested_tensor'] = enable_nested_tensor
        self.param_dict['mask_check'] = mask_check
        self.param_dict['encoder_layer'] = encoder_layer
        self.param_dict['num_layers'] = num_layers
        self.param_dict.update(kwargs)
        nn.modules.transformer.TransformerEncoder.__init__(self, **self.param_dict)
    
    
class TransformerEncoderLayer(nn.modules.transformer.TransformerEncoderLayer, FateTorch):
        
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['dim_feedforward'] = dim_feedforward
        self.param_dict['dropout'] = dropout
        self.param_dict['layer_norm_eps'] = layer_norm_eps
        self.param_dict['batch_first'] = batch_first
        self.param_dict['norm_first'] = norm_first
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['d_model'] = d_model
        self.param_dict['nhead'] = nhead
        self.param_dict.update(kwargs)
        nn.modules.transformer.TransformerEncoderLayer.__init__(self, **self.param_dict)
    
    
class AdaptiveAvgPool1d(nn.modules.pooling.AdaptiveAvgPool1d, FateTorch):
        
    def __init__(self, output_size, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AdaptiveAvgPool1d.__init__(self, **self.param_dict)
    
    
class AdaptiveAvgPool2d(nn.modules.pooling.AdaptiveAvgPool2d, FateTorch):
        
    def __init__(self, output_size, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AdaptiveAvgPool2d.__init__(self, **self.param_dict)
    
    
class AdaptiveAvgPool3d(nn.modules.pooling.AdaptiveAvgPool3d, FateTorch):
        
    def __init__(self, output_size, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AdaptiveAvgPool3d.__init__(self, **self.param_dict)
    
    
class AdaptiveMaxPool1d(nn.modules.pooling.AdaptiveMaxPool1d, FateTorch):
        
    def __init__(self, output_size, return_indices=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['return_indices'] = return_indices
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AdaptiveMaxPool1d.__init__(self, **self.param_dict)
    
    
class AdaptiveMaxPool2d(nn.modules.pooling.AdaptiveMaxPool2d, FateTorch):
        
    def __init__(self, output_size, return_indices=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['return_indices'] = return_indices
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AdaptiveMaxPool2d.__init__(self, **self.param_dict)
    
    
class AdaptiveMaxPool3d(nn.modules.pooling.AdaptiveMaxPool3d, FateTorch):
        
    def __init__(self, output_size, return_indices=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['return_indices'] = return_indices
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AdaptiveMaxPool3d.__init__(self, **self.param_dict)
    
    
class AvgPool1d(nn.modules.pooling.AvgPool1d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['count_include_pad'] = count_include_pad
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AvgPool1d.__init__(self, **self.param_dict)
    
    
class AvgPool2d(nn.modules.pooling.AvgPool2d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['count_include_pad'] = count_include_pad
        self.param_dict['divisor_override'] = divisor_override
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AvgPool2d.__init__(self, **self.param_dict)
    
    
class AvgPool3d(nn.modules.pooling.AvgPool3d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['count_include_pad'] = count_include_pad
        self.param_dict['divisor_override'] = divisor_override
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.AvgPool3d.__init__(self, **self.param_dict)
    
    
class FractionalMaxPool2d(nn.modules.pooling.FractionalMaxPool2d, FateTorch):
        
    def __init__(self, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['output_size'] = output_size
        self.param_dict['output_ratio'] = output_ratio
        self.param_dict['return_indices'] = return_indices
        self.param_dict['_random_samples'] = _random_samples
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.FractionalMaxPool2d.__init__(self, **self.param_dict)
    
    
class FractionalMaxPool3d(nn.modules.pooling.FractionalMaxPool3d, FateTorch):
        
    def __init__(self, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['output_size'] = output_size
        self.param_dict['output_ratio'] = output_ratio
        self.param_dict['return_indices'] = return_indices
        self.param_dict['_random_samples'] = _random_samples
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.FractionalMaxPool3d.__init__(self, **self.param_dict)
    
    
class LPPool1d(nn.modules.pooling.LPPool1d, FateTorch):
        
    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['norm_type'] = norm_type
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.LPPool1d.__init__(self, **self.param_dict)
    
    
class LPPool2d(nn.modules.pooling.LPPool2d, FateTorch):
        
    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['norm_type'] = norm_type
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.LPPool2d.__init__(self, **self.param_dict)
    
    
class MaxPool1d(nn.modules.pooling.MaxPool1d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['return_indices'] = return_indices
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.MaxPool1d.__init__(self, **self.param_dict)
    
    
class MaxPool2d(nn.modules.pooling.MaxPool2d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['return_indices'] = return_indices
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.MaxPool2d.__init__(self, **self.param_dict)
    
    
class MaxPool3d(nn.modules.pooling.MaxPool3d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['return_indices'] = return_indices
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.MaxPool3d.__init__(self, **self.param_dict)
    
    
class MaxUnpool1d(nn.modules.pooling.MaxUnpool1d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.MaxUnpool1d.__init__(self, **self.param_dict)
    
    
class MaxUnpool2d(nn.modules.pooling.MaxUnpool2d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.MaxUnpool2d.__init__(self, **self.param_dict)
    
    
class MaxUnpool3d(nn.modules.pooling.MaxUnpool3d, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling.MaxUnpool3d.__init__(self, **self.param_dict)
    
    
class _AdaptiveAvgPoolNd(nn.modules.pooling._AdaptiveAvgPoolNd, FateTorch):
        
    def __init__(self, output_size, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling._AdaptiveAvgPoolNd.__init__(self, **self.param_dict)
    
    
class _AdaptiveMaxPoolNd(nn.modules.pooling._AdaptiveMaxPoolNd, FateTorch):
        
    def __init__(self, output_size, return_indices=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['return_indices'] = return_indices
        self.param_dict['output_size'] = output_size
        self.param_dict.update(kwargs)
        nn.modules.pooling._AdaptiveMaxPoolNd.__init__(self, **self.param_dict)
    
    
class _AvgPoolNd(nn.modules.pooling._AvgPoolNd, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.pooling._AvgPoolNd.__init__(self, **self.param_dict)
    
    
class _LPPoolNd(nn.modules.pooling._LPPoolNd, FateTorch):
        
    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['norm_type'] = norm_type
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling._LPPoolNd.__init__(self, **self.param_dict)
    
    
class _MaxPoolNd(nn.modules.pooling._MaxPoolNd, FateTorch):
        
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['stride'] = stride
        self.param_dict['padding'] = padding
        self.param_dict['dilation'] = dilation
        self.param_dict['return_indices'] = return_indices
        self.param_dict['ceil_mode'] = ceil_mode
        self.param_dict['kernel_size'] = kernel_size
        self.param_dict.update(kwargs)
        nn.modules.pooling._MaxPoolNd.__init__(self, **self.param_dict)
    
    
class _MaxUnpoolNd(nn.modules.pooling._MaxUnpoolNd, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.pooling._MaxUnpoolNd.__init__(self, **self.param_dict)
    
    
class BatchNorm1d(nn.modules.batchnorm.BatchNorm1d, FateTorch):
        
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_features'] = num_features
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.BatchNorm1d.__init__(self, **self.param_dict)
    
    
class BatchNorm2d(nn.modules.batchnorm.BatchNorm2d, FateTorch):
        
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_features'] = num_features
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.BatchNorm2d.__init__(self, **self.param_dict)
    
    
class BatchNorm3d(nn.modules.batchnorm.BatchNorm3d, FateTorch):
        
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_features'] = num_features
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.BatchNorm3d.__init__(self, **self.param_dict)
    
    
class LazyBatchNorm1d(nn.modules.batchnorm.LazyBatchNorm1d, FateTorch):
        
    def __init__(self, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.LazyBatchNorm1d.__init__(self, **self.param_dict)
    
    
class LazyBatchNorm2d(nn.modules.batchnorm.LazyBatchNorm2d, FateTorch):
        
    def __init__(self, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.LazyBatchNorm2d.__init__(self, **self.param_dict)
    
    
class LazyBatchNorm3d(nn.modules.batchnorm.LazyBatchNorm3d, FateTorch):
        
    def __init__(self, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.LazyBatchNorm3d.__init__(self, **self.param_dict)
    
    
class SyncBatchNorm(nn.modules.batchnorm.SyncBatchNorm, FateTorch):
        
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['process_group'] = process_group
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_features'] = num_features
        self.param_dict.update(kwargs)
        nn.modules.batchnorm.SyncBatchNorm.__init__(self, **self.param_dict)
    
    
class _BatchNorm(nn.modules.batchnorm._BatchNorm, FateTorch):
        
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_features'] = num_features
        self.param_dict.update(kwargs)
        nn.modules.batchnorm._BatchNorm.__init__(self, **self.param_dict)
    
    
class _LazyNormBase(nn.modules.batchnorm._LazyNormBase, FateTorch):
        
    def __init__(self, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict.update(kwargs)
        nn.modules.batchnorm._LazyNormBase.__init__(self, **self.param_dict)
    
    
class _NormBase(nn.modules.batchnorm._NormBase, FateTorch):
        
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['eps'] = eps
        self.param_dict['momentum'] = momentum
        self.param_dict['affine'] = affine
        self.param_dict['track_running_stats'] = track_running_stats
        self.param_dict['device'] = device
        self.param_dict['dtype'] = dtype
        self.param_dict['num_features'] = num_features
        self.param_dict.update(kwargs)
        nn.modules.batchnorm._NormBase.__init__(self, **self.param_dict)
    
    
class ConstantPad1d(nn.modules.padding.ConstantPad1d, FateTorch):
        
    def __init__(self, padding, value, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict['value'] = value
        self.param_dict.update(kwargs)
        nn.modules.padding.ConstantPad1d.__init__(self, **self.param_dict)
    
    
class ConstantPad2d(nn.modules.padding.ConstantPad2d, FateTorch):
        
    def __init__(self, padding, value, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict['value'] = value
        self.param_dict.update(kwargs)
        nn.modules.padding.ConstantPad2d.__init__(self, **self.param_dict)
    
    
class ConstantPad3d(nn.modules.padding.ConstantPad3d, FateTorch):
        
    def __init__(self, padding, value, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict['value'] = value
        self.param_dict.update(kwargs)
        nn.modules.padding.ConstantPad3d.__init__(self, **self.param_dict)
    
    
class ReflectionPad1d(nn.modules.padding.ReflectionPad1d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ReflectionPad1d.__init__(self, **self.param_dict)
    
    
class ReflectionPad2d(nn.modules.padding.ReflectionPad2d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ReflectionPad2d.__init__(self, **self.param_dict)
    
    
class ReflectionPad3d(nn.modules.padding.ReflectionPad3d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ReflectionPad3d.__init__(self, **self.param_dict)
    
    
class ReplicationPad1d(nn.modules.padding.ReplicationPad1d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ReplicationPad1d.__init__(self, **self.param_dict)
    
    
class ReplicationPad2d(nn.modules.padding.ReplicationPad2d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ReplicationPad2d.__init__(self, **self.param_dict)
    
    
class ReplicationPad3d(nn.modules.padding.ReplicationPad3d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ReplicationPad3d.__init__(self, **self.param_dict)
    
    
class ZeroPad2d(nn.modules.padding.ZeroPad2d, FateTorch):
        
    def __init__(self, padding, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['padding'] = padding
        self.param_dict.update(kwargs)
        nn.modules.padding.ZeroPad2d.__init__(self, **self.param_dict)
    
    
class _ConstantPadNd(nn.modules.padding._ConstantPadNd, FateTorch):
        
    def __init__(self, value, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['value'] = value
        self.param_dict.update(kwargs)
        nn.modules.padding._ConstantPadNd.__init__(self, **self.param_dict)
    
    
class _ReflectionPadNd(nn.modules.padding._ReflectionPadNd, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.padding._ReflectionPadNd.__init__(self, **self.param_dict)
    
    
class _ReplicationPadNd(nn.modules.padding._ReplicationPadNd, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.padding._ReplicationPadNd.__init__(self, **self.param_dict)
    
    
class BCELoss(nn.modules.loss.BCELoss, FateTorch):
        
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.BCELoss.__init__(self, **self.param_dict)
    
    
class BCEWithLogitsLoss(nn.modules.loss.BCEWithLogitsLoss, FateTorch):
        
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict['pos_weight'] = pos_weight
        self.param_dict.update(kwargs)
        nn.modules.loss.BCEWithLogitsLoss.__init__(self, **self.param_dict)
    
    
class CTCLoss(nn.modules.loss.CTCLoss, FateTorch):
        
    def __init__(self, blank=0, reduction='mean', zero_infinity=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['blank'] = blank
        self.param_dict['reduction'] = reduction
        self.param_dict['zero_infinity'] = zero_infinity
        self.param_dict.update(kwargs)
        nn.modules.loss.CTCLoss.__init__(self, **self.param_dict)
    
    
class CosineEmbeddingLoss(nn.modules.loss.CosineEmbeddingLoss, FateTorch):
        
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['margin'] = margin
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.CosineEmbeddingLoss.__init__(self, **self.param_dict)
    
    
class CrossEntropyLoss(nn.modules.loss.CrossEntropyLoss, FateTorch):
        
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['ignore_index'] = ignore_index
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict['label_smoothing'] = label_smoothing
        self.param_dict.update(kwargs)
        nn.modules.loss.CrossEntropyLoss.__init__(self, **self.param_dict)
    
    
class GaussianNLLLoss(nn.modules.loss.GaussianNLLLoss, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.loss.GaussianNLLLoss.__init__(self, **self.param_dict)
    
    
class HingeEmbeddingLoss(nn.modules.loss.HingeEmbeddingLoss, FateTorch):
        
    def __init__(self, margin=1.0, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['margin'] = margin
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.HingeEmbeddingLoss.__init__(self, **self.param_dict)
    
    
class HuberLoss(nn.modules.loss.HuberLoss, FateTorch):
        
    def __init__(self, reduction='mean', delta=1.0, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['reduction'] = reduction
        self.param_dict['delta'] = delta
        self.param_dict.update(kwargs)
        nn.modules.loss.HuberLoss.__init__(self, **self.param_dict)
    
    
class KLDivLoss(nn.modules.loss.KLDivLoss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', log_target=False, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict['log_target'] = log_target
        self.param_dict.update(kwargs)
        nn.modules.loss.KLDivLoss.__init__(self, **self.param_dict)
    
    
class L1Loss(nn.modules.loss.L1Loss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.L1Loss.__init__(self, **self.param_dict)
    
    
class MSELoss(nn.modules.loss.MSELoss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.MSELoss.__init__(self, **self.param_dict)
    
    
class MarginRankingLoss(nn.modules.loss.MarginRankingLoss, FateTorch):
        
    def __init__(self, margin=0.0, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['margin'] = margin
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.MarginRankingLoss.__init__(self, **self.param_dict)
    
    
class MultiLabelMarginLoss(nn.modules.loss.MultiLabelMarginLoss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.MultiLabelMarginLoss.__init__(self, **self.param_dict)
    
    
class MultiLabelSoftMarginLoss(nn.modules.loss.MultiLabelSoftMarginLoss, FateTorch):
        
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.MultiLabelSoftMarginLoss.__init__(self, **self.param_dict)
    
    
class MultiMarginLoss(nn.modules.loss.MultiMarginLoss, FateTorch):
        
    def __init__(self, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['p'] = p
        self.param_dict['margin'] = margin
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.MultiMarginLoss.__init__(self, **self.param_dict)
    
    
class NLLLoss(nn.modules.loss.NLLLoss, FateTorch):
        
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['ignore_index'] = ignore_index
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.NLLLoss.__init__(self, **self.param_dict)
    
    
class NLLLoss2d(nn.modules.loss.NLLLoss2d, FateTorch):
        
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['ignore_index'] = ignore_index
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.NLLLoss2d.__init__(self, **self.param_dict)
    
    
class PoissonNLLLoss(nn.modules.loss.PoissonNLLLoss, FateTorch):
        
    def __init__(self, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['log_input'] = log_input
        self.param_dict['full'] = full
        self.param_dict['size_average'] = size_average
        self.param_dict['eps'] = eps
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.PoissonNLLLoss.__init__(self, **self.param_dict)
    
    
class SmoothL1Loss(nn.modules.loss.SmoothL1Loss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', beta=1.0, **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict['beta'] = beta
        self.param_dict.update(kwargs)
        nn.modules.loss.SmoothL1Loss.__init__(self, **self.param_dict)
    
    
class SoftMarginLoss(nn.modules.loss.SoftMarginLoss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.SoftMarginLoss.__init__(self, **self.param_dict)
    
    
class TripletMarginLoss(nn.modules.loss.TripletMarginLoss, FateTorch):
        
    def __init__(self, margin=1.0, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['margin'] = margin
        self.param_dict['p'] = p
        self.param_dict['eps'] = eps
        self.param_dict['swap'] = swap
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss.TripletMarginLoss.__init__(self, **self.param_dict)
    
    
class TripletMarginWithDistanceLoss(nn.modules.loss.TripletMarginWithDistanceLoss, FateTorch):
        
    def __init__(self, **kwargs):
        FateTorch.__init__(self)
        self.param_dict.update(kwargs)
        nn.modules.loss.TripletMarginWithDistanceLoss.__init__(self, **self.param_dict)
    
    
class _Loss(nn.modules.loss._Loss, FateTorch):
        
    def __init__(self, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss._Loss.__init__(self, **self.param_dict)
    
    
class _WeightedLoss(nn.modules.loss._WeightedLoss, FateTorch):
        
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', **kwargs):
        FateTorch.__init__(self)
        self.param_dict['weight'] = weight
        self.param_dict['size_average'] = size_average
        self.param_dict['reduce'] = reduce
        self.param_dict['reduction'] = reduction
        self.param_dict.update(kwargs)
        nn.modules.loss._WeightedLoss.__init__(self, **self.param_dict)
    
    