import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from federatedml.util import LOGGER

"""
Base
"""


class SignatureBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    @property
    def embeded_param(self):
        return None
    
    def extract_sign(self, W):
        pass

    def sign_loss(self, W, sign):
        pass

    def embeded_param_num(self):
        pass


def is_sign_block(block):
    return issubclass(type(block), SignatureBlock)


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, relu=True):
        super().__init__()

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias= False)

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def generate_signature(conv_block: SignatureBlock, num_bits):
    
    sign = torch.sign(torch.rand(num_bits) - 0.5)
    W = torch.randn(len(conv_block.embeded_param.flatten()), num_bits)

    return (W, sign)


"""
Function & Class for Conv Layer
"""


class SignatureConv(SignatureBlock):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SignatureConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight = self.conv.weight
        
        self._embed_para_num = None
        self.init_scale()
        self.init_bias()
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def embeded_param_num(self):
        return self._embed_para_num

    def init_bias(self):
        self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
        init.zeros_(self.bias)

    def init_scale(self):
        self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
        init.ones_(self.scale)
        self._embed_para_num = self.scale.shape[0]

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    @property
    def embeded_param(self):
        # embedded in the BatchNorm param, as the same in the paper
        return self.scale

    def extract_sign(self, W):
        # W is the linear weight for extracting signature
        with torch.no_grad():
            return self.scale.view([1, -1]).mm(W).sign().flatten()

    def sign_loss(self, W, sign):
        loss = F.relu(-self.scale.view([1, -1]).mm(W).mul(sign.view(-1))).sum()
        return loss

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x * self.scale[None, :, None, None] + self.bias[None, :, None, None]
        x = self.relu(x)
        return x


"""
Function & Class for LM
"""


def recursive_replace_layernorm(module, layer_name_set=None):

    """
    Recursively replaces the LayerNorm layers of a given module with SignatureLayerNorm layers.
    
    Parameters:
        module (torch.nn.Module): The module in which LayerNorm layers should be replaced.
        layer_name_set (set[str], optional): A set of layer names to be replaced. If None,
                                             all LayerNorm layers in the module will be replaced.
    """
        
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.LayerNorm):
            if layer_name_set is not None and name not in layer_name_set:
                continue
            setattr(module, name, SignatureLayerNorm.from_layer_norm_layer(sub_module))
            LOGGER.debug(f"Replace {name} with SignatureLayerNorm")
        recursive_replace_layernorm(sub_module, layer_name_set)


class SignatureLayerNorm(SignatureBlock):

    def __init__(self, normalized_shape=None, eps=1e-5, elementwise_affine=True, layer_norm_inst=None):
        super(SignatureLayerNorm, self).__init__()
        if layer_norm_inst is not None and isinstance(layer_norm_inst, nn.LayerNorm):
            self.ln = layer_norm_inst
        else:
            self.ln = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

        self._embed_param_num = self.ln.weight.numel()

    @property
    def embeded_param(self):
        return self.ln.weight
    
    def embeded_param_num(self):
        return  self._embed_param_num

    @staticmethod
    def from_layer_norm_layer(layer_norm_layer: nn.LayerNorm):
        return SignatureLayerNorm(layer_norm_inst=layer_norm_layer)

    def extract_sign(self, W):
        # W is the linear weight for extracting signature
        with torch.no_grad():
            return self.ln.weight.view([1, -1]).mm(W).sign().flatten()

    def sign_loss(self, W, sign):
        loss = F.relu(-self.ln.weight.view([1, -1]).mm(W).mul(sign.view(-1))).sum()
        return loss

    def forward(self, x):
        return self.ln(x)


if __name__ == "__main__":
    conv = SignatureConv(3, 384, 3, 1, 1)
    layer_norm = SignatureLayerNorm((768, ))
    layer_norm_2 = SignatureLayerNorm.from_layer_norm_layer(layer_norm.ln)
    