import torch.nn as nn
import torch.nn.functional as F
from federatedml.nn.model_zoo.ipr.sign_block import ConvBlock, SignatureConv


# The layer define for ResNet18, add signature to last layer
signed_layer_define = {
    'layer1': {
        '0': {'convbnrelu_1': {'flag': False}, 'convbn_2': {'flag': False}},
        '1': {'convbnrelu_1': {'flag': False}, 'convbn_2': {'flag': False}}
    },
    'layer2': {
        '0': {'convbnrelu_1': {'flag': False}, 'convbn_2': {'flag': False}, 'shortcut': {'flag': False}},
        '1': {'convbnrelu_1': {'flag': False}, 'convbn_2': {'flag': False}}
    },
    'layer3': {
        '0': {'convbnrelu_1': {'flag': False}, 'convbn_2': {'flag': False}, 'shortcut': {'flag': False}},
        '1': {'convbnrelu_1': {'flag': False}, 'convbn_2': {'flag': False}}
    },
    'layer4': {
        '0': {'convbnrelu_1': {'flag': True}, 'convbn_2': {'flag': True}, 'shortcut': {'flag': False}},
        '1': {'convbnrelu_1': {'flag': True}, 'convbn_2': {'flag': True}}
    }
}


def get_convblock(passport_kwargs):
    def convblock_(*args, **kwargs):
        if passport_kwargs['flag']:
            return SignatureConv(*args, **kwargs)
        else:
            return ConvBlock(*args, **kwargs)

    return convblock_


class BasicPrivateBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kwargs={}):#(512, 512, 2) (512, 512, 1)
        super(BasicPrivateBlock, self).__init__()

        self.convbnrelu_1 = get_convblock(kwargs['convbnrelu_1'])(in_planes, planes, 3, stride, 1)
        self.convbn_2 = get_convblock(kwargs['convbn_2'])(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_convblock(kwargs['shortcut'])(in_planes, self.expansion * planes, 1, stride, 0) # input, output, kernel_size=1

    def forward(self, x):
        
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)

        if not isinstance(self.shortcut, nn.Sequential):
            out = out + self.shortcut(x)
        else: 
            out = out + x
        out = F.relu(out)
        return out


class SignResnet18(nn.Module):

    def __init__(self, num_classes=100): #BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs

        super(SignResnet18, self).__init__()
        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        block = BasicPrivateBlock
        model_define = signed_layer_define

        self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, model_define=model_define['layer1'])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, model_define=model_define['layer2'])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, model_define=model_define['layer3'])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, model_define=model_define['layer4'])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, model_define): #BasicPrivateBlock, planes = 512, numblocks = 2, stride =2, **model_kwargs
        strides = [stride] + [1] * (num_blocks - 1) # [2] + [1]*1 = [2, 1]
        layers = []
        for i, stride in enumerate(strides): #stride = 2 & 1
            layers.append(block(self.in_planes, planes, stride, model_define[str(i)])) # (512, 512, 2)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
       
        out = self.convbnrelu_1(x)

        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


if __name__ == '__main__':

    net = SignResnet18(num_classes=10)
