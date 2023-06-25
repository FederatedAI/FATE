import torch.nn as nn
from federatedml.nn.model_zoo.ipr.sign_block import SignatureConv, ConvBlock


class SignAlexNet(nn.Module):

    """
    This is a modified Alexnet: its 4,5,6 layers are replaced by Singnature Conv Block
    """

    def __init__(self, num_classes):
        super().__init__()
        in_channels = 3
        maxpoolidx = [1, 3, 7]
        signed_layer = [4, 5, 6]
        layers = []
        inp = in_channels

        # channels & kennel size
        # the same setting as the FedIPR paper
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }

        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                if layeridx in signed_layer:
                    layers.append(SignatureConv(inp, oups[layeridx], k, 1, p))
                else:
                    layers.append(ConvBlock(inp, oups[layeridx], k, 1, p))
                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        for m in self.features:
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
