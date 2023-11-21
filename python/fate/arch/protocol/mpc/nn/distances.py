#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from .loss import _Loss
from .module import Module


class CosineSimilarity(Module):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Not used in CrypTen
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
        - Output: :math:`(\ast_1, \ast_2)`
    Examples::
        >>> input1 = crypten.randn(100, 128)
        >>> input2 = crypten.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """

    def __init__(self, dim=1, eps=1e-8):
        super(CosineSimilarity, self).__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return x1.cosine_similarity(x2, self.dim)

    # Remove need to call module.encrypt()
    __getattribute__ = _Loss.__getattribute__
