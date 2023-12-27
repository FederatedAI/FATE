#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from torch import nn


class Multi(nn.Module):
    def __init__(self, feat=18, class_num=4) -> None:
        super().__init__()
        self.class_num = class_num
        self.model = nn.Sequential(nn.Linear(feat, 10), nn.ReLU(), nn.Linear(10, class_num))

    def forward(self, x):
        if self.training:
            return self.model(x)
        else:
            return nn.Softmax(dim=-1)(self.model(x))
