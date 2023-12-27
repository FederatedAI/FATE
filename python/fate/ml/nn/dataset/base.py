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

from torch.utils.data import Dataset as Dataset_
import abc
import pandas as pd


class Dataset(Dataset_):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()

    # Function to implemented
    @abc.abstractmethod
    def load(self, data_or_path):
        raise NotImplementedError(
            "You must implement load function so that Client can pass file-path to this " "class"
        )

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def has_label(self) -> bool:
        pass

    def get_classes(self) -> list:
        pass

    def get_match_ids(self) -> pd.DataFrame:
        pass

    def get_sample_ids(self) -> pd.DataFrame:
        pass

    def get_sample_id_name(self) -> str:
        pass

    def get_match_id_name(self) -> str:
        pass
