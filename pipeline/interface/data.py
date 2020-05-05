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
#


class Data(object):
    def __init__(self, data=None, train_data=None, validate_data=None, test_data=None):
        self.data = data
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data

    """
    @property
    def train_data(self):
        return self._train_data

    @property
    def validate_data(self):
        return self._validate_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def data(self):
        return self._data
    """


