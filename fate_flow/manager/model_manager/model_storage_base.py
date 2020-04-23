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
import abc


class ModelStorageBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def store(self, model_id: str, model_version: str, store_address: dict):
        """
        Store the model from local cache to a reliable system
        :param model_id:
        :param model_version:
        :param store_address:
        :return:
        """
        raise Exception("Subclasses must implement this function")

    @abc.abstractmethod
    def restore(self, model_id: str, model_version: str, store_address: dict):
        """
        Restore model from storage system to local cache
        :param model_id:
        :param model_version:
        :param store_address:
        :return:
        """
        raise Exception("Subclasses must implement this function")
