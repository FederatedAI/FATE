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

import copy

from federatedml.model_base import ModelBase
from federatedml.feature.instance import Instance
from federatedml.param.feature_transform_param import FeatureTransformParam
from federatedml.protobuf.generated import feature_transform_meta_pb2, feature_transform_param_pb2
from federatedml.util import LOGGER


class FeatureTransform(ModelBase):
    def __init__(self):
        super(FeatureTransform, self).__init__()
        self.model_param = FeatureTransformParam()
        self.rules = None
        self.need_run = None

        self.header = None

        self.model_param_name = 'FeatureTransformParam'
        self.model_meta_name = 'FeatureTransformMeta'

    def _init_model(self, params):
        self.model_param = params
        self.rules = params.rules
        self.need_run = params.need_run

    def _get_meta(self):
        meta = feature_transform_meta_pb2.FeatureTransformMeta (
            rules=self.rules,
            need_run=self.need_run
        )
        return meta

    def _get_param(self):
        param = feature_transform_param_pb2.FeatureTransformParam(header=self.header)
        return param

    def export_model(self):
        LOGGER.info(f"Enter Feature Transform export_model")
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        self.model_output = result
        LOGGER.info(f"Finish Feature Transform export_model")
        return None

    def load_model(self, model_dict):
        LOGGER.info(f"Enter Feature Transform load_model")
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        param_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)

        self.rules = meta_obj.rules
        self.need_run = meta_obj.need_run        
        self.header = param_obj.header
        LOGGER.info(f"Finish Feature Transform load_model")
        return None

    def fit(self, data):
        LOGGER.info(f"Enter Feature Transform fit")
        self.header = data.schema["header"]
        new_data = data.map(lambda k, v: FeatureTransform._apply(k, v, self.rules, self.header))
        new_data.schema = copy.deepcopy(data.schema)
        new_data.schema['header'] = [rule['dst_name'] for rule in self.rules]
        LOGGER.info(f"Finish Feature Transform fit")
        return new_data

    def transform(self, data):
        LOGGER.info(f"Enter Feature Transform transform")
        self.header = data.schema["header"]
        new_data = data.map(lambda k, v: FeatureTransform._apply(k, v, self.rules, self.header))
        new_data.schema = copy.deepcopy(data.schema)
        new_data.schema['header'] = [rule['dst_name'] for rule in self.rules]
        LOGGER.info(f"Finish Feature Transform transform")
        return new_data

    @staticmethod
    def _apply(k, v, rules, headers):
        features = FeatureTransform._rule(rules, headers, v.features)
        instance = Instance(inst_id=v.inst_id, weight=v.weight, features=features, label=v.label)
        return (k, instance)

    @staticmethod
    def _rule(rules, headers, features):
        results = []

        for rule in rules:
            src_name = rule['src_name']
            idx = headers.index(src_name)
            feature_val = features[idx]
            results.extend(FeatureTransform._transform(rule['transforms'], feature_val))
        
        return results        

    @staticmethod
    def _transform(transforms, val):
        results = []

        for transform in transforms:
            if FeatureTransform._op(transform['ops'], val):
                results.append(transform['label'])  

        return results

    @staticmethod
    def _op(ops, val):
        results = []

        for op in ops:
            op_name = op['name']
            op_val = op['val']
            op_result = FeatureTransform._operator(val, op_name, op_val)
            results.append(op_result)
            if not op_result:
                break

        return (False not in results)

    @staticmethod
    def _operator(a, op, b):
        if op == 'gt':
            return a > b
        elif op == 'gte':
            return a >= b 
        elif op == 'lt':
            return a < b
        elif op == 'lte':
            return a <= b
        elif op == 'contains':
            return (a.find(b) >= 0)
        elif op == 'startswith':
            return a.startswith(b)
        elif op == 'endswith':
            return a.endswith(b)
        else:
            return a == b
