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

import numpy as np
from google.protobuf import json_format

from arch.api.utils import log_utils
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.model_base import ModelBase
from federatedml.statistic import data_overview
from federatedml.util import abnormal_detection
from federatedml.framework.weights import Weights
from federatedml.optim.initialize import Initializer
from federatedml.optim.optimizer import optimizer_factory
from federatedml.protobuf.generated import fm_model_param_pb2
from federatedml.model_selection import start_cross_validation
from federatedml.optim.convergence import converge_func_factory
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.util.validation_strategy import ValidationStrategy
from federatedrec.factorization_machine.fm_model_weight import FactorizationMachineWeights

LOGGER = log_utils.getLogger()


class BaseFactorizationMachine(ModelBase):
    def __init__(self):
        super(BaseFactorizationMachine, self).__init__()

        # attribute:
        self.initializer = Initializer()
        self.model_name = 'FactorizationMachine'
        self.model_param_name = 'FactorizationMachineParam'
        self.model_meta_name = 'FactorizationMachineMeta'
        self.n_iter_ = 0
        self.classes_ = None
        self.feature_shape = None
        self.gradient_operator = None
        self.transfer_variable = None
        self.loss_history = []
        self.is_converged = False
        self.header = None
        self.role = ''
        self.mode = ''
        self.schema = {}
        self.cipher_operator = None
        self.model_weights = None
        self.validation_freqs = None

        # one_ve_rest parameter
        self.in_one_vs_rest = False
        self.need_one_vs_rest = None
        self.one_vs_rest_classes = []
        self.one_vs_rest_obj = None

    def _init_model(self, params):
        self.model_param = params
        self.alpha = params.alpha
        self.init_param_obj = params.init_param
        self.fit_intercept = self.init_param_obj.fit_intercept
        self.batch_size = params.batch_size
        self.max_iter = params.max_iter
        self.optimizer = optimizer_factory(params)
        self.converge_func = converge_func_factory(params.early_stop, params.tol)
        self.encrypted_calculator = None
        self.validation_freqs = params.validation_freqs
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=True)

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def set_header(self, header):
        self.header = header

    def get_header(self, data_instances):
        if self.header is not None:
            return self.header
        return data_instances.schema.get("header")

    def _get_meta(self):
        raise NotImplementedError("This method should be be called here")

    def export_model(self):
        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def callback_loss(self, iter_num, loss):
        metric_meta = MetricMeta(name='train',
                                 metric_type="LOSS",
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='loss', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='loss',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, loss)])

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def init_validation_strategy(self, train_data=None, validate_data=None):
        validation_strategy = ValidationStrategy(self.role, self.mode, self.validation_freqs)
        validation_strategy.set_train_data(train_data)
        validation_strategy.set_validate_data(validate_data)
        return validation_strategy

    def cross_validation(self, data_instances):
        return start_cross_validation.run(self, data_instances)

    def _get_cv_param(self):
        self.model_param.cv_param.role = self.role
        self.model_param.cv_param.mode = self.mode
        return self.model_param.cv_param

    def set_schema(self, data_instance, header=None):
        if header is None:
            self.schema["header"] = self.header
        else:
            self.schema["header"] = header
        data_instance.schema = self.schema
        return data_instance

    def init_schema(self, data_instance):
        if data_instance is None:
            return
        self.schema = data_instance.schema
        self.header = self.schema.get('header')

    def get_single_model_param(self):
        weight_dict = {}
        embed_dict = {}
        LOGGER.debug("in get_single_model_param, model_weights: {}, coef: {}, header: {}".format(
            self.model_weights.unboxed, self.model_weights.coef_, self.header
        ))
        for idx, header_name in enumerate(self.header):
            coef_i = self.model_weights.coef_[idx]
            weight_dict[header_name] = coef_i
            embed_i = self.model_weights.embed_[idx].tolist()
            embed = fm_model_param_pb2.Embedding(weight=embed_i)
            embed_dict[header_name] = embed

        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  'weight': weight_dict,
                  'embedding': embed_dict,
                  'embed_size': self.model_weights.embed_shape[1],
                  'intercept': self.model_weights.intercept_,
                  'header': self.header
                  }
        return result

    def _get_param(self):
        header = self.header
        LOGGER.debug("In get_param, header: {}".format(header))
        if header is None:
            param_protobuf_obj = fm_model_param_pb2.FMModelParam()
            return param_protobuf_obj
        if self.need_one_vs_rest:
            # one_vs_rest_class = list(map(str, self.one_vs_rest_obj.classes))
            one_vs_rest_result = self.one_vs_rest_obj.save(fm_model_param_pb2.SingleModel)
            single_result = {'header': header, 'need_one_vs_rest': True}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()
            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result
        LOGGER.debug("in _get_param, single_result: {}".format(single_result))

        param_protobuf_obj = fm_model_param_pb2.FMModelParam(**single_result)
        json_result = json_format.MessageToJson(param_protobuf_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return param_protobuf_obj

    def load_model(self, model_dict):
        LOGGER.debug("Start Loading model")
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        self.fit_intercept = meta_obj.fit_intercept
        self.header = list(result_obj.header)
        # For hetero-fm arbiter predict function
        if self.header is None:
            return

        need_one_vs_rest = result_obj.need_one_vs_rest
        LOGGER.debug("in _load_model need_one_vs_rest: {}".format(need_one_vs_rest))
        if need_one_vs_rest:
            one_vs_rest_result = result_obj.one_vs_rest_result
            self.one_vs_rest_obj = one_vs_rest_factory(classifier=self, role=self.role,
                                                       mode=self.mode, has_arbiter=True)
            self.one_vs_rest_obj.load_model(one_vs_rest_result)
            self.need_one_vs_rest = True
        else:
            self.load_single_model(result_obj)
            self.need_one_vs_rest = False

    def load_single_model(self, single_model_obj):
        LOGGER.info("It's a binary task, start to load single model")
        feature_shape = len(self.header)
        weight_dict = dict(single_model_obj.weight)

        coef_ = np.zeros(feature_shape)
        for idx, header_name in enumerate(self.header):
            coef_[idx] = weight_dict.get(header_name)

        embed_ = np.zeros((feature_shape, single_model_obj.embed_size))
        for idx, header_name in enumerate(self.header):
            i_embed = np.array(single_model_obj.embedding[header_name].weight)
            embed_[idx] = i_embed

        intercept_ = 0.0
        if self.fit_intercept:
            intercept_ = single_model_obj.intercept

        self.model_weights = \
            FactorizationMachineWeights(coef_, embed_, intercept_,
                                        fit_intercept=self.fit_intercept)
        return self

    def one_vs_rest_fit(self, train_data=None, validate_data=None):
        LOGGER.debug("Class num larger than 2, need to do one_vs_rest")
        self.one_vs_rest_obj.fit(data_instances=train_data, validate_data=validate_data)

    def one_vs_rest_predict(self, validate_data):
        if not self.one_vs_rest_obj:
            LOGGER.warning("Not one_vs_rest fit before, return now")
            return
        return self.one_vs_rest_obj.predict(data_instances=validate_data)

