#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from fate_arch.computing.non_distributed import LocalData
from federatedml.model_base import ModelBase
from federatedml.model_selection import start_cross_validation
from federatedml.nn.backend.utils.data import load_dataset
from federatedml.nn.dataset.base import Dataset, ShuffleWrapDataset
from federatedml.param.hetero_nn_param import HeteroNNParam
from federatedml.transfer_variable.transfer_class.hetero_nn_transfer_variable import HeteroNNTransferVariable
from federatedml.util import consts


class HeteroNNBase(ModelBase):

    def __init__(self):
        super(HeteroNNBase, self).__init__()

        self.tol = None
        self.early_stop = None
        self.seed = 100
        self.epochs = None
        self.batch_size = None
        self._header = []

        self.predict_param = None
        self.hetero_nn_param = None

        self.batch_generator = None
        self.model = None

        self.partition = None
        self.validation_freqs = None
        self.early_stopping_rounds = None
        self.metrics = []
        self.use_first_metric_only = False

        self.transfer_variable = HeteroNNTransferVariable()
        self.model_param = HeteroNNParam()
        self.mode = consts.HETERO

        self.selector_param = None
        self.floating_point_precision = None

        self.history_iter_epoch = 0
        self.iter_epoch = 0

        self.data_x = []
        self.data_y = []
        self.dataset_cache_dict = {}

        self.label_num = None

        # nn related param
        self.top_model_define = None
        self.bottom_model_define = None
        self.interactive_layer_define = None
        self.dataset_shuffle = True
        self.dataset = None
        self.dataset_param = None
        self.dataset_shuffle_seed = 100

    def _init_model(self, hetero_nn_param: HeteroNNParam):

        self.interactive_layer_lr = hetero_nn_param.interactive_layer_lr
        self.epochs = hetero_nn_param.epochs
        self.batch_size = hetero_nn_param.batch_size
        self.seed = hetero_nn_param.seed
        self.early_stop = hetero_nn_param.early_stop
        self.validation_freqs = hetero_nn_param.validation_freqs
        self.early_stopping_rounds = hetero_nn_param.early_stopping_rounds
        self.metrics = hetero_nn_param.metrics
        self.use_first_metric_only = hetero_nn_param.use_first_metric_only

        self.tol = hetero_nn_param.tol

        self.predict_param = hetero_nn_param.predict_param
        self.hetero_nn_param = hetero_nn_param
        self.selector_param = hetero_nn_param.selector_param
        self.floating_point_precision = hetero_nn_param.floating_point_precision

        # nn configs
        self.bottom_model_define = hetero_nn_param.bottom_nn_define
        self.top_model_define = hetero_nn_param.top_nn_define
        self.interactive_layer_define = hetero_nn_param.interactive_layer_define

        # dataset
        dataset_param = hetero_nn_param.dataset.to_dict()
        self.dataset = dataset_param['dataset_name']
        self.dataset_param = dataset_param['param']

    def reset_flowid(self):
        new_flowid = ".".join([self.flowid, "evaluate"])
        self.set_flowid(new_flowid)

    def recovery_flowid(self):
        new_flowid = ".".join(self.flowid.split(".", -1)[: -1])
        self.set_flowid(new_flowid)

    def _build_bottom_model(self):
        pass

    def _build_interactive_model(self):
        pass

    def _restore_model_meta(self, meta):
        # self.hetero_nn_param.interactive_layer_lr = meta.interactive_layer_lr
        self.hetero_nn_param.task_type = meta.task_type
        if not self.component_properties.is_warm_start:
            self.batch_size = meta.batch_size
            self.epochs = meta.epochs
            self.tol = meta.tol
            self.early_stop = meta.early_stop

        self.model.set_hetero_nn_model_meta(meta.hetero_nn_model_meta)

    def _restore_model_param(self, param):
        self.model.set_hetero_nn_model_param(param.hetero_nn_model_param)
        self._header = list(param.header)
        self.history_iter_epoch = param.iter_epoch
        self.iter_epoch = param.iter_epoch

    def set_partition(self, data_inst):
        self.partition = data_inst.partitions
        self.model.set_partition(self.partition)

    def cross_validation(self, data_instances):
        return start_cross_validation.run(self, data_instances)

    def prepare_dataset(self, data, data_type='train', check_label=False):

        # train input & validate input are DTables or path str
        if isinstance(data, LocalData):
            data = data.path

        if isinstance(data, Dataset) or isinstance(data, ShuffleWrapDataset):
            ds = data
        else:
            ds = load_dataset(
                self.dataset,
                data,
                self.dataset_param,
                self.dataset_cache_dict)

            if not ds.has_sample_ids():
                raise ValueError(
                    'Dataset has no sample id, this is not allowed in hetero-nn, please make sure'
                    ' that you implement get_sample_ids()')

            if self.dataset_shuffle:
                ds = ShuffleWrapDataset(
                    ds, shuffle_seed=self.dataset_shuffle_seed)
                if self.role == consts.GUEST:
                    self.transfer_variable.dataset_info.remote(
                        ds.idx_map, idx=-1, suffix=('idx_map', data_type))
                if self.role == consts.HOST:
                    idx_map = self.transfer_variable.dataset_info.get(
                        idx=0, suffix=('idx_map', data_type))
                    assert len(idx_map) == len(ds), 'host dataset len != guest dataset len, please check your dataset,' \
                                                    'guest len {}, host len {}'.format(len(idx_map), len(ds))
                    ds.set_shuffled_idx(idx_map)

            if check_label:
                try:
                    all_classes = ds.get_classes()
                except NotImplementedError as e:
                    raise NotImplementedError(
                        'get_classes() is not implemented, please implement this function'
                        ' when you are using hetero-nn. Let it return classes in a list.'
                        ' Please see built-in dataset(table.py for example) for reference')
                except BaseException as e:
                    raise e

                from federatedml.util import LOGGER
                LOGGER.debug('all classes is {}'.format(all_classes))
                if self.label_num is None:
                    if self.task_type == consts.CLASSIFICATION:
                        self.label_num = len(all_classes)
                    elif self.task_type == consts.REGRESSION:
                        self.label_num = 1

        return ds

    # override function
    @staticmethod
    def set_predict_data_schema(predict_datas, schemas):
        if predict_datas is None:
            return predict_datas
        if isinstance(predict_datas, list):
            predict_data = predict_datas[0]
            schema = schemas[0]
        else:
            predict_data = predict_datas
            schema = schemas
        if predict_data is not None:
            predict_data.schema = {
                "header": [
                    "label",
                    "predict_result",
                    "predict_score",
                    "predict_detail",
                    "type",
                ],
                "sid": 'id',
                "content_type": "predict_result"
            }
            if schema.get("match_id_name") is not None:
                predict_data.schema["match_id_name"] = schema.get(
                    "match_id_name")
        return predict_data
