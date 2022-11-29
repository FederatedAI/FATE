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


from torch.utils.data import DataLoader

from federatedml.framework.hetero.procedure import batch_generator
from federatedml.nn.hetero.base import HeteroNNBase
from federatedml.nn.hetero.model import HeteroNNHostModel
from federatedml.param.hetero_nn_param import HeteroNNParam as NNParameter
from federatedml.protobuf.generated.hetero_nn_model_meta_pb2 import HeteroNNMeta
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import HeteroNNParam
from federatedml.util import consts, LOGGER

MODELMETA = "HeteroNNHostMeta"
MODELPARAM = "HeteroNNHostParam"


class HeteroNNHost(HeteroNNBase):

    def __init__(self):
        super(HeteroNNHost, self).__init__()

        self.batch_generator = batch_generator.Host()
        self.model = None
        self.role = consts.HOST
        self.input_shape = None
        self.default_table_partitions = 4

    def _init_model(self, hetero_nn_param):
        super(HeteroNNHost, self)._init_model(hetero_nn_param)

    def export_model(self):

        if self.need_cv:
            return None

        model = {MODELMETA: self._get_model_meta(),
                 MODELPARAM: self._get_model_param()}

        return model

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        param = model_dict.get(MODELPARAM)
        meta = model_dict.get(MODELMETA)
        if self.hetero_nn_param is None:
            self.hetero_nn_param = NNParameter()
            self.hetero_nn_param.check()
            self.predict_param = self.hetero_nn_param.predict_param
        self._build_model()
        self._restore_model_meta(meta)
        self._restore_model_param(param)

    def _build_model(self):
        self.model = HeteroNNHostModel(self.hetero_nn_param, self.flowid)
        self.model.set_transfer_variable(self.transfer_variable)
        self.model.set_partition(self.default_table_partitions)

    def predict(self, data_inst):

        ds = self.prepare_dataset(data_inst, data_type='predict')
        batch_size = len(ds) if self.batch_size == -1 else self.batch_size
        for batch_data in DataLoader(ds, batch_size=batch_size):
            # ignore label if the dataset offers label
            if isinstance(batch_data, tuple) and len(batch_data) > 1:
                batch_data = batch_data[0]
            self.model.predict(batch_data)

    def fit(self, data_inst, validate_data=None):

        if hasattr(
                data_inst,
                'partitions') and data_inst.partitions is not None:
            self.default_table_partitions = data_inst.partitions
            LOGGER.debug(
                'reset default partitions is {}'.format(
                    self.default_table_partitions))

        train_ds = self.prepare_dataset(data_inst, data_type='train')
        if validate_data is not None:
            val_ds = self.prepare_dataset(validate_data, data_type='validate')
        else:
            val_ds = None

        self.callback_list.on_train_begin(train_ds, val_ds)

        if not self.component_properties.is_warm_start:
            self._build_model()
            epoch_offset = 0
        else:
            self.callback_warm_start_init_iter(self.history_iter_epoch)
            epoch_offset = self.history_iter_epoch + 1

        batch_size = len(train_ds) if self.batch_size == - \
            1 else self.batch_size

        for cur_epoch in range(epoch_offset, epoch_offset + self.epochs):
            self.iter_epoch = cur_epoch
            for batch_idx, batch_data in enumerate(
                    DataLoader(train_ds, batch_size=batch_size)):
                self.model.train(batch_data, cur_epoch, batch_idx)

            self.callback_list.on_epoch_end(cur_epoch)
            if self.callback_variables.stop_training:
                LOGGER.debug('early stopping triggered')
                break
            is_converge = self.transfer_variable.is_converge.get(
                idx=0, suffix=(cur_epoch,))
            if is_converge:
                LOGGER.debug(
                    "Training process is converged in epoch {}".format(cur_epoch))
                break

        self.callback_list.on_train_end()

    def _get_model_meta(self):
        model_meta = HeteroNNMeta()
        model_meta.batch_size = self.batch_size
        model_meta.hetero_nn_model_meta.CopyFrom(
            self.model.get_hetero_nn_model_meta())
        model_meta.module = 'HeteroNN'
        return model_meta

    def _get_model_param(self):
        model_param = HeteroNNParam()
        model_param.iter_epoch = self.iter_epoch
        model_param.header.extend(self._header)
        model_param.hetero_nn_model_param.CopyFrom(
            self.model.get_hetero_nn_model_param())
        model_param.best_iteration = self.callback_variables.best_iteration

        return model_param
