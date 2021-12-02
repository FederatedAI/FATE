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

import operator
import copy
import numpy as np

from federatedml.framework.hetero.procedure import batch_generator
from federatedml.framework.hetero.procedure.hetero_sshe_linear_model import HeteroSSHEGuestBase
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.coordinated_linear_model.poisson_regression.\
    base_poisson_regression import BasePoissonRegression
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.param.hetero_sshe_poisson_param import HeteroSSHEPoissonParam
from federatedml.protobuf.generated import poisson_model_param_pb2, poisson_model_meta_pb2
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_table, fixedpoint_numpy
from federatedml.transfer_variable.transfer_class.batch_generator_transfer_variable import \
    BatchGeneratorTransferVariable
from federatedml.util import consts, fate_operator, LOGGER
from federatedml.util.io_check import assert_io_num_rows_equal


class HeteroPoissonGuest(HeteroSSHEGuestBase):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroPoissonRegression'
        self.model_param_name = 'HeteroPoissonRegressionParam'
        self.model_meta_name = 'HeteroPoissonRegressionMeta'
        self.model_param = HeteroSSHEPoissonParam()
        self.labels = None
        self.labels_original = None
        # self.label_type = int
        self.exposure_index = -1
        self.mu_self = None

    def _init_model(self, params):
        super()._init_model(params)
        self.exposure_colname = params.exposure_colname

    def forward(self, weights, features, suffix, cipher, offset=None, log_offset=None):
        # self._cal_z(weights, features, suffix, cipher)
        if not self.reveal_every_iter:
            LOGGER.info(f"[forward]: Calculate z in share...")
            w_self, w_remote = weights
            z = self._cal_z_in_share(w_self, w_remote, features, suffix, self.cipher)
            w_self_value = self.fixedpoint_encoder.decode(w_self.value)
            mu_self = self.fixedpoint_encoder.decode(features.value).mapValues(lambda x: np.array([np.exp(x.dot(w_self_value))]))
        else:
            LOGGER.info(f"[forward]: Calculate z directly...")
            w = weights.unboxed
            z = features.dot_local(w)
            mu_self = self.fixedpoint_encoder.decode(features.value).mapValues(lambda x: np.array([np.exp(x.dot(w))]))

        mu_self = mu_self.join(offset, lambda x, y: x * y)
        self.mu_self = fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(mu_self),
                                                    q_field=self.fixedpoint_encoder.n,
                                                    endec=self.fixedpoint_encoder)
        remote_z = self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                                 is_remote=False,
                                                                 cipher=None,
                                                                 z=None)[0]

        self.wx_self = z
        self.wx_remote = remote_z
        complete_z = self.wx_self + self.wx_remote + log_offset

        self.encrypted_wx = complete_z
        self.encrypted_error = complete_z - self.labels

        tensor_name = ".".join(("complete_z",) + suffix)
        shared_z = SecureMatrix.from_source(tensor_name,
                                            complete_z,
                                            cipher,
                                            self.fixedpoint_encoder.n,
                                            self.fixedpoint_encoder)
        return shared_z

    def compute_loss(self, weights, suffix, cipher=None, batch_offset=None):
        """
         Compute hetero poisson loss with log link:
            log loss = (1/N) * \sum(exp(wx) - y * log(exp(wx) * exposure))
            loss = (1/N) * \sum(exp(wx_g) * exp(wx_h) - y(wx_g + wx_h) - y * log(exposure))
            loss = (1/N) * \sum(mu_g * mu_h - y(wx_g + wx_h) - y * offset_log)
        """
        LOGGER.info(f"[compute_loss]: Calculate loss ...")
        wx_complete = (self.wx_remote + self.wx_self) * -1
        wxy = (wx_complete * self.labels_original * -1).reduce(operator.add)

        mu_remote = self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                                  is_remote=False,
                                                                  cipher=None,
                                                                  mu_self=None)[0]
        mu_complete = (mu_remote * self.mu_self).reduce(operator.add)

        offset = 0
        if batch_offset:
            offset = (self.labels_original.join(batch_offset, lambda x, y: -x * y)).reduce(operator.add)

        loss = wxy + mu_complete + offset

        batch_num = self.batch_num[int(suffix[2])]
        loss = loss * (-1 / batch_num)

        tensor_name = ".".join(("shared_loss",) + suffix)
        share_loss = SecureMatrix.from_source(tensor_name=tensor_name,
                                              source=loss,
                                              cipher=None,
                                              q_field=self.fixedpoint_encoder.n,
                                              encoder=self.fixedpoint_encoder)

        tensor_name = ".".join(("loss",) + suffix)
        loss = share_loss.get(tensor_name=tensor_name,
                              broadcast=False)[0]

        if self.reveal_every_iter:
            loss_norm = self.optimizer.loss_norm(weights)
            if loss_norm:
                loss += loss_norm
        else:
            if self.optimizer.penalty == consts.L2_PENALTY:
                w_self, w_remote = weights

                w_encode = np.hstack((w_remote.value, w_self.value))

                w_encode = np.array([w_encode])

                w_tensor_name = ".".join(("loss_norm_w",) + suffix)
                w_tensor = fixedpoint_numpy.FixedPointTensor(value=w_encode,
                                                             q_field=self.fixedpoint_encoder.n,
                                                             endec=self.fixedpoint_encoder,
                                                             tensor_name=w_tensor_name)

                w_tensor_transpose_name = ".".join(("loss_norm_w_transpose",) + suffix)
                w_tensor_transpose = fixedpoint_numpy.FixedPointTensor(value=w_encode.T,
                                                                       q_field=self.fixedpoint_encoder.n,
                                                                       endec=self.fixedpoint_encoder,
                                                                       tensor_name=w_tensor_transpose_name)

                loss_norm_tensor_name = ".".join(("loss_norm",) + suffix)

                loss_norm = w_tensor.dot(w_tensor_transpose, target_name=loss_norm_tensor_name).get(broadcast=False)
                loss_norm = 0.5 * self.optimizer.alpha * loss_norm[0][0]
                loss = loss + loss_norm

        LOGGER.info(f"[compute_loss]: loss={loss}, reveal_every_iter={self.reveal_every_iter}")

        return loss

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Prediction of poisson
        Parameters
        ----------
        data_instances: Table of Instance, input data

        Returns
        ----------
        Table
            include input data label, predict result, predicted label
        """
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        header = data_instances.schema.get("header")
        self.exposure_index = BasePoissonRegression.get_exposure_index(header, self.exposure_colname)
        exposure_index = self.exposure_index

        # OK
        exposure = data_instances.mapValues(lambda v: BasePoissonRegression.load_exposure(v, exposure_index))
        data_instances = self.align_data_header(data_instances, self.header)

        LOGGER.debug(
            f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")

        pred_res = BasePoissonRegression.compute_mu(data_instances, self.model_weights.coef_, self.model_weights.intercept_, exposure)

        host_preds = self.transfer_variable.host_prob.get(idx=-1)

        LOGGER.info("Get probability from Host")

        for host_pred in host_preds:
            if not self.is_respectively_reveal:
                host_pred = self.cipher.distribute_decrypt(host_pred)
            pred_res = pred_res.join(host_pred, lambda g, h: g * h)
        predict_result = self.predict_score_to_output(data_instances=data_instances,
                                                      predict_score=pred_res,
                                                      classes=None)

        return predict_result

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = poisson_model_param_pb2.PoissonModelParam()
            return param_protobuf_obj

        single_result = self.get_single_model_param()
        param_protobuf_obj = poisson_model_param_pb2.PoissonModelParam(**single_result)
        return param_protobuf_obj

    def _get_meta(self):
        meta_protobuf_obj = poisson_model_meta_pb2.PoissonModelMeta(penalty=self.model_param.penalty,
                                                              tol=self.model_param.tol,
                                                              alpha=self.alpha,
                                                              optimizer=self.model_param.optimizer,
                                                              batch_size=self.batch_size,
                                                              learning_rate=self.model_param.learning_rate,
                                                              max_iter=self.max_iter,
                                                              early_stop=self.model_param.early_stop,
                                                              fit_intercept=self.fit_intercept,
                                                              reveal_strategy=self.model_param.reveal_strategy)
        return meta_protobuf_obj

    def load_model(self, model_dict):
        result_obj, _ = super().load_model(model_dict)
        self.load_single_model(result_obj)

    def fit_single_model(self, data_instances, validate_data=None):
        LOGGER.info(f"Start to train single {self.model_name}")
        self.callback_list.on_train_begin(data_instances, validate_data)

        self.exposure_index = BasePoissonRegression.get_exposure_index(self.header, self.exposure_colname)
        exposure_index = self.exposure_index
        if exposure_index > -1:
            self.header.pop(exposure_index)
            LOGGER.info("Guest provides exposure value.")
        exposure = data_instances.mapValues(lambda v: BasePoissonRegression.load_exposure(v, exposure_index))
        data_instances = data_instances.mapValues(lambda v: BasePoissonRegression.load_instance(v, exposure_index))

        model_shape = self.get_features_shape(data_instances)
        instances_count = data_instances.count()

        if not self.component_properties.is_warm_start:
            w = self._init_weights(model_shape)
            self.model_weights = LinearModelWeights(l=w,
                                                    fit_intercept=self.model_param.init_param.fit_intercept)
            last_models = copy.deepcopy(self.model_weights)
        else:
            last_models = copy.deepcopy(self.model_weights)
            w = last_models.unboxed
            self.callback_warm_start_init_iter(self.n_iter_)

        self.batch_generator.initialize_batch_generator(data_instances, batch_size=self.batch_size)

        with SPDZ(
                "hetero_sshe",
                local_party=self.local_party,
                all_parties=self.parties,
                q_field=self.q_field,
                use_mix_rand=self.model_param.use_mix_rand,
        ) as spdz:
            spdz.set_flowid(self.flowid)
            self.secure_matrix_obj.set_flowid(self.flowid)
            self.labels_original = data_instances.mapValues(lambda x: np.array([x.label], dtype=int))
            self.labels = self.labels_original.mapValues(lambda x: np.array([BasePoissonRegression.safe_log(x[0])], dtype=float))
            w_self, w_remote = self.share_model(w, suffix="init")
            last_w_self, last_w_remote = w_self, w_remote
            LOGGER.debug(f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

            batch_data_generator = self.batch_generator.generate_batch_data()

            self.cipher_tool = []
            encoded_batch_data = []
            for batch_data in batch_data_generator:
                if self.fit_intercept:
                    batch_features = batch_data.mapValues(lambda x: np.hstack((x.features, 1.0)))
                else:
                    batch_features = batch_data.mapValues(lambda x: x.features)
                self.batch_num.append(batch_data.count())

                encoded_batch_data.append(
                    fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(batch_features),
                                                      q_field=self.fixedpoint_encoder.n,
                                                      endec=self.fixedpoint_encoder))

                self.cipher_tool.append(EncryptModeCalculator(self.cipher,
                                                              self.encrypted_mode_calculator_param.mode,
                                                              self.encrypted_mode_calculator_param.re_encrypted_rate))

            while self.n_iter_ < self.max_iter:
                self.callback_list.on_epoch_begin(self.n_iter_)
                LOGGER.info(f"start to n_iter: {self.n_iter_}")

                loss_list = []

                self.optimizer.set_iters(self.n_iter_)
                if not self.reveal_every_iter:
                    self.self_optimizer.set_iters(self.n_iter_)
                    self.remote_optimizer.set_iters(self.n_iter_)

                for batch_idx, batch_data in enumerate(encoded_batch_data):
                    current_suffix = (str(self.n_iter_), str(batch_idx))
                    batch_offset = exposure.join(batch_data.value, lambda ei, d: np.array([ei]))
                    batch_offset_log = batch_offset.mapValues(lambda v: np.array([BasePoissonRegression.safe_log(v[0])]))

                    if self.reveal_every_iter:
                        y = self.forward(weights=self.model_weights,
                                         features=batch_data,
                                         suffix=current_suffix,
                                         cipher=self.cipher_tool[batch_idx],
                                         offset=batch_offset,
                                         log_offset=batch_offset_log)
                    else:
                        y = self.forward(weights=(w_self, w_remote),
                                         features=batch_data,
                                         suffix=current_suffix,
                                         cipher=self.cipher_tool[batch_idx],
                                         offset=batch_offset,
                                         log_offset=batch_offset_log)

                    error = y - self.labels
                    self_g, remote_g = self.backward(error=error,
                                                     features=batch_data,
                                                     suffix=current_suffix,
                                                     cipher=self.cipher_tool[batch_idx])

                    # loss computing;
                    suffix = ("loss",) + current_suffix
                    if self.reveal_every_iter:
                        batch_loss = self.compute_loss(weights=self.model_weights, suffix=suffix,
                                                       cipher=self.cipher_tool[batch_idx], batch_offset=batch_offset_log)
                    else:
                        batch_loss = self.compute_loss(weights=(w_self, w_remote), suffix=suffix,
                                                       cipher=self.cipher_tool[batch_idx], batch_offset=batch_offset_log)

                    if batch_loss is not None:
                        batch_loss = batch_loss * self.batch_num[batch_idx]
                    loss_list.append(batch_loss)

                    if self.reveal_every_iter:
                        # LOGGER.debug(f"before reveal: self_g shape: {self_g.shape}, remote_g_shape: {remote_g}，"
                        #              f"self_g: {self_g}")

                        new_g = self.reveal_models(self_g, remote_g, suffix=current_suffix)

                        # LOGGER.debug(f"after reveal: new_g shape: {new_g.shape}, new_g: {new_g}"
                        #              f"self.model_param.reveal_strategy: {self.model_param.reveal_strategy}")

                        if new_g is not None:
                            self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                             has_applied=False)

                        else:
                            self.model_weights = LinearModelWeights(
                                l=np.zeros(self_g.shape),
                                fit_intercept=self.model_param.init_param.fit_intercept)
                    else:
                        if self.optimizer.penalty == consts.L2_PENALTY:
                            self_g = self_g + self.self_optimizer.alpha * w_self
                            remote_g = remote_g + self.remote_optimizer.alpha * w_remote

                        # LOGGER.debug(f"before optimizer: {self_g}, {remote_g}")

                        self_g = self.self_optimizer.apply_gradients(self_g)
                        remote_g = self.remote_optimizer.apply_gradients(remote_g)

                        # LOGGER.debug(f"after optimizer: {self_g}, {remote_g}")
                        w_self -= self_g
                        w_remote -= remote_g

                    LOGGER.debug(f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

                loss = np.sum(loss_list) / instances_count
                self.loss_history.append(loss)
                if self.need_call_back_loss:
                    self.callback_loss(self.n_iter_, loss)

                if self.converge_func_name in ["diff", "abs"]:
                    self.is_converged = self.check_converge_by_loss(loss, suffix=(str(self.n_iter_),))
                elif self.converge_func_name == "weight_diff":
                    if self.reveal_every_iter:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=last_models.unboxed,
                            new_w=self.model_weights.unboxed,
                            suffix=(str(self.n_iter_),))
                        last_models = copy.deepcopy(self.model_weights)
                    else:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=(last_w_self, last_w_remote),
                            new_w=(w_self, w_remote),
                            suffix=(str(self.n_iter_),))
                        last_w_self, last_w_remote = copy.deepcopy(w_self), copy.deepcopy(w_remote)
                else:
                    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                self.n_iter_ += 1

                if self.stop_training:
                    break

                if self.is_converged:
                    break

            # Finally reconstruct
            if not self.reveal_every_iter:
                new_w = self.reveal_models(w_self, w_remote, suffix=("final",))
                if new_w is not None:
                    self.model_weights = LinearModelWeights(
                        l=new_w,
                        fit_intercept=self.model_param.init_param.fit_intercept)

        LOGGER.debug(f"loss_history: {self.loss_history}")
        self.set_summary(self.get_model_summary())

    def fit(self, data_instances, validate_data=None):
        LOGGER.info("Starting to fit hetero_sshe_poisson_regression")
        self.batch_generator = batch_generator.Guest()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.header = data_instances.schema.get("header", [])
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)

        self.fit_single_model(data_instances, validate_data)

    def get_metrics_param(self):
        return EvaluateParam(eval_type="regression", metrics=self.metrics)
