from federatedml.hetero_ftl.ftl_base import FTL
from federatedml.statistic.intersect import intersect_guest
from arch.api.utils import log_utils
from federatedml.hetero_ftl.ftl_dataloder import FTLDataLoader

from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta

from federatedml.optim.convergence import converge_func_factory
from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor

from arch.api import session

import numpy as np

from federatedml.util import consts

from federatedml.util.io_check import assert_io_num_rows_equal

from federatedml.statistic import data_overview

LOGGER = log_utils.getLogger()


class FTLGuest(FTL):

    def __init__(self):
        super(FTLGuest, self).__init__()
        self.phi = None  # Φ_A
        self.phi_product = None  # (Φ_A)‘(Φ_A) [feature_dim, feature_dim]
        self.overlap_y = None  # y_i ∈ N_c
        self.overlap_y_2 = None  # (y_i ∈ N_c )^2
        self.overlap_ua = None  # u_i ∈ N_AB
        self.constant_k = None  # κ
        self.feat_dim = None  # output feature dimension
        self.send_components = None  # components to send
        self.convergence = None

        self.overlap_y_pt = None  # paillier tensor

        self.history_loss = []  # list to record history loss

        self.role = consts.GUEST

    def init_intersect_obj(self):
        intersect_obj = intersect_guest.RsaIntersectionGuest(self.intersect_param)
        intersect_obj.guest_party_id = self.component_properties.local_partyid
        intersect_obj.host_party_id_list = self.component_properties.host_party_idlist
        LOGGER.debug('intersect done')
        return intersect_obj

    def check_convergence(self, loss):
        LOGGER.info("check convergence")
        if self.convergence is None:
            self.convergence = converge_func_factory("diff", self.tol)
        return self.convergence.is_converge(loss)

    def compute_phi_and_overlap_ua(self, data_loader: FTLDataLoader):

        """
        compute Φ and ua of overlap samples
        """

        phi = None  # [1, feature_dim]  Φ_A
        overlap_ua = []
        for i in range(len(data_loader)):
            batch_x, batch_y = data_loader[i]
            ua_batch = self.nn.predict(batch_x)  # [batch_size, feature_dim]

            relative_overlap_index = data_loader.get_relative_overlap_index(i)
            if len(relative_overlap_index) != 0:
                if self.verbose:
                    LOGGER.debug('batch {}/{} overlap index is {}'.format(i, len(data_loader), relative_overlap_index))
                overlap_ua.append(ua_batch[relative_overlap_index])

            phi_tmp = np.expand_dims(np.sum(batch_y * ua_batch, axis=0), axis=0)
            if phi is None:
                phi = phi_tmp
            else:
                phi += phi_tmp

        phi = phi / self.data_num

        return phi, overlap_ua

    def batch_compute_components(self, data_loader: FTLDataLoader):

        """
        compute guest components
        """

        phi, overlap_ua = self.compute_phi_and_overlap_ua(data_loader)  # Φ_A [1, feature_dim]

        phi_product = np.matmul(phi.transpose(), phi)  # (Φ_A)‘(Φ_A) [feature_dim, feature_dim]

        if self.overlap_y is None:
            self.overlap_y = data_loader.get_overlap_y()  # {C(y)=y} [1, feat_dim]
        if self.overlap_y_2 is None:
            self.overlap_y_2 = self.overlap_y * self.overlap_y  # {D(y)=y^2} # [1, feat_dim]

        overlap_ua = np.concatenate(overlap_ua, axis=0)  # [overlap_num, feat_dim]

        # 3 components will be sent to host
        y_overlap_2_phi_2 = 0.25 * np.expand_dims(self.overlap_y_2, axis=2) * phi_product
        y_overlap_phi = -0.5 * self.overlap_y * phi
        mapping_comp_a = -overlap_ua * self.constant_k

        return phi, phi_product, overlap_ua, [y_overlap_2_phi_2, y_overlap_phi, mapping_comp_a]

    def exchange_components(self, comp_to_send, epoch_idx):

        """
        send guest components and get host components
        """

        if self.mode == 'encrypted':
            comp_to_send = self.encrypt_tensor(comp_to_send)

        self.transfer_variable.guest_components.remote(comp_to_send, suffix=(epoch_idx, 'exchange_guest_comp'))
        host_components = self.transfer_variable.host_components.get(idx=0, suffix=(epoch_idx, 'exchange_host_comp'))
        if self.mode == 'encrypted':
            host_paillier_tensors = [PaillierTensor(tb_obj=tb) for tb in host_components]
            return host_paillier_tensors
        else:
            return host_components

    def decrypt_inter_result(self, encrypted_const, grad_a_overlap, epoch_idx, local_round=-1):

        """
        add random mask to encrypted inter-result, get decrypted data from host add subtract random mask
        """

        rand_0 = self.rng_generator.generate_random_number(encrypted_const.shape)
        encrypted_const = encrypted_const + rand_0
        rand_1 = PaillierTensor(ori_data=self.rng_generator.generate_random_number(grad_a_overlap.shape))
        grad_a_overlap = grad_a_overlap + rand_1

        send_data = [encrypted_const, grad_a_overlap.get_obj()]
        self.transfer_variable.guest_side_gradients.remote(send_data, suffix=(epoch_idx,
                                                                              local_round,
                                                                              'guest_de_send'))
        rs = self.transfer_variable.decrypted_guest_gradients.get(suffix=(epoch_idx,
                                                                          local_round,
                                                                          'guest_de_get'), idx=0)

        const = rs[0] - rand_0
        grad_a_overlap = PaillierTensor(tb_obj=rs[1]) - rand_1

        return const, grad_a_overlap

    def decrypt_host_data(self, epoch_idx, local_round=-1):

        inter_grad = self.transfer_variable.host_side_gradients.get(suffix=(epoch_idx,
                                                                            local_round,
                                                                            'host_de_send'), idx=0)
        inter_grad_pt = PaillierTensor(tb_obj=inter_grad)
        self.transfer_variable.decrypted_host_gradients.remote(inter_grad_pt.decrypt(self.encrypter).get_obj(),
                                                               suffix=(epoch_idx,
                                                                       local_round,
                                                                       'host_de_get'))

    def decrypt_loss_val(self, encrypted_loss, epoch_idx):

        self.transfer_variable.encrypted_loss.remote(encrypted_loss, suffix=(epoch_idx, 'send_loss'))
        decrypted_loss = self.transfer_variable.decrypted_loss.get(idx=0, suffix=(epoch_idx, 'get_loss'))
        return decrypted_loss

    def compute_backward_gradients(self, host_components, data_loader: FTLDataLoader, epoch_idx, local_round=-1):

        """
        compute backward gradients using host components
        """

        # they are Paillier tensors or np array
        overlap_ub, overlap_ub_2, mapping_comp_b = host_components[0], host_components[1], host_components[2]

        y_overlap_2_phi = np.expand_dims(self.overlap_y_2 * self.phi, axis=1)

        if self.mode == 'plain':

            loss_grads_const_part1 = 0.25 * np.squeeze(np.matmul(y_overlap_2_phi, overlap_ub_2), axis=1)
            loss_grads_const_part2 = self.overlap_y * overlap_ub

            const = np.sum(loss_grads_const_part1, axis=0) - 0.5 * np.sum(loss_grads_const_part2, axis=0)

            grad_a_nonoverlap = self.alpha * const * data_loader.y[data_loader.get_non_overlap_indexes()] / self.data_num
            grad_a_overlap = self.alpha * const * self.overlap_y / self.data_num + mapping_comp_b

            return np.concatenate([grad_a_overlap, grad_a_nonoverlap], axis=0)

        elif self.mode == 'encrypted':

            loss_grads_const_part1 = overlap_ub_2.matmul_3d(0.25 * y_overlap_2_phi, multiply='right')
            loss_grads_const_part1 = loss_grads_const_part1.squeeze(axis=1)

            if self.overlap_y_pt is None:
                self.overlap_y_pt = PaillierTensor(self.overlap_y, partitions=self.partitions)

            loss_grads_const_part2 = overlap_ub * self.overlap_y_pt

            encrypted_const = loss_grads_const_part1.reduce_sum() - 0.5 * loss_grads_const_part2.reduce_sum()

            grad_a_overlap = self.overlap_y_pt.map_ndarray_product((self.alpha/self.data_num * encrypted_const)) + mapping_comp_b

            const, grad_a_overlap = self.decrypt_inter_result(encrypted_const, grad_a_overlap, epoch_idx=epoch_idx
                                                              , local_round=local_round)

            self.decrypt_host_data(epoch_idx, local_round=local_round)

            grad_a_nonoverlap = self.alpha * const * data_loader.y[data_loader.get_non_overlap_indexes()]/self.data_num

            return np.concatenate([grad_a_overlap.numpy(), grad_a_nonoverlap], axis=0)

    def compute_loss(self, host_components, epoch_idx, overlap_num):

        """
        compute training loss
        """

        overlap_ub, overlap_ub_2, mapping_comp_b = host_components[0], host_components[1], host_components[2]

        if self.mode == 'plain':

            loss_overlap = np.sum((-self.overlap_ua * self.constant_k) * overlap_ub)

            ub_phi = np.matmul(overlap_ub, self.phi.transpose())
            part1 = -0.5*np.sum(self.overlap_y*ub_phi)
            part2 = 1.0/8*np.sum(ub_phi * ub_phi)
            part3 = len(self.overlap_y)*np.log(2)
            loss_y = part1 + part2 + part3
            return self.alpha * (loss_y/overlap_num) + loss_overlap/overlap_num

        elif self.mode == 'encrypted':

            loss_overlap = overlap_ub.element_wise_product((-self.overlap_ua*self.constant_k))
            sum = np.sum(loss_overlap.reduce_sum())
            ub_phi = overlap_ub.T.fast_matmul_2d(self.phi.transpose())

            part1 = -0.5 * np.sum((self.overlap_y * ub_phi))
            ub_2 = overlap_ub_2.reduce_sum()
            enc_phi_uB_2_phi = np.matmul(np.matmul(self.phi, ub_2), self.phi.transpose())
            part2 = 1/8 * np.sum(enc_phi_uB_2_phi)
            part3 = len(self.overlap_y)*np.log(2)

            loss_y = part1 + part2 + part3
            en_loss = (self.alpha/self.overlap_num) * loss_y + sum / overlap_num

            loss_val = self.decrypt_loss_val(en_loss, epoch_idx)

            return loss_val

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def fit(self, data_inst, validate_data):

        LOGGER.info('start to fit a ftl model, '
                    'run mode is {},'
                    'communication efficient mode is {}'.format(self.mode, self.comm_eff))

        data_loader, self.x_shape, self.data_num, self.overlap_num = self.prepare_data(self.init_intersect_obj(),
                                                                                       data_inst, guest_side=True)
        self.input_dim = self.x_shape[0]

        # cache data_loader for faster validation
        self.cache_dataloader[self.get_dataset_key(data_inst)] = data_loader

        self.partitions = data_inst._partitions

        self.initialize_nn(input_shape=self.x_shape)
        self.feat_dim = self.nn._model.output_shape[1]
        self.constant_k = 1 / self.feat_dim
        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)

        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"unit_name": "iters"}))

        # compute intermediate result of first epoch
        self.phi, self.phi_product, self.overlap_ua, self.send_components = self.batch_compute_components(data_loader)

        for epoch_idx in range(self.epochs):

            LOGGER.debug('fitting epoch {}'.format(epoch_idx))

            host_components = self.exchange_components(self.send_components, epoch_idx=epoch_idx)

            loss = None

            for local_round_idx in range(self.local_round):

                if self.comm_eff:
                    LOGGER.debug('running local iter {}'.format(local_round_idx))

                grads = self.compute_backward_gradients(host_components, data_loader, epoch_idx=epoch_idx,
                                                        local_round=local_round_idx)
                self.update_nn_weights(grads, data_loader, epoch_idx, decay=self.comm_eff)

                if local_round_idx == 0:
                    loss = self.compute_loss(host_components, epoch_idx, len(data_loader.get_overlap_indexes()))

                if local_round_idx + 1 != self.local_round:
                    self.phi, self.overlap_ua = self.compute_phi_and_overlap_ua(data_loader)

            self.callback_metric("loss", "train", [Metric(epoch_idx, loss)])
            self.history_loss.append(loss)

            # updating variables for next epochs
            if epoch_idx + 1 == self.epochs:
                # only need to update phi in last epochs
                self.phi, _ = self.compute_phi_and_overlap_ua(data_loader)
            else:
                # compute phi, phi_product, overlap_ua etc. for next epoch
                self.phi, self.phi_product, self.overlap_ua, self.send_components = self.batch_compute_components(
                    data_loader)

            # check early_stopping
            if self.validation_strategy is not None:
                self.validation_strategy.validate(self, epoch_idx)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            # check n_iter_no_change
            if self.n_iter_no_change is True:
                if self.check_convergence(loss):
                    self.sync_stop_flag(epoch_idx, stop_flag=True)
                    break
                else:
                    self.sync_stop_flag(epoch_idx, stop_flag=False)

            LOGGER.debug('fitting epoch {} done, loss is {}'.format(epoch_idx, loss))

        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"Best": min(self.history_loss)}))

        LOGGER.debug('fitting ftl model done')


    def predict(self, data_inst):

        LOGGER.debug('guest start to predict')

        data_loader_key = self.get_dataset_key(data_inst)

        data_inst_ = data_overview.header_alignment(data_inst, self.store_header)

        if data_loader_key in self.cache_dataloader:
            data_loader = self.cache_dataloader[data_loader_key]
        else:
            data_loader, _, _, _ = self.prepare_data(self.init_intersect_obj(), data_inst_, guest_side=True)
            self.cache_dataloader[data_loader_key] = data_loader

        LOGGER.debug('try to get predict u from host, suffix is {}'.format((0, 'host_u')))
        host_predicts = self.transfer_variable.predict_host_u.get(idx=0, suffix=(0, 'host_u'))

        predict_score = np.matmul(host_predicts, self.phi.transpose())
        predicts = self.sigmoid(predict_score)  # convert to predict scores
        predicts = list(map(float, predicts))

        predict_tb = session.parallelize(zip(data_loader.get_overlap_keys(), predicts,), include_key=True,
                                         partition=data_inst._partitions)

        threshold = self.predict_param.threshold
        predict_result = self.predict_score_to_output(data_inst_, predict_tb, classes=[0, 1], threshold=threshold)


        LOGGER.debug('ftl guest prediction done')

        return predict_result

    def export_model(self):
        model_param = self.get_model_param()
        model_param.phi_a.extend(self.phi.tolist()[0])
        return {"FTLGuestMeta": self.get_model_meta(), "FTLHostParam": model_param}

    def load_model(self, model_dict):
        model_param = None
        model_meta = None
        for _, value in model_dict["model"].items():
            for model in value:
                if model.endswith("Meta"):
                    model_meta = value[model]
                if model.endswith("Param"):
                    model_param = value[model]
        LOGGER.info("load model")

        self.set_model_meta(model_meta)
        self.set_model_param(model_param)
        self.phi = np.array([model_param.phi_a])
