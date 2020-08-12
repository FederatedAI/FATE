from federatedml.hetero_ftl.ftl_base import FTL
from federatedml.statistic.intersect import intersect_host
from arch.api.utils import log_utils
from federatedml.hetero_ftl.ftl_dataloder import FTLDataLoader
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.util import consts

from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor

from federatedml.util.io_check import assert_io_num_rows_equal

from federatedml.statistic import data_overview

import numpy as np

LOGGER = log_utils.getLogger()


class FTLHost(FTL):

    def __init__(self):
        super(FTLHost, self).__init__()
        self.overlap_ub = None  # u_b
        self.overlap_ub_2 = None  # u_b squared
        self.mapping_comp_b = None
        self.constant_k = None  # κ
        self.feat_dim = None  # output feature dimension
        self.m_b = None  # random mask
        self.role = consts.HOST

    def init_intersect_obj(self):
        LOGGER.debug('creating intersect obj done')
        intersect_obj = intersect_host.RsaIntersectionHost(self.intersect_param)
        intersect_obj.host_party_id = self.component_properties.local_partyid
        intersect_obj.host_party_id_list = self.component_properties.host_party_idlist
        return intersect_obj

    def batch_compute_components(self, data_loader: FTLDataLoader):

        """
        compute host components
        """

        overlap_ub = []
        for i in range(len(data_loader)):
            batch_x = data_loader[i]
            ub_batch = self.nn.predict(batch_x)
            overlap_ub.append(ub_batch)

        overlap_ub = np.concatenate(overlap_ub, axis=0)
        overlap_ub_2 = np.matmul(np.expand_dims(overlap_ub, axis=2), np.expand_dims(overlap_ub, axis=1))
        mapping_comp_b = - overlap_ub * self.constant_k

        if self.verbose:
            LOGGER.debug('overlap_ub is {}'.format(overlap_ub))
            LOGGER.debug('overlap_ub_2 is {}'.format(overlap_ub_2))

        return overlap_ub, overlap_ub_2, mapping_comp_b

    def exchange_components(self, comp_to_send, epoch_idx):

        """
        compute host components and sent to guest
        """

        if self.mode == 'encrypted':
            comp_to_send = self.encrypt_tensor(comp_to_send)

        self.transfer_variable.host_components.remote(comp_to_send, suffix=(epoch_idx, 'exchange_host_comp'))
        guest_components = self.transfer_variable.guest_components.get(idx=0, suffix=(epoch_idx, 'exchange_guest_comp'))

        if self.mode == 'encrypted':
            guest_paillier_tensors = [PaillierTensor(tb_obj=tb) for tb in guest_components]
            return guest_paillier_tensors
        else:
            return guest_components

    def decrypt_guest_data(self, epoch_idx, local_round=-1):

        encrypted_guest_data = self.transfer_variable.guest_side_gradients.\
                                    get(suffix=(epoch_idx, local_round, 'guest_de_send'), idx=0)
        encrypted_consts, grad_table = encrypted_guest_data[0], encrypted_guest_data[1]
        inter_grad = PaillierTensor(tb_obj=grad_table, )
        decrpyted_grad = inter_grad.decrypt(self.encrypter)
        decrypted_const = self.encrypter.recursive_decrypt(encrypted_consts)
        self.transfer_variable.decrypted_guest_gradients.remote([decrypted_const, decrpyted_grad.get_obj()],
                                                                suffix=(epoch_idx, local_round, 'guest_de_get'))

    def decrypt_inter_result(self, loss_grad_b, epoch_idx, local_round=-1):

        rand_0 = PaillierTensor(ori_data=self.rng_generator.generate_random_number(loss_grad_b.shape))
        grad_a_overlap = loss_grad_b + rand_0
        self.transfer_variable.host_side_gradients.remote(grad_a_overlap.get_obj(),
                                                          suffix=(epoch_idx, local_round, 'host_de_send'))
        de_loss_grad_b = self.transfer_variable.decrypted_host_gradients\
                                               .get(suffix=(epoch_idx, local_round, 'host_de_get'), idx=0)
        de_loss_grad_b = PaillierTensor(tb_obj=de_loss_grad_b) - rand_0

        return de_loss_grad_b

    def compute_backward_gradients(self, guest_components, data_loader: FTLDataLoader, epoch_idx, local_round=-1):

        """
        compute host bottom model gradients
        """

        y_overlap_2_phi_2, y_overlap_phi, mapping_comp_a = guest_components[0], guest_components[1], guest_components[2]

        ub_overlap_ex = np.expand_dims(self.overlap_ub, axis=1)

        if self.mode == 'plain':

            ub_overlap_y_overlap_2_phi_2 = np.matmul(ub_overlap_ex, y_overlap_2_phi_2)
            l1_grad_b = np.squeeze(ub_overlap_y_overlap_2_phi_2, axis=1) + y_overlap_phi
            loss_grad_b = self.alpha * l1_grad_b + mapping_comp_a

            return loss_grad_b

        if self.mode == 'encrypted':

            ub_overlap_ex = np.expand_dims(self.overlap_ub, axis=1)
            ub_overlap_y_overlap_2_phi_2 = y_overlap_2_phi_2.matmul_3d(ub_overlap_ex, multiply='right')
            ub_overlap_y_overlap_2_phi_2 = ub_overlap_y_overlap_2_phi_2.squeeze(axis=1)

            l1_grad_b = ub_overlap_y_overlap_2_phi_2 + y_overlap_phi
            en_loss_grad_b = l1_grad_b * self.alpha + mapping_comp_a

            self.decrypt_guest_data(epoch_idx, local_round=local_round)
            loss_grad_b = self.decrypt_inter_result(en_loss_grad_b, epoch_idx, local_round=local_round)

            return loss_grad_b.numpy()

    def compute_loss(self, epoch_idx):

        """
        help guest compute ftl loss. plain mode will skip/ in encrypted mode will decrypt received loss
        """

        if self.mode == 'plain':
            return

        elif self.mode == 'encrypted':
            encrypted_loss = self.transfer_variable.encrypted_loss.get(idx=0, suffix=(epoch_idx, 'send_loss'))
            rs = self.encrypter.recursive_decrypt(encrypted_loss)
            self.transfer_variable.decrypted_loss.remote(rs, suffix=(epoch_idx, 'get_loss'))

    def fit(self, data_inst, validate_data):

        LOGGER.info('start to fit a ftl model, '
                    'run mode is {},'
                    'communication efficient mode is {}'.format(self.mode, self.comm_eff))

        data_loader, self.x_shape, self.data_num, self.overlap_num = self.prepare_data(self.init_intersect_obj(),
                                                                                       data_inst, guest_side=False)
        self.input_dim = self.x_shape[0]
        # cache data_loader for faster validation
        self.cache_dataloader[(data_inst.get_name(), data_inst.get_namespace())] = data_loader

        self.partitions = data_inst._partitions
        self.initialize_nn(input_shape=self.x_shape)
        self.feat_dim = self.nn._model.output_shape[1]
        self.constant_k = 1 / self.feat_dim
        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)

        for epoch_idx in range(self.epochs):

            LOGGER.debug('fitting epoch {}'.format(epoch_idx))

            self.overlap_ub, self.overlap_ub_2, self.mapping_comp_b = self.batch_compute_components(data_loader)
            send_components = [self.overlap_ub, self.overlap_ub_2, self.mapping_comp_b]
            guest_components = self.exchange_components(send_components, epoch_idx)

            for local_round_idx in range(self.local_round):

                if self.comm_eff:
                    LOGGER.debug('running local iter {}'.format(local_round_idx))

                grads = self.compute_backward_gradients(guest_components, data_loader, epoch_idx,
                                                        local_round=local_round_idx)
                self.update_nn_weights(grads, data_loader, epoch_idx, decay=self.comm_eff)

                if local_round_idx == 0:
                    self.compute_loss(epoch_idx)

                if local_round_idx + 1 != self.local_round:
                    self.overlap_ub, self.overlap_ub_2, self.mapping_comp_b = self.batch_compute_components(data_loader)

            if self.validation_strategy is not None:
                self.validation_strategy.validate(self, epoch_idx)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            if self.n_iter_no_change is True:
                stop_flag = self.sync_stop_flag(epoch_idx)
                if stop_flag:
                    break

            LOGGER.debug('fitting epoch {} done'.format(epoch_idx))

    @assert_io_num_rows_equal
    def predict(self, data_inst):

        LOGGER.debug('host start to predict')

        self.transfer_variable.predict_host_u.disable_auto_clean()

        data_loader_key = self.get_dataset_key(data_inst)

        data_inst_ = data_overview.header_alignment(data_inst, self.store_header)

        if data_loader_key in self.cache_dataloader:
            data_loader = self.cache_dataloader[data_loader_key]
        else:
            data_loader, _, _, _ = self.prepare_data(self.init_intersect_obj(), data_inst_, guest_side=False)
            self.cache_dataloader[data_loader_key] = data_loader

        ub_batches = []

        for i in range(len(data_loader)):
            batch_x = data_loader[i]
            ub_batch = self.nn.predict(batch_x)
            ub_batches.append(ub_batch)

        predicts = np.concatenate(ub_batches, axis=0)

        self.transfer_variable.predict_host_u.remote(predicts, suffix=(0, 'host_u'))

        LOGGER.debug('ftl host prediction done')

        return None

    def export_model(self):
        return {"FTLHostMeta": self.get_model_meta(), "FTLHostParam": self.get_model_param()}

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




