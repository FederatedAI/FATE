from federatedml.nn.homo_nn.nn_model import get_nn_builder
from federatedml.model_base import ModelBase
from federatedml.param.ftl_param import FTLParam
from federatedml.nn.homo_nn.nn_model import NNModel
from federatedml.nn.backend.tf_keras.nn_model import KerasNNModel
from federatedml.util.classify_label_checker import ClassifyLabelChecker
from federatedml.transfer_variable.transfer_class.ftl_transfer_variable_transfer_variable import FTLTransferVariable
from federatedml.hetero_ftl.ftl_dataloder import FTLDataLoader
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.nn.hetero_nn.util import random_number_generator
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.util import consts
from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor
from federatedml.protobuf.generated.ftl_model_param_pb2 import FTLModelParam
from federatedml.protobuf.generated.ftl_model_meta_pb2 import FTLModelMeta, FTLPredictParam, FTLOptimizerParam
from federatedml.util.validation_strategy import ValidationStrategy

from arch.api.utils import log_utils
import json

import numpy as np

LOGGER = log_utils.getLogger()


class FTL(ModelBase):

    def __init__(self):
        super(FTL, self).__init__()

        # input para
        self.nn_define = None
        self.alpha = None
        self.tol = None
        self.learning_rate = None
        self.n_iter_no_change = None
        self.validation_freqs = None
        self.early_stopping_rounds = None
        self.use_first_metric_only = None
        self.optimizer = None
        self.intersect_param = None
        self.config_type = None
        self.comm_eff = None
        self.local_round = 1

        self.encrypted_mode_calculator_param = None

        # runtime variable
        self.verbose = False
        self.nn: KerasNNModel = None
        self.nn_builder = None
        self.model_param = FTLParam()
        self.x_shape = None
        self.input_dim = None
        self.data_num = 0
        self.overlap_num = 0
        self.transfer_variable = FTLTransferVariable()
        self.data_convertor = KerasSequenceDataConverter()
        self.mode = 'plain'
        self.encrypt_calculators = []
        self.encrypter = None
        self.partitions = 10
        self.batch_size = None
        self.epochs = None
        self.store_header = None  # header of input data table

        self.cache_dataloader = {}

        self.validation_strategy = None

    def _init_model(self, param: FTLParam):

        self.nn_define = param.nn_define
        self.alpha = param.alpha
        self.tol = param.tol
        self.n_iter_no_change = param.n_iter_no_change
        self.validation_freqs = param.validation_freqs
        self.optimizer = param.optimizer
        self.intersect_param = param.intersect_param
        self.config_type = param.config_type
        self.batch_size = param.batch_size
        self.epochs = param.epochs
        self.mode = param.mode
        self.comm_eff = param.communication_efficient
        self.local_round = param.local_round
        assert 'learning_rate' in self.optimizer.kwargs, 'optimizer setting must contain learning_rate'
        self.learning_rate = self.optimizer.kwargs['learning_rate']

        if not self.comm_eff:
            self.local_round = 1
            LOGGER.debug('communication efficient mode is not enabled, local_round set as 1')

        self.encrypted_mode_calculator_param = param.encrypted_mode_calculator_param
        self.encrypter = self.generate_encrypter(param)
        self.predict_param = param.predict_param
        self.rng_generator = random_number_generator.RandomNumberGenerator()

    @staticmethod
    def debug_data_inst(data_inst):
        collect_data = list(data_inst.collect())
        LOGGER.debug('showing DTable')
        for d in collect_data:
            LOGGER.debug('key {} id {}, features {} label {}'.format(d[0], d[1].inst_id, d[1].features, d[1].label))

    @staticmethod
    def check_label(data_inst):

        """
        check label. FTL only supports binary classification, and labels should be 1 or -1
        """

        LOGGER.debug('checking label')
        label_checker = ClassifyLabelChecker()
        num_class, class_set = label_checker.validate_label(data_inst)
        if num_class != 2:
            raise ValueError('ftl only support binary classification, however {} labels are provided.'.format(num_class))

        if 1 in class_set and -1 in class_set:
            return data_inst
        else:
            new_label_mapping = {list(class_set)[0]: 1, list(class_set)[1]: -1}

            def reset_label(inst):
                inst.label = new_label_mapping[inst.label]
                return inst

            new_table = data_inst.mapValues(reset_label)
            return new_table

    def generate_encrypter(self, param) -> PaillierEncrypt:
        LOGGER.info("generate encrypter")
        if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
            encrypter = PaillierEncrypt()
            encrypter.generate_key(param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet!!!")

        return encrypter

    def generated_encrypted_calculator(self):
        encrypted_calculator = EncryptModeCalculator(self.encrypter,
                                                     self.encrypted_mode_calculator_param.mode,
                                                     self.encrypted_mode_calculator_param.re_encrypted_rate)
        return encrypted_calculator

    def encrypt_tensor(self, components, return_dtable=True):

        """
        transform numpy array into Paillier tensor and encrypt
        """

        if len(self.encrypt_calculators) == 0:
            self.encrypt_calculators = [self.generated_encrypted_calculator() for i in range(3)]
        encrypted_tensors = []
        for comp, calculator in zip(components, self.encrypt_calculators):
            encrypted_tensor = PaillierTensor(ori_data=comp, partitions=self.partitions)
            if return_dtable:
                encrypted_tensors.append(encrypted_tensor.encrypt(calculator).get_obj())
            else:
                encrypted_tensors.append(encrypted_tensor.encrypt(calculator))

        return encrypted_tensors

    def init_validation_strategy(self, train_data=None, validate_data=None):
        validation_strategy = ValidationStrategy(self.role, consts.HETERO, self.validation_freqs,
                                                 self.early_stopping_rounds, self.use_first_metric_only)
        validation_strategy.set_train_data(train_data)
        validation_strategy.set_validate_data(validate_data)
        return validation_strategy

    def learning_rate_decay(self, learning_rate, epoch):
        """
        learning_rate decay
        """
        return learning_rate * 1 / np.sqrt(epoch + 1)

    def sync_stop_flag(self, num_round, stop_flag=None):
        """
        stop flag for n_iter_no_change
        """
        LOGGER.info("sync stop flag, boosting round is {}".format(num_round))
        if self.role == consts.GUEST:
            self.transfer_variable.stop_flag.remote(stop_flag,
                                                    role=consts.HOST,
                                                    idx=-1,
                                                    suffix=(num_round,))
        elif self.role == consts.HOST:
            return self.transfer_variable.stop_flag.get(idx=0, suffix=(num_round, ))

    def prepare_data(self, intersect_obj, data_inst, guest_side=False):

        """
        find intersect ids and prepare dataloader
        """
        overlap_samples = intersect_obj.run(data_inst)  # find intersect ids
        non_overlap_samples = data_inst.subtractByKey(overlap_samples)

        self.store_header = data_inst.schema['header']
        LOGGER.debug('data inst header is {}'.format(self.store_header))

        if overlap_samples.count() == 0:
            raise ValueError('no intersect samples')

        LOGGER.debug('has {} overlap samples'.format(overlap_samples.count()))

        if guest_side:
            data_inst = self.check_label(data_inst)

        batch_size = self.batch_size
        if self.batch_size == -1:
            batch_size = data_inst.count() + 1  # make sure larger than sample number
        data_loader = FTLDataLoader(non_overlap_samples=non_overlap_samples,
                                    batch_size=batch_size, overlap_samples=overlap_samples, guest_side=guest_side)

        LOGGER.debug("data details are :{}".format(data_loader.data_basic_info()))

        return data_loader, data_loader.x_shape, data_inst.count(), len(data_loader.get_overlap_indexes())

    def initialize_nn(self, input_shape):

        """
        initializing nn weights
        """

        loss = "keep_predict_loss"
        self.nn_builder = get_nn_builder(config_type=self.config_type)
        self.nn: NNModel = self.nn_builder(loss=loss, nn_define=self.nn_define, optimizer=self.optimizer, metrics=None,
                                           input_shape=input_shape)

        np.random.seed(114514)
        if self.role == consts.HOST:
            np_weight_w = np.random.normal(size=(634, self.nn._model.output_shape[1]))
            np_weight_w = np.random.normal(size=(self.nn._model.input_shape[1], self.nn._model.output_shape[1]))
        else:
            np_weight_w = np.random.normal(size=(self.nn._model.input_shape[1], self.nn._model.output_shape[1]))

        np_weight_b = np.zeros((self.nn._model.output_shape[1], ))
        LOGGER.debug('weights are {}, shape is {}'.format(np_weight_w, np_weight_w.shape))

        self.nn._model.set_weights([np_weight_w, np_weight_b])

        LOGGER.debug('printing nn layers structure')
        for layer in self.nn._model.layers:
            LOGGER.debug('input shape {}, output shape {}'.format(layer.input_shape, layer.output_shape))

    def generate_mask(self, shape):
        """
        generate random number mask
        """
        return self.rng_generator.generate_random_number(shape)

    def _batch_gradient_update(self, X, grads):
        """
        compute and update gradients for all samples
        """
        data = self.data_convertor.convert_data(X, grads)
        self.nn.train(data)

    def _get_mini_batch_gradient(self, X_batch, backward_grads_batch):

        """
        compute gradient for a mini batch
        """
        grads = self.nn.get_weight_gradients(X_batch, backward_grads_batch)
        return grads

    def update_nn_weights(self, backward_grads, data_loader: FTLDataLoader, epoch_idx, decay=False):

        """
        updating bottom nn model weights using backward gradients
        """

        LOGGER.debug('updating grads at epoch {}'.format(epoch_idx))

        assert len(data_loader.x) == len(backward_grads)

        weight_grads = []
        for i in range(len(data_loader)):
            start, end = data_loader.get_batch_indexes(i)
            batch_x = data_loader.x[start: end]
            batch_grads = backward_grads[start: end]
            batch_weight_grads = self._get_mini_batch_gradient(batch_x, batch_grads)
            if len(weight_grads) == 0:
                weight_grads.extend(batch_weight_grads)
            else:
                for w, bw in zip(weight_grads, batch_weight_grads):
                    w += bw

        if decay:
            new_learning_rate = self.learning_rate_decay(self.learning_rate, epoch_idx)
            self.nn.set_learning_rate(new_learning_rate)
            LOGGER.debug('epoch {} optimizer details are {}'.format(epoch_idx, self.nn.export_optimizer_config()))

        self.nn.apply_gradients(weight_grads)

    def export_nn(self):
        return self.nn.export_model()

    @staticmethod
    def get_dataset_key(data_inst):
        return data_inst.get_name(), data_inst.get_namespace()

    def get_model_meta(self):

        model_meta = FTLModelMeta()
        model_meta.config_type = self.config_type
        model_meta.nn_define = json.dumps(self.nn_define)
        model_meta.batch_size = self.batch_size
        model_meta.epochs = self.epochs
        model_meta.tol = self.tol
        model_meta.input_dim = self.input_dim

        predict_param = FTLPredictParam()

        optimizer_param = FTLOptimizerParam()
        optimizer_param.optimizer = self.optimizer.optimizer
        optimizer_param.kwargs = json.dumps(self.optimizer.kwargs)

        model_meta.optimizer_param.CopyFrom(optimizer_param)
        model_meta.predict_param.CopyFrom(predict_param)

        return model_meta

    def get_model_param(self):

        model_param = FTLModelParam()
        model_bytes = self.nn.export_model()
        model_param.model_bytes = model_bytes
        model_param.header.extend(list(self.store_header))

        return model_param

    def set_model_meta(self, model_meta):

        self.config_type = model_meta.config_type
        self.nn_define = json.loads(model_meta.nn_define)
        self.batch_size = model_meta.batch_size
        self.epochs = model_meta.epochs
        self.tol = model_meta.tol
        self.optimizer = FTLParam()._parse_optimizer(FTLParam().optimizer)
        self.input_dim = model_meta.input_dim

        self.optimizer.optimizer = model_meta.optimizer_param.optimizer
        self.optimizer.kwargs = json.loads(model_meta.optimizer_param.kwargs)

        self.initialize_nn((self.input_dim, ))

    def set_model_param(self, model_param):

        self.nn.restore_model(model_param.model_bytes)
        self.store_header = list(model_param.header)
        LOGGER.debug('stored header load, is {}'.format(self.store_header))







