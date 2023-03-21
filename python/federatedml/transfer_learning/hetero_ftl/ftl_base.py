import copy
import json
import functools
import numpy as np
from federatedml.util import LOGGER
from federatedml.transfer_learning.hetero_ftl.backend.nn_model import get_nn_builder
from federatedml.model_base import ModelBase
from federatedml.param.ftl_param import FTLParam
from federatedml.transfer_learning.hetero_ftl.backend.tf_keras.nn_model import KerasNNModel
from federatedml.util.classify_label_checker import ClassifyLabelChecker
from federatedml.transfer_variable.transfer_class.ftl_transfer_variable import FTLTransferVariable
from federatedml.transfer_learning.hetero_ftl.ftl_dataloder import FTLDataLoader
from federatedml.transfer_learning.hetero_ftl.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.nn.backend.utils import rng as random_number_generator
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.protobuf.generated.ftl_model_param_pb2 import FTLModelParam
from federatedml.protobuf.generated.ftl_model_meta_pb2 import FTLModelMeta, FTLPredictParam, FTLOptimizerParam


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
        self.config_type = 'keras'
        self.comm_eff = None
        self.local_round = 1

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
        self.encrypter = None
        self.partitions = 16
        self.batch_size = None
        self.epochs = None
        self.store_header = None  # header of input data table
        self.model_float_type = np.float32

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

        self.encrypter = self.generate_encrypter(param)
        self.predict_param = param.predict_param
        self.rng_generator = random_number_generator.RandomNumberGenerator()

    @staticmethod
    def debug_data_inst(data_inst):
        collect_data = list(data_inst.collect())
        LOGGER.debug('showing Table')
        for d in collect_data:
            LOGGER.debug('key {} id {}, features {} label {}'.format(d[0], d[1].inst_id, d[1].features, d[1].label))

    @staticmethod
    def reset_label(inst, mapping):
        new_inst = copy.deepcopy(inst)
        new_inst.label = mapping[new_inst.label]
        return new_inst

    @staticmethod
    def check_label(data_inst):
        """
        check label. FTL only supports binary classification, and labels should be 1 or -1
        """

        LOGGER.debug('checking label')
        label_checker = ClassifyLabelChecker()
        num_class, class_set = label_checker.validate_label(data_inst)
        if num_class != 2:
            raise ValueError(
                'ftl only support binary classification, however {} labels are provided.'.format(num_class))

        if 1 in class_set and -1 in class_set:
            return data_inst
        else:
            soreted_class_set = sorted(list(class_set))
            new_label_mapping = {soreted_class_set[1]: 1, soreted_class_set[0]: -1}
            reset_label = functools.partial(FTL.reset_label, mapping=new_label_mapping)
            new_table = data_inst.mapValues(reset_label)
            new_table.schema = copy.deepcopy(data_inst.schema)
            return new_table

    def generate_encrypter(self, param) -> PaillierEncrypt:
        LOGGER.info("generate encrypter")
        if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
            encrypter = PaillierEncrypt()
            encrypter.generate_key(param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet!!!")

        return encrypter

    def encrypt_tensor(self, components, return_dtable=True):
        """
        transform numpy array into Paillier tensor and encrypt
        """

        encrypted_tensors = []
        for comp in components:
            encrypted_tensor = PaillierTensor(comp, partitions=self.partitions)
            if return_dtable:
                encrypted_tensors.append(encrypted_tensor.encrypt(self.encrypter).get_obj())
            else:
                encrypted_tensors.append(encrypted_tensor.encrypt(self.encrypter))

        return encrypted_tensors

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
        if guest_side:
            data_inst = self.check_label(data_inst)

        overlap_samples = intersect_obj.run_intersect(data_inst)  # find intersect ids
        overlap_samples = intersect_obj.get_value_from_data(overlap_samples, data_inst)
        non_overlap_samples = data_inst.subtractByKey(overlap_samples)

        LOGGER.debug('num of overlap/non-overlap sampels: {}/{}'.format(overlap_samples.count(),
                                                                        non_overlap_samples.count()))

        if overlap_samples.count() == 0:
            raise ValueError('no overlap samples')

        if guest_side and non_overlap_samples == 0:
            raise ValueError('overlap samples are required in guest side')

        self.store_header = data_inst.schema['header']
        LOGGER.debug('data inst header is {}'.format(self.store_header))

        LOGGER.debug('has {} overlap samples'.format(overlap_samples.count()))

        batch_size = self.batch_size
        if self.batch_size == -1:
            batch_size = data_inst.count() + 1  # make sure larger than sample number
        data_loader = FTLDataLoader(non_overlap_samples=non_overlap_samples,
                                    batch_size=batch_size, overlap_samples=overlap_samples, guest_side=guest_side)

        LOGGER.debug("data details are :{}".format(data_loader.data_basic_info()))

        return data_loader, data_loader.x_shape, data_inst.count(), len(data_loader.get_overlap_indexes())

    def get_model_float_type(self, nn):
        weights = nn.get_trainable_weights()
        self.model_float_type = weights[0].dtype

    def initialize_nn(self, input_shape):
        """
        initializing nn weights
        """

        loss = "keep_predict_loss"
        self.nn_builder = get_nn_builder(config_type=self.config_type)
        self.nn = self.nn_builder(loss=loss, nn_define=self.nn_define, optimizer=self.optimizer, metrics=None,
                                  input_shape=input_shape)
        self.get_model_float_type(self.nn)

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
        X_batch = X_batch.astype(self.model_float_type)
        backward_grads_batch = backward_grads_batch.astype(self.model_float_type)
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
        return id(data_inst)

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

        self.initialize_nn((self.input_dim,))

    def set_model_param(self, model_param):

        self.nn.restore_model(model_param.model_bytes)
        self.store_header = list(model_param.header)
        LOGGER.debug('stored header load, is {}'.format(self.store_header))
