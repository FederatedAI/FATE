import torch
import copy
import numpy as np
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.ensemble import HeteroSecureBoostingTreeGuest, HeteroSecureBoostingTreeHost, HomoSecureBoostingTreeClient
from federatedml.linear_model.logistic_regression.homo_logistic_regression.homo_lr_guest import HomoLRGuest
from federatedml.nn.homo_nn.enter_point import HomoNNClient
from federatedml.protobuf.homo_model_convert.sklearn.logistic_regression import LRComponentConverter
from federatedml.protobuf.homo_model_convert.pytorch.nn import NNComponentConverter
from federatedml.protobuf.homo_model_convert.lightgbm.gbdt import HomoSBTComponentConverter


SHAP_FLOW_ID_PREFIX = 'shap'


class HomoModelAdaptor(object):

    def __init__(self, model_dict, algo_class):

        model_content = None

        for key in model_dict['isometric_model']:
            model_content = model_dict['isometric_model'][key]
            break

        self.homo_model = None
        self.predict_func = None
        if algo_class == HomoSecureBoostingTreeClient:
            self.homo_model = HomoSBTComponentConverter().convert(model_content)

        elif algo_class == HomoLRGuest:
            self.homo_model = LRComponentConverter().convert(model_content)
            def pred_func(x):
                return self.homo_model.predict_proba(x)[::, 1]
            self.predict_func = pred_func

        elif algo_class == HomoNNClient:
            self.homo_model = NNComponentConverter().convert(model_content)
            LOGGER.debug('homo model is {}'.format(self.homo_model))
            def pred_func(x):
                tensor_rs = self.homo_model(torch.Tensor(x))
                return tensor_rs.detach().numpy().reshape((x.shape[0], -1))
            self.predict_func = pred_func

    def predict(self, arr: np.ndarray):
        return self.predict_func(arr)

    def __repr__(self):
        return 'Homo Adaptor with {}'.format(self.homo_model)


class HeteroModelAdaptor(object):

    def __init__(self, role, meta, param, component_properties, flow_id=None):

        self.fate_model = None
        self.predict_suffix = 0
        self.cur_flow_id = ''

        if role == consts.HOST:
            boosting_model = HeteroSecureBoostingTreeHost()
        elif role == consts.GUEST:
            boosting_model = HeteroSecureBoostingTreeGuest()
        else:
            raise ValueError('unknown role {}'.format(role))

        boosting_model.set_model_meta(meta)
        boosting_model.set_model_param(param)
        boosting_model.role = role
        boosting_model.component_properties = copy.deepcopy(component_properties)

        if flow_id is not None:
            self.cur_flow_id = flow_id + '.' + SHAP_FLOW_ID_PREFIX

        self.fate_model = boosting_model

    def predict(self, data_inst):
        self.fate_model.set_flowid(self.cur_flow_id + '.' + str(self.predict_suffix))
        self.predict_suffix += 1
        return self.fate_model.predict(data_inst)


