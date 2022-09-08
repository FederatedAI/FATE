import importlib
from federatedml.model_base import ModelBase
from federatedml.custom_nn.nn_base_module import NNBaseModule
from federatedml.custom_nn.homo_nn_param import CustNNParam
from federatedml.util import LOGGER


def load_custom_nn_module(nn_class_name, class_file_name):
    nn_modules = importlib.import_module('federatedml.custom_nn.nn.{}'.format(class_file_name))

    try:
        nn_class = nn_modules.__dict__[nn_class_name]
        return nn_class()
    except ValueError as e:
        raise e


class HomoNNClient(ModelBase):

    def __init__(self):
        super(HomoNNClient, self).__init__()
        self.model_param = CustNNParam()
        self.cust_module = None

    def _init_model(self, param: CustNNParam):
        self.class_name = param.class_name
        self.class_file_name = param.class_file_name
        self.nn_params = param.nn_params
        self.cust_module: NNBaseModule = load_custom_nn_module(self.class_name, self.class_file_name)
        self.cust_module.set_flowid(self.flowid)
        self.cust_module.set_role(self.component_properties.role)
        if not issubclass(type(self.cust_module), NNBaseModule):
            raise ValueError('cust module must be the subclass of NNBaseModule')

        LOGGER.debug('model loaded {}'.format(self.cust_module))

    def fit(self, input_data):

        LOGGER.debug('self flow id is {}'.format(self.flowid))
        self.cust_module.train(cpn_input=input_data, **self.nn_params)

    def predict(self, input_data):

        return self.cust_module.predict(cpn_input=input_data, **self.nn_params)
