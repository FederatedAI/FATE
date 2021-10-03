import copy
import numpy as np

from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.norm_data_transfer_variable import NormDataTransferVariable
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.util import abnormal_detection
from federatedml.param.norm_data_param import NormDataParam
from federatedml.feature.instance import Instance



class NormDataGuest(ModelBase):
    def __init__(self):
        super(NormDataGuest, self).__init__()
        self.transfer_variable = NormDataTransferVariable()
        #这一步必不可少，即使你并未给model_param赋值
        self.model_param = NormDataParam()
        self.norm_x = None

    def _abnormal_detection(self,data_instances):
        """检查输入的数据是否有效"""
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        ModelBase.check_schema_content(data_instances.schema)

    @staticmethod
    def load_data(data_instance):
        """
        分别将标签数据设置为1和-1
        """
        data_instance = copy.deepcopy(data_instance)
        if data_instance.label != 1:
            data_instance.label = -1
        return data_instance

    def fit(self, data_inst):
        """
        正式开始处理数据
        """
        self._abnormal_detection(data_inst)
        data_instances = data_inst.mapValues(self.load_data)

        LOGGER.info("开始归一化数据")
        LOGGER.info("开始计算sum_square_x_guest的结果")
        sum_square_x_guest = data_instances.mapValues(lambda x : np.sum(np.power(x.features,2)))

        LOGGER.info("从host方接收sum_square_x_host的值")
        sum_square_x_host = self.transfer_variable.host_to_guest.get(idx=-1,suffix=(0,0))

        LOGGER.info("开始求平方根norm_x的值")
        self.norm_x = sum_square_x_guest.join(sum_square_x_host[0],lambda g,h : (g + h) ** 0.5)

        LOGGER.info("将norm_x发送给对方")
        self.transfer_variable.guest_to_host.remote(self.norm_x,role=consts.HOST,idx=-1,suffix=(1,1))

        LOGGER.info("正式归一化")
        self.data_output = data_inst.join(self.norm_x,lambda x,y : Instance(features=np.true_divide(x.features,y),label=x.label))
        return self.data_output

    def save_data(self):
        return self.data_output





