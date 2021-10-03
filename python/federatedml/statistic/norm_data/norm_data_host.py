import  copy
import numpy as np

from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.norm_data_transfer_variable import NormDataTransferVariable
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.util import abnormal_detection
from federatedml.feature.instance import Instance
from federatedml.param.norm_data_param import NormDataParam

class NormDataHost(ModelBase):
    def __init__(self):
        super(NormDataHost, self).__init__()
        self.transfer_variable = NormDataTransferVariable()
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
        设置数据标签为1和-1
        """
        data_instance = copy.deepcopy(data_instance)
        if data_instance.label != 1:
            data_instance.label = -1
        return data_instance

    def fit(self, data_inst):
        """
        开始正式处理数据
        """
        self._abnormal_detection(data_inst)
        data_instances = data_inst.mapValues(self.load_data)

        LOGGER.info("开始归一化数据")
        LOGGER.info("开始计算sum_square_x_host的结果")
        sum_square_x_host = data_instances.mapValues(lambda x : np.sum(np.power(x.features,2)))

        LOGGER.info("将sum_square_x_host发送给guest方")
        self.transfer_variable.host_to_guest.remote(obj=sum_square_x_host,role=consts.GUEST,idx=-1,suffix=(0,0))

        LOGGER.info("从guest方接收结果norm_x")
        self.norm_x = self.transfer_variable.guest_to_host.get(idx=-1,suffix=(1,1))

        LOGGER.info("开始归一化数据")
        self.data_output = data_inst.join(self.norm_x[0],lambda x,y : Instance(features=np.true_divide(x.features,y),label=x.label))
        return self.data_output

    def save_data(self):
        return self.data_output




