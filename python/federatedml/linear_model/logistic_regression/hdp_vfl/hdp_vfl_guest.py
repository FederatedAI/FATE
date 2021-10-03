import copy

from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.hdp_vfl_transfer_variable import HdpVflTransferVariable
from federatedml.linear_model.logistic_regression.hdp_vfl.batch_data import Guest
from federatedml.util import LOGGER
from federatedml.param import hdp_vfl_param
from federatedml.util import abnormal_detection
from federatedml.linear_model.linear_model_weight import LRModelWeightsGuest
from federatedml.util import consts

class HdpVflGuest(ModelBase):
    def __init__(self):
        """
        如果参数和模型参数无关，只与训练过程有关，那么就放在这里
        """
        super().__init__()
        self.batch_generator = Guest()
        self.model_param = hdp_vfl_param.HdpVflParam()
        self.transfer_variable = HdpVflTransferVariable()
        #用来存最终的模型参数
        self.model = None
        self.ir_a = None
        self.ir_b = None

    def _init_model(self, params):
        """
        这里主要是将HdpVflGuest相关的参数进行赋值，所以相关的属性都放在了这里
        """
        self.epsilon = params.epsilon
        self.delta = params.delta
        self.L = params.L
        self.beta_theta = params.beta_theta
        self.beta_y = params.beta_y
        self.e = params.e
        #这里的r指的是对于一次完整的数据集，应当经历的小批量的次数。所以每次的小批量应当等于总数据量除以r
        self.r = params.r
        self.k = params.k
        self.learning_rate = params.learning_rate
        self.lamb = params.lamb
        self.k_y = params.k_y

    @staticmethod
    def load_data(data_instance):
        """
        设置数据标签为1或者-1
        """
        data_instance = copy.deepcopy(data_instance)
        if data_instance.label != 1:
            data_instance.label = -1
        return data_instance

    def _abnormal_detection(self,data_instances):
        """
        主要用来检查数据的有效性
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        ModelBase.check_schema_content(data_instances.schema)

    def register_gradient_sync(self,transfer_variable):
        self.ir_a = transfer_variable.ir_a
        self.ir_b = transfer_variable.ir_b

    def fit(self, data_instances):
        LOGGER.info("开始纵向逻辑回归")
        #检查数据
        self._abnormal_detection(data_instances)
        #导入数据
        data_instances = data_instances.mapValues(HdpVflGuest.load_data)

        data_test = self.transfer_variable.ir_a
        #下面开始模型的初始化
        self.model = LRModelWeightsGuest()
        self.model.initialize(data_instances)

        #批处理数据模块初始化
        self.batch_generator.register_batch_generator(self.transfer_variable)
        #这里的self.r指的是对于一个完整的数据集，小批量的次数
        batch_size = int(data_instances.count() / self.r )
        suffix = (data_instances.count(),self.r)
        self.batch_generator.initialize_batch_generator(data_instances,batch_size,suffix=suffix)

        # 传输变量初始化
        self.register_gradient_sync(self.transfer_variable)

        #开始正式的循环迭代训练过程,初始化迭代次数为0
        iteration = 0
        test_suffix = ("iter",)
        while iteration <= self.e:
            #获取当前批次的数据
            for data_inst in self.batch_generator.generator_batch_data():
                LOGGER.info("从host端接收sec_ir_b")
                suffix_t = test_suffix + (iteration,)
                sec_ir_b = self.ir_b.get(suffix=suffix_t)

                LOGGER.info("开始计算ir_a")
                ir_a = self.model.intermediate_result(data_inst,sec_ir_b[0],self.model.w)

                LOGGER.info("开始计算高斯噪声所需要的loc、sigma")
                loc,sigma = self.model.gaussian(self.delta,self.epsilon,self.beta_theta,self.L,
                                                self.e,int(self.e * self.r),self.learning_rate,
                                                data_inst.count(),self.k,self.beta_y,self.k_y)

                LOGGER.info("开始求sec_ir_a的值")
                sec_ir_a = self.model.sec_intermediate_result(ir_a,loc,sigma)

                LOGGER.info("开始将sec_ir_a发送给host方")
                self.ir_a.remote(obj=sec_ir_a,role=consts.HOST,idx=-1,suffix=suffix_t)

                LOGGER.info("开始计算梯度gradient_a")
                gradient_a = self.model.compute_gradient(data_inst,ir_a,data_inst.count())

                LOGGER.info("开始更新模型参数w")
                self.model.update_model(gradient_a,self.learning_rate,self.lamb)

                LOGGER.info("开始梯度剪切")
                self.model.norm_clip(self.k)

                iteration += 1

        LOGGER.info("训练正式结束")
        LOGGER.info("guest方的模型参数是：{}".format(self.model.w))

        return self.model.w

    def save_data(self):
        return self.model.w


