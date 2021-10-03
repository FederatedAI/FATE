from federatedml.param.base_param import BaseParam

class NormDataParam(BaseParam):
    def __init__(self,test_1 = 1,test_2 = 2):
        super(NormDataParam, self).__init__()
        self.test_1 = test_1
        self.test_2 = test_2
    #当前算法组件不需要参数
    def check(self):
        if type(self.test_1).__name__ != "int":
            raise ValueError("这里用作测试")
        if type(self.test_2).__name__ != "int":
            raise ValueError("这里用作测试数据")