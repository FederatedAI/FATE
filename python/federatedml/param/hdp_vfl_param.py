from federatedml.param.base_param import BaseParam



class HdpVflParam(BaseParam):
    """
    HDP-VFL将会使用的参数
    Parameters
    ---------------------
    有关于差分隐私的参数设置
    epsilon : float,默认值0.1
    delta : float,高斯噪声允许的范围差，默认值0.1
    以下几个参数与差分隐私有关，但是暂时未知它们的含义。因为如下参数都是位于公式中的，可以直接赋值。几何意义未知
    L : int,利普希茨常数，这里默认值为1
    beta_theta : float,默认值是0.25
    beta_y : float,默认值是1.1



    逻辑回归算法常用参数
    e : int,epochs,数据集重复了多少次，默认值50
    r : int,number of mini-batches,指的是对于一次完整的数据集，应当经过的小批量的次数
    k : int,这里为了防止梯度爆炸设置的梯度剪切，默认值为1
    learning_rate : float,学习率，默认是0.05
    lamb : float,正则化系数，默认值未知0.001

    其他
    k_y : int,target bound,默认值是1




    """
    def __init__(self,epsilon=0.1,delta=0.1,L=1,beta_theta=0.25,beta_y=1.1,
                 e=50,r=5,k=1,learning_rate=0.05,lamb=0.001,k_y=1):
        super(HdpVflParam,self).__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.L = L
        self.beta_theta = beta_theta
        self.beta_y = beta_y
        self.e = e
        self.r = r
        self.k = k
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.k_y = k_y

    def check(self):
        """
        主要用来检查赋值的参数是否有问题
        """
        if type(self.epsilon).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数epsilon{}不支持，应当是浮点数".format(self.epsilon))

        if type(self.delta).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数delta{}不支持，应当是浮点数".format(self.delta))

        if type(self.L).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数L{}不支持，应当是浮点数".format(self.L))

        if type(self.beta_theta).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数beta_theta{}不支持，应当是浮点数".format(self.beta_theta))

        if type(self.beta_y).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数beta_y{}不支持，应当是浮点数".format(self.beta_y))

        if type(self.e).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数e{}不支持，应当是浮点数".format(self.e))

        if type(self.r).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数r{}不支持，应当是浮点数".format(self.r))

        if type(self.k).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数k{}不支持，应当是浮点数".format(self.k))

        if type(self.learning_rate).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数learning_rate{}不支持，应当是浮点数".format(self.learning_rate))

        if type(self.lamb).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数lamb{}不支持，应当是浮点数".format(self.lamb))

        if type(self.k_y).__name__ not in ["float","int"]:
            raise ValueError("hdp-vfl的参数k_y{}不支持，应当是浮点数".format(self.k_y))
