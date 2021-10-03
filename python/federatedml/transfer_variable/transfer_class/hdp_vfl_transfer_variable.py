from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables



class HdpVflTransferVariable(BaseTransferVariables):
    """
    Parameters
    ----------------
    batch_data_index : 这个参数主要用来同步双方的小批量数据
    batch_info:传输当前批次的信息
    ir_a : 加入噪声的中间结果，这里是梯度的一部分。由guest方发给host方
    ir_b : 加入噪声的中间结果，这里是host方数据和参数的内积，然后添加噪声，由host方发给guest方

    其他参数后续再添加

    """
    def __init__(self,flowid=0):
        super().__init__(flowid)
        self.batch_data_index = self._create_variable(name="batch_data_index",src=["guest"],dst=["host"])
        self.batch_info = self._create_variable(name="batch_info",src=["guest"],dst=["host"])
        self.ir_b = self._create_variable(name="ir_b", src=["host"], dst=["guest"])
        self.ir_a = self._create_variable(name="ir_a",src=["guest"],dst=["host"])


