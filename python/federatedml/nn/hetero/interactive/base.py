import numpy as np

from federatedml.param.hetero_nn_param import HeteroNNParam
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables


class InteractiveLayerBase(object):

    def __init__(self, params: HeteroNNParam, **kwargs):
        self.params = params
        self.transfer_variable: BaseTransferVariables = None

    def set_flow_id(self, flow_id):
        if self.transfer_variable is not None:
            self.transfer_variable.set_flowid(flow_id)

    def set_batch(self, batch_size):
        pass

    def forward(self, x, epoch: int, batch_idx: int, train: bool = True, **kwargs) -> np.ndarray:
        pass

    def guest_backward(self, error, epoch: int, batch_idx: int, **kwargs):
        pass

    def host_backward(self, epoch: int, batch_idx: int, **kwargs):
        pass

    def export_model(self) -> bytes:
        pass

    def restore_model(self, model_bytes: bytes):
        pass

    def set_backward_select_strategy(self):
        pass
