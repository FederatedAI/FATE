from federatedml.model_interpret.federated_tree_shap import TreeSHAP
from federatedml.param.model_interpret_param import ModelInterpretParam
from federatedml.model_interpret.federated_kernel_shap import HeteroKernelSHAPGuest, HeteroKernelSHAPHost
from federatedml.util import consts, LOGGER


def run_explain(model, background_data, to_explain_data, param:ModelInterpretParam, role, flow_id):

    if param.method == 'shap':

        if role == consts.GUEST:
            shap = HeteroKernelSHAPGuest()

        else:
            shap = HeteroKernelSHAPHost()

        shap.load_model_inst(model)
        shap.set_flowid(flow_id)
        shap.load_param(param)
        rs = shap.fit(to_explain_data, background_data)
