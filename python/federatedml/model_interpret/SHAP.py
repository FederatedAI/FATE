import copy
from federatedml.model_base import ModelBase
from federatedml.param.shap_param import SHAPParam
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.model_interpret.model_discriminator import model_discriminator
from federatedml.model_interpret.explainer.explainer_base import Explainer
from federatedml.model_interpret.explainer.tree_explainer.tree_shap import HeteroTreeSHAP, HomoTreeSHAP
from federatedml.model_interpret.explainer.kernel_explainer.kernel_shap import HeteroKernelSHAP, HomoKernelSHAP


class SHAP(ModelBase):

    def __init__(self):

        super(SHAP, self).__init__()
        self.model_param = SHAPParam()
        self.explainer: Explainer = None
        self.ref_type = None
        self.explain_all = True
        self.interpret_limit = 1

    def _init_model(self, param: SHAPParam):
        self.ref_type = param.reference_type
        self.explain_all = param.explain_all_host_feature

    @staticmethod
    def _get_model_type(model_dict):
        """
        hetero or homo ?
        """
        sbt_key_prefix = consts.HETERO_SBT_GUEST_MODEL.replace('Guest', '')
        homo_sbt_key_prefix = consts.HOMO_SBT_GUEST_MODEL.replace('Guest', '')
        for key in model_dict:
            for model_key in model_dict[key]:
                if sbt_key_prefix in model_key:
                    return consts.HETERO_SBT
                elif homo_sbt_key_prefix in model_key:
                    return consts.HOMO_SBT

        return None

    """
    Model IO
    """

    def load_model(self, model_dict):

        """
        Load Model from isometric model, and set explainer
        """

        model_name, meta, param = model_discriminator(model_dict)
        #
        # if model_name == consts.HETERO_SBT:
        #     self.explainer = HeteroTreeSHAP(self.role, self.flowid)
        #     self.explainer.set_component_properties(copy.deepcopy(self.component_properties))
        #     if self.explain_all:
        #         self.explainer.set_full_explain()
        # elif model_name == consts.HOMO_SBT:
        #     self.explainer = HomoTreeSHAP(self.role, self.flowid)
        # self.explainer.init_model(meta, param)

        # LOGGER.debug('model dict is {}'.format(model_dict))
        from federatedml.model_interpret.model_adaptor import HeteroModelAdaptor, HomoModelAdaptor
        from federatedml.linear_model.logistic_regression.homo_logistic_regression.homo_lr_guest import HomoLRGuest
        from federatedml.nn.homo_nn.enter_point import HomoNNClient

        fate_model = HeteroModelAdaptor(self.role, meta, param, self.component_properties, self.flowid)
        self.explainer = HeteroKernelSHAP(self.role, self.flowid)
        self.explainer.set_component_properties(copy.deepcopy(self.component_properties))
        self.explainer.init_model(fate_model)
        if self.explain_all:
            self.explainer.set_full_explain()

        # self.explainer = HomoKernelSHAP(self.role, self.flowid)
        # homo_model = HomoModelAdaptor(model_dict, HomoNNClient)
        # self.explainer.init_model(homo_model)
        # LOGGER.debug('homo model is {}'.format(homo_model))
        # # LOGGER.info('using explainer {}, role is {}'.format(self.explainer, self.role))

    """
    fit
    """

    def fit(self, data_inst):

        LOGGER.debug('self role is {}'.format(self.role))
        import time
        s = time.time()
        explain_rs = self.explainer.explain(data_inst, self.interpret_limit)
        # interaction_rs = self.explainer.explain_interaction(data_inst, self.interpret_limit)
        e = time.time()
        LOGGER.debug('running takes {}'.format(e-s))
