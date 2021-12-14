import copy
from federatedml.model_base import ModelBase
from federatedml.param.shap_param import SHAPParam
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.model_interpret.model_discriminator import extract_model, reconstruct_model_dict
from federatedml.model_interpret.explainer.explainer_base import Explainer
from federatedml.model_interpret.model_adaptor import HeteroModelAdaptor, HomoModelAdaptor
from federatedml.model_interpret.explainer.tree_shap_explainer import HeteroTreeSHAP, HomoTreeSHAP
from federatedml.model_interpret.explainer.kernel_shap_explainer import HeteroKernelSHAP, HomoKernelSHAP
from federatedml.ensemble import HeteroSecureBoostingTreeGuest, HeteroSecureBoostingTreeHost, HomoSecureBoostingTreeClient
from federatedml.linear_model.linear_model_base import BaseLinearModel


class SHAP(ModelBase):

    def __init__(self):

        super(SHAP, self).__init__()
        self.model_param = SHAPParam()
        self.explainer: Explainer = None
        self.ref_type = None
        self.explain_all = True
        self.interpret_limit = 10

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

        if self.role == consts.ARBITER:
            # arbiter quit
            return
        assert 'isometric_model' in model_dict, 'Did not find key "isometric". Input model must be an isometric model.'
        meta, param, module_name, algo_inst = extract_model(model_dict, self.role)
        # isometric model -> model
        new_model_dict = reconstruct_model_dict(model_dict)

        if consts.HOMO in module_name.lower():
            fed_type = consts.HOMO
        elif consts.HETERO in module_name.lower():
            fed_type = consts.HETERO
        else:
            raise ValueError('illegal algo module: {}'.format(module_name))
        LOGGER.debug('fed type is {}'.format(fed_type))

        if fed_type == consts.HETERO:
            # tree models use TreeSHAP only
            if type(algo_inst) == HeteroSecureBoostingTreeHost or type(algo_inst) == HeteroSecureBoostingTreeGuest:
                self.explainer = HeteroTreeSHAP(self.role, self.flowid)
                # init tree model
                self.explainer.init_model(new_model_dict, copy.deepcopy(self.component_properties))
            else:
                fate_model = HeteroModelAdaptor(self.role, new_model_dict, algo_inst, self.component_properties,
                                                self.flowid)
                self.explainer = HeteroKernelSHAP(self.role, self.flowid)
                self.explainer.init_model(fate_model, copy.deepcopy(self.component_properties))

            if issubclass(type(algo_inst), BaseLinearModel):
                LOGGER.debug('linear model detected, not support full explain')
                self.explain_all = False

            if self.explain_all:
                self.explainer.set_full_explain()

        elif fed_type == consts.HOMO:

            # tree models use TreeSHAP only
            if type(algo_inst) == HomoSecureBoostingTreeClient:
                self.explainer = HomoTreeSHAP(self.role, self.flowid)
                self.explainer.init_model(meta, param)
            else:
                self.explainer = HomoKernelSHAP(self.role, self.flowid)
                homo_model = HomoModelAdaptor(new_model_dict, algo_inst)
                self.explainer.init_model(homo_model)
                LOGGER.debug('homo model is {}'.format(homo_model))

        LOGGER.info('using explainer {}, role is {}'.format(self.explainer, self.role))

    """
    fit
    """

    def fit(self, data_inst):

        # arbiter quit
        if self.role == consts.ARBITER:
            return

        LOGGER.debug('self role is {}'.format(self.role))
        import time
        s = time.time()
        explain_rs = self.explainer.explain(data_inst, self.interpret_limit)
        # interaction_rs = self.explainer.explain_interaction(data_inst, self.interpret_limit)
        e = time.time()
        LOGGER.debug('running takes {}'.format(e-s))
