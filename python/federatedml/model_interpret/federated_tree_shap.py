import numpy as np
import lightgbm as lgb
from federatedml.model_base import ModelBase
from federatedml.protobuf.model_migrate.sbt_model_to_lgb import sbt_to_lgb
from federatedml.param.shap_param import TreeSHAPParam
from federatedml.util import LOGGER
from federatedml.util import consts


def mock_hetero_tree_node(tree_param):
    pass


def extract_route_from_host_nodes(tree_param):
    pass


class TreeSHAP(ModelBase):

    def __init__(self):

        super(TreeSHAP, self).__init__()
        self.interpret_limit = 10
        self.tree_param = None
        self.tree_meta = None
        self.run_mode = consts.HOMO
        self.model_param = TreeSHAPParam()

    def _init_model(self, param):
        self.interpret_limit = param.interpret_limit

    def convert_homo_sbt_to_lgb(self):
        model_str = sbt_to_lgb(self.tree_param, self.tree_meta)
        lgb_model = lgb.Booster(model_str=model_str)
        return lgb_model

    @staticmethod
    def _get_model_type(model_dict):
        """
        fast-sbt or sbt ? hetero or homo ?
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

    def make_output_rs(self, lgb_model, pred_contrib):
        pass

    def make_predict_format(self, to_interpret_inst):

        ids = []
        data_list = []
        for id_, inst in to_interpret_inst:
            ids.append(id_)
            data_list.append(inst.features)
        data_arr = np.array(data_list)
        return ids, data_arr

    def extract_host_route(self, to_interpret_inst):
        pass

    def homo_fit(self, data_inst):

        to_interpret_inst = data_inst.take(self.interpret_limit)
        lgb_model = self.convert_homo_sbt_to_lgb()
        ids, data_arr = self.make_predict_format(to_interpret_inst)
        contrib = lgb_model.predict(data_arr, pred_contrib=True)
        LOGGER.debug('to interpret inst {}'.format(data_arr))
        LOGGER.debug('contrib is {}'.format(contrib))
        # interpret_result = lgb.predict(to_interpret_inst)

    def hetero_fit(self, data_inst):
        pass

    def load_model(self, model_dict):

        key = 'isometric_model'
        model_type = self._get_model_type(model_dict[key])
        if model_type == consts.HOMO_SBT:
            self.run_mode = consts.HOMO
        elif model_type == consts.HETERO_SBT:
            self.run_mode = consts.HETERO_SBT
        else:
            raise ValueError('illegal input model: {}'.format(model_dict))

        for model_key in model_dict[key]:
            model_content = model_dict[key][model_key]
            LOGGER.debug('model content {}'.format(model_content))
            for content_name in model_content:
                if 'Meta' in content_name:
                    self.tree_meta = model_content[content_name]
                elif 'Param' in content_name:
                    self.tree_param = model_content[content_name]

    def fit(self, data_inst):

        if self.run_mode == consts.HOMO:
            LOGGER.debug('running homo tree shap')
            self.homo_fit(data_inst)
        elif self.run_mode == consts.HETERO:
            pass

