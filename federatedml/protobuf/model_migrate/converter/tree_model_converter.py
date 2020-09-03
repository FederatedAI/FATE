from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from typing import Dict
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.util import consts


class AutoReplace(object):

    def __init__(self, guest_mapping, host_mapping, arbiter_mapping):
        self.g_map = guest_mapping
        self.h_map = host_mapping
        self.a_map = arbiter_mapping

    def map_finder(self, sitename):
        if consts.GUEST == sitename:
            return self.g_map
        elif consts.HOST == sitename:
            return self.h_map
        elif consts.ARBITER in sitename:
            return self.a_map
        else:
            raise ValueError('this sitename contains no site name {}'.format(sitename))

    def anonymous_format(self, string):

        sitename, party_id, idx = string.split('_')
        mapping = self.map_finder(sitename)
        new_party_id = mapping[int(party_id)]
        return sitename + '_' + str(new_party_id) + '_' + idx

    def colon_format(self, string: str):
        sitename, party_id = string.split(':')
        mapping = self.map_finder(sitename)
        new_party_id = mapping[int(party_id)]
        return sitename + ':' + str(new_party_id)

    def replace(self, string):

        if ':' in string:
            return self.colon_format(string)
        elif '_' in string:
            return self.anonymous_format(string)
        else:
            # nothing to replace
            return string


class HeteroSBTConverter(ProtoConverterBase):

    def convert(self, param: BoostingTreeModelParam, meta: BoostingTreeModelMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ):

        feat_importance_list = list(param.feature_importances)
        tree_list = list(param.trees_)
        replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)

        # fp == feature importance
        for fp in feat_importance_list:
            fp.sitename = replacer.replace(fp.sitename)
            fp.fullname = replacer.replace(fp.fullname)

        for tree in tree_list:
            tree_nodes = list(tree.tree_)
            for node in tree_nodes:
                node.sitename = replacer.replace(node.sitename)

        return param, meta
