import numpy as np

from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.feature.feature_selection.model_adapter.adapter_base import BaseAdapter
from federatedml.util import consts
# from federatedml.util.fate_operator import generate_anonymous
from federatedml.util.anonymous_generator import generate_anonymous


def feature_importance_converter(model_meta, model_param):

    # extract feature importance from model param
    fid_mapping = dict(model_param.feature_name_fid_mapping)
    feat_importance_list = list(model_param.feature_importances)
    fids = list(fid_mapping.keys())

    cols_names, importance_val = [], []

    for feat_importance in feat_importance_list:
        fid = feat_importance.fid
        importance = feat_importance.importance
        feature_name = fid_mapping[fid]
        cols_names.append(feature_name)
        importance_val.append(importance)

    for fid in fids:
        if fid_mapping[fid] not in cols_names:
            cols_names.append(fid_mapping[fid])
            importance_val.append(0)

    single_info = isometric_model.SingleMetricInfo(
        values=np.array(importance_val),
        col_names=cols_names
    )
    result = isometric_model.IsometricModel()
    result.add_metric_value(metric_name=consts.FEATURE_IMPORTANCE, metric_info=single_info)

    return result


def feature_importance_with_anonymous_converter(model_meta, model_param):

    # extract feature importance from model param

    fid_mapping = dict(model_param.feature_name_fid_mapping)
    feat_importance_list = list(model_param.feature_importances)
    guest_fids = list(fid_mapping.keys())
    guest_cols, guest_val = [], []

    # key is int party id, value is a dict, which has two key: col_name and value
    host_side_data = {}

    for feat_importance in feat_importance_list:
        fid = feat_importance.fid
        importance = feat_importance.importance
        site_name = feat_importance.sitename
        site_name = site_name.split(':')
        if site_name[0] == consts.HOST:
            host_id = int(site_name[1])
            if host_id not in host_side_data:
                host_side_data[host_id] = {'col_name': [], 'value': []}
            host_col_name = generate_anonymous(fid, host_id, role=consts.HOST)
            host_side_data[host_id]['col_name'].append(host_col_name)
            host_side_data[host_id]['value'].append(importance)
        else:
            guest_cols.append(fid_mapping[fid])
            guest_val.append(importance)

    for fid in guest_fids:
        if fid_mapping[fid] not in guest_cols:
            guest_cols.append(fid_mapping[fid])
            guest_val.append(0)

    host_party_ids = []
    host_values = []
    host_col_names = []
    for hid in host_side_data:
        host_party_ids.append(hid)
        host_values.append(host_side_data[hid]['value'])
        host_col_names.append(host_side_data[hid]['col_name'])

    single_info = isometric_model.SingleMetricInfo(
        values=np.array(guest_val),
        col_names=guest_cols,
        host_party_ids=host_party_ids,
        host_values=host_values,
        host_col_names=host_col_names
    )
    result = isometric_model.IsometricModel()
    result.add_metric_value(metric_name=consts.FEATURE_IMPORTANCE, metric_info=single_info)
    return result


class HomoSBTAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):
        return feature_importance_converter(model_meta, model_param)


class HeteroSBTAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):
        return feature_importance_with_anonymous_converter(model_meta, model_param)


class HeteroFastSBTAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):
        model_name = model_param.model_name

        if model_name == consts.HETERO_FAST_SBT_LAYERED:
            return feature_importance_with_anonymous_converter(model_meta, model_param)
        elif model_name == consts.HETERO_FAST_SBT_MIX:
            return feature_importance_converter(model_meta, model_param)
        else:
            raise ValueError('model name {} is illegal'.format(model_name))