import numpy as np

from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.feature.feature_selection.model_adapter.adapter_base import BaseAdapter
from federatedml.util import consts


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
    local_fids = list(fid_mapping.keys())
    local_cols, local_val = [], []

    # key is int party id, value is a dict, which has two key: col_name and value
    host_side_data = {}

    for feat_importance in feat_importance_list:
        fid = feat_importance.fid
        importance = feat_importance.importance
        site_name = feat_importance.sitename
        if site_name == consts.HOST_LOCAL:
            local_cols.append(fid_mapping[fid])
            local_val.append(importance)
        else:
            site_name = site_name.split(':')
            if site_name[0] == consts.HOST:
                continue
            else:
                local_cols.append(fid_mapping[fid])
                local_val.append(importance)

    for fid in local_fids:
        if fid_mapping[fid] not in local_cols:
            local_cols.append(fid_mapping[fid])
            local_val.append(0)

    single_info = isometric_model.SingleMetricInfo(
        values=np.array(local_val),
        col_names=local_cols
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
