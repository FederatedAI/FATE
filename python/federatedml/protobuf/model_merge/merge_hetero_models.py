import copy
import tempfile
import json
import pickle
import base64
from federatedml.protobuf.model_merge.merge_sbt import merge_sbt
from federatedml.protobuf.model_merge.merge_hetero_lr import merge_lr
from nyoka import lgb_to_pmml
from sklearn2pmml import sklearn2pmml


def get_pmml_str(pmml_pipeline, target_name):
    tmp_f = tempfile.NamedTemporaryFile()
    path = tmp_f.name
    lgb_to_pmml(pmml_pipeline, pmml_pipeline['lgb'].feature_name_, target_name, path)
    with open(path, 'r') as read_f:
        str_ = read_f.read()
    tmp_f.close()
    return str_


def output_sklearn_pmml_str(pmml_pipeline, ):
    tmp_f = tempfile.NamedTemporaryFile()
    path = tmp_f.name
    sklearn2pmml(pmml_pipeline, path, with_repr=True)
    with open(path, 'r') as read_f:
        str_ = read_f.read()
    tmp_f.close()
    return str_


def hetero_model_merge(guest_param: dict, guest_meta: dict, host_params: list, host_metas: list, model_type: str,
                       output_format: str, target_name: str = 'y', host_rename=False, include_guest_coef=False):
    """
    Merge a hetero model
    :param guest_param: a json dict contains guest model param
    :param guest_meta: a json dict contains guest model meta
    :param host_params: a list contains json dicts of host params
    :param host_metas: a list contains json dicts of host metas
    :param model_type: specify the model type:
                       secureboost, alias tree, sbt
                       logistic_regression, alias LR
    :param output_format: output format of merged model, support:
                          lightgbm, for tree models only
                          sklearn, for linear models only
                          pmml, for all types
    :param target_name: if output format is pmml, need to specify the targe(label) name
    :param host_rename: add suffix to secureboost host features
    :param include_guest_coef: default False

    :return: Merged Model Class
    """
    guest_param = copy.deepcopy(guest_param)
    guest_meta = copy.deepcopy(guest_meta)
    host_params = copy.deepcopy(host_params)
    host_metas = copy.deepcopy(host_metas)

    if not isinstance(model_type, str):
        raise ValueError('model type should be a str, but got {}'.format(model_type))

    if output_format.lower() not in {'lightgbm', 'lgb', 'sklearn', 'pmml'}:
        raise ValueError('unknown output format: {}'.format(output_format))

    if model_type.lower() in ['secureboost', 'tree', 'sbt']:
        model = merge_sbt(guest_param, guest_meta, host_params, host_metas, output_format, target_name,
                          host_rename=host_rename)
        if output_format == 'pmml':
            return get_pmml_str(model, target_name)
        else:
            return model

    elif model_type.lower() in {'logistic_regression', 'lr'}:
        model = merge_lr(guest_param, guest_meta, host_params, host_metas, output_format, include_guest_coef)
        if output_format == 'pmml':
            return output_sklearn_pmml_str(model)
        else:
            return json.dumps(str(base64.b64encode(pickle.dumps(model)), "utf-8"))
    else:
        raise ValueError('model type should be one in ["sbt", "lr"], '
                         'but got unknown model type: {}'.format(model_type))
