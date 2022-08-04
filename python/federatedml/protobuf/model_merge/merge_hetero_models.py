from federatedml.protobuf.model_merge.merge_sbt import merge_sbt
from nyoka import lgb_to_pmml
import copy


class MergedModel(object):

    def __init__(self, model, model_type, output_format, target_name):
        self.model = model
        self.model_type = model_type
        self.output_format = output_format
        self.target_name = target_name

    def __repr__(self):
        return self.model.__repr__()

    def dump(self, path):

        if self.model_type in ['secureboost', 'tree', 'sbt']:
            if self.output_format == 'pmml':
                feature_names = self.model['lgb'].feature_name_
                lgb_to_pmml(self.model, feature_names, self.target_name, path)
            else:
                with open(path, 'w') as f:
                    f.write(self.model)

        # linear model here


def hetero_model_merge(guest_param: dict, guest_meta: dict, host_params: list, host_metas: list, model_type: str,
                       output_format: str, target_name: str = 'y'):

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

    :return: Merged Model Class
    """
    guest_param = copy.deepcopy(guest_param)
    guest_meta = copy.deepcopy(guest_meta)
    host_params = copy.deepcopy(host_params)
    host_metas = copy.deepcopy(host_metas)

    if not isinstance(model_type, str):
        raise ValueError('model type should be a str, but got {}'.format(model_type))

    if output_format.lower() not in ['lightgbm', 'lgb', 'sklearn', 'pmml']:
        raise ValueError('unknown output format: {}'.format(output_format))

    if model_type.lower() in ['secureboost', 'tree', 'sbt']:
        model = merge_sbt(guest_param, guest_meta, host_params, host_metas, output_format, target_name)
        return MergedModel(model, model_type, output_format, target_name)
    elif model_type.lower() in ['logistic_regression', 'lr']:
        pass
    else:
        raise ValueError('model type should be one in ["sbt", "lr"], '
                         'but got unknown model type: {}'.format(model_type))
