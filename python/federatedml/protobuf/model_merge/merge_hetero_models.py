from federatedml.protobuf.model_merge.merge_sbt import merge_sbt
from nyoka import lgb_to_pmml


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
                       linear_regression, alias LinR
    :param output_format: output format of merged model, support:
                          lightgbm, for tree models only
                          sklearn, for linear models only
                          pmml, for all types
    :param target_name: if output format is pmml, need to specify the targe(label) name

    :return: Merged Model Class
    """

    if not isinstance(model_type, str):
        raise ValueError('model type should be a str, but got {}'.format(model_type))

    if output_format.lower() not in ['lightgbm', 'lgb', 'sklearn', 'pmml']:
        raise ValueError('unknown output format: {}'.format(output_format))

    if model_type.lower() in ['secureboost', 'tree', 'sbt']:
        model = merge_sbt(guest_param, guest_meta, host_params, host_metas, output_format, target_name)
        return MergedModel(model, model_type, output_format, target_name)
    elif model_type.lower() in ['logistic_regression', 'lr']:
        pass
    elif model_type.lower() in ['linear_regression', 'linr']:
        pass
    else:
        raise ValueError('model type should be one in ["sbt", "lr", "linr"], '
                         'but got unknown model type: {}'.format(model_type))


if __name__ == '__main__':
    import json
    guest_json_path = '/home/cwj/standalone_fate_install_1.8.0/fateflow/model_local_cache' \
                      '/guest#9999#guest-9999#host-9998#model/202205181728266208040/variables/data' \
                      '/hetero_secure_boost_0/model'

    host_json_path = '/home/cwj/standalone_fate_install_1.8.0/fateflow/model_local_cache/' \
                     'host#9998#guest-9999#host-9998#model/202205181728266208040/variables/data/hetero_secure_boost_0/model'

    """
    Merging codes
    """

    param_name_guest = 'HeteroSecureBoostingTreeGuestParam.json'
    meta_name_guest = 'HeteroSecureBoostingTreeGuestMeta.json'
    param_name_host = 'HeteroSecureBoostingTreeHostParam.json'
    guest_param_ = json.loads(open(guest_json_path + '/' + param_name_guest, 'r').read())
    guest_meta_ = json.loads(open(guest_json_path + '/' + meta_name_guest, 'r').read())
    host_param_ = json.loads(open(host_json_path + '/' + param_name_host, 'r').read())

    ret = hetero_model_merge(guest_param_, guest_meta_, [host_param_], [], 'sbt', 'lgb', 'y')