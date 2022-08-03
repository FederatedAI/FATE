import json
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from lightgbm.sklearn import _LGBMLabelEncoder
from federatedml.protobuf.homo_model_convert.lightgbm.gbdt import sbt_to_lgb
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from google.protobuf import json_format


def _merge_sbt(guest_param, host_param, host_sitename):

    # update feature name fid mapping
    guest_fid_map = guest_param['featureNameFidMapping']
    host_fid_map = host_param['featureNameFidMapping']
    guest_feat_len = len(guest_fid_map)
    start = guest_feat_len
    host_new_fid = {}

    for k, v in host_fid_map.items():
        guest_fid_map[str(start)] = v
        host_new_fid[k] = str(start)
        start += 1

    new_host_fid_map = {}
    for key, item in host_fid_map.items():
        new_key = host_new_fid[key]
        new_host_fid_map[new_key] = item + '_' + host_sitename

    guest_fid_map.update(new_host_fid_map)
    guest_param['featureNameFidMapping'] = guest_fid_map

    # merging trees
    for tree_guest, tree_host in zip(guest_param['trees'], host_param['trees']):

        tree_guest['splitMaskdict'].update(tree_host['splitMaskdict'])
        tree_guest['missingDirMaskdict'].update(tree_host['missingDirMaskdict'])

        for node_g, node_h in zip(tree_guest['tree'], tree_host['tree']):

            if str(node_h['id']) in tree_host['splitMaskdict']:
                node_g['fid'] = int(host_new_fid[str(node_h['fid'])])
                node_g['sitename'] = host_sitename
                node_g['bid'] = 0

    return guest_param


def extract_host_name(host_param, idx):

    try:
        anonymous_dict = host_param['anonymousNameMapping']
        split_dict = None
        split_dict_2 = None
        for key in anonymous_dict:
            split_dict = key.split('_')
            split_dict_2 = key.split(':')
            break
        if len(split_dict) == 3:
            return str(split_dict[0]) + '_' + str(split_dict[1])
        elif len(split_dict_2) == 3:
            return str(split_dict_2[0]) + '_' + str(split_dict_2[1])
        else:
            return None
    except Exception as e:
        return 'host_{}'.format(idx)


def merge_sbt(guest_param: dict, guest_meta: dict, host_params: list, host_metas: list, output_format: str,
              target_name='y'):

    result_param = None
    for idx, host_param in enumerate(host_params):
        host_name = extract_host_name(host_param, idx)
        if result_param is None:
            result_param = _merge_sbt(guest_param, host_param, host_name)
        else:
            result_param = _merge_sbt(result_param, host_param, host_name)

    pb_param = json_format.Parse(json.dumps(result_param), BoostingTreeModelParam())
    pb_meta = json_format.Parse(json.dumps(guest_meta), BoostingTreeModelMeta())
    lgb_model = sbt_to_lgb(pb_param, pb_meta, False)

    if output_format in ['lgb', 'lightgbm']:
        return lgb_model
    elif output_format in ['pmml']:
        classes = list(map(int, pb_param.classes_))
        bst = lgb.Booster(model_str=lgb_model)
        new_clf = lgb.LGBMClassifier()
        new_clf._Booster = bst
        new_clf._n_features = len(bst.feature_name())
        new_clf._n_classes = len(np.unique(classes))
        new_clf._le = _LGBMLabelEncoder().fit(np.array(classes))
        new_clf.fitted_ = True
        new_clf._classes = new_clf._le.classes_
        test_pipeline = Pipeline([("lgb", new_clf)])
        return test_pipeline

    else:
        raise ValueError('unknown output type {}'.format(output_format))


if __name__ == '__main__':
    """
    Param to modify
    """

    host_sitename_ = 'host:9998'

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



