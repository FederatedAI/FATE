import json
import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from lightgbm.sklearn import _LGBMLabelEncoder
from federatedml.protobuf.homo_model_convert.lightgbm.gbdt import sbt_to_lgb
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from google.protobuf import json_format
from federatedml.util.anonymous_generator_util import Anonymous


def _merge_sbt(guest_param, host_param, host_sitename, rename_host=True):
    # update feature name fid mapping
    guest_fid_map = guest_param['featureNameFidMapping']
    guest_fid_map = {int(k): v for k, v in guest_fid_map.items()}
    host_fid_map = sorted([(int(k), v) for k, v in host_param['featureNameFidMapping'].items()], key=lambda x: x[0])
    guest_feat_len = len(guest_fid_map)
    start = guest_feat_len
    host_new_fid = {}

    for k, v in host_fid_map:
        guest_fid_map[start] = v if not rename_host else v + '_' + host_sitename
        host_new_fid[k] = start
        start += 1

    guest_param['featureNameFidMapping'] = guest_fid_map

    # merging trees
    for tree_guest, tree_host in zip(guest_param['trees'], host_param['trees']):

        tree_guest['splitMaskdict'].update(tree_host['splitMaskdict'])
        tree_guest['missingDirMaskdict'].update(tree_host['missingDirMaskdict'])

        for node_g, node_h in zip(tree_guest['tree'], tree_host['tree']):

            if str(node_h['id']) in tree_host['splitMaskdict']:
                node_g['fid'] = int(host_new_fid[int(node_h['fid'])])
                node_g['sitename'] = host_sitename
                node_g['bid'] = 0

    return guest_param


def extract_host_name(host_param, idx):

    try:
        anonymous_obj = Anonymous()
        anonymous_dict = host_param['anonymousNameMapping']
        role, party_id = None, None
        for key in anonymous_dict:
            role = anonymous_obj.get_role_from_anonymous_column(key)
            party_id = anonymous_obj.get_party_id_from_anonymous_column(key)
            break
        if role is not None and party_id is not None:
            return role + '_' + party_id
        else:
            return None
    except Exception as e:
        return 'host_{}'.format(idx)


def merge_sbt(guest_param: dict, guest_meta: dict, host_params: list, host_metas: list, output_format: str,
              target_name='y', host_rename=True):

    result_param = None
    for idx, host_param in enumerate(host_params):
        host_name = extract_host_name(host_param, idx)
        if result_param is None:
            result_param = _merge_sbt(guest_param, host_param, host_name, host_rename)
        else:
            result_param = _merge_sbt(result_param, host_param, host_name, host_rename)

    pb_param = json_format.Parse(json.dumps(result_param), BoostingTreeModelParam())
    pb_meta = json_format.Parse(json.dumps(guest_meta), BoostingTreeModelMeta())
    lgb_model = sbt_to_lgb(pb_param, pb_meta, False)

    if output_format in ['lgb', 'lightgbm']:
        return lgb_model
    elif output_format in ['pmml']:
        classes = list(map(int, pb_param.classes_))
        bst = lgb.Booster(model_str=lgb_model)
        new_clf = lgb.LGBMRegressor() if guest_meta['taskType'] == 'regression' else lgb.LGBMClassifier()
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
