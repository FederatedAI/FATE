from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam, NodeParam, \
    DecisionTreeModelParam, FeatureImportanceInfo
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
from federatedml.protobuf.model_migrate.model_migrate import model_migration
import copy

host_old = [10000, 9999]
host_new = [114, 514, ]

guest_old = [10000]
guest_new = [1919]

param = BoostingTreeModelParam()

fp0 = FeatureImportanceInfo()
fp0.fullname = 'host_10000_0'
fp0.sitename = 'host:10000'

fp1 = FeatureImportanceInfo()
fp1.sitename = 'host:9999'
fp1.fullname = 'host_9999_1'

fp2 = FeatureImportanceInfo(fullname='x0')
fp2.sitename = 'guest:10000'

feature_importance = [fp0, fp1, fp2]
param.feature_importances.extend(feature_importance)

tree_0 = DecisionTreeModelParam(tree_=[NodeParam(sitename='guest:10000'), NodeParam(sitename='guest:10000'),
                                       NodeParam(sitename='guest:10000')])
tree_1 = DecisionTreeModelParam(tree_=[NodeParam(sitename='host:10000'), NodeParam(sitename='host:9999'),
                                       NodeParam(sitename='host:10000')])
tree_2 = DecisionTreeModelParam(tree_=[NodeParam(sitename='host:9999'), NodeParam(sitename='guest:10000'),
                                       NodeParam(sitename='host:9999')])
tree_3 = DecisionTreeModelParam()

param.trees_.extend([tree_0, tree_1, tree_2, tree_3])
rs = model_migration({'HelloParam': param, 'HelloMeta': {}}, 'HeteroSecureBoost', old_guest_list=guest_old,
                     new_guest_list=guest_new, old_host_list=host_old, new_host_list=host_new, )
