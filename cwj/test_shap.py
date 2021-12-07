import xgboost as xgb
import lightgbm as lgb
from shap import PermutationExplainer, TreeExplainer, KernelExplainer
import pandas as pd
import numpy as np
from shap.utils import partition_tree_shuffle, MaskedModel

data = pd.read_csv('../examples/data/vehicle_scale_homo_guest.csv')

label = data['y']
data = data.drop(columns=['y'])
model = xgb.XGBClassifier()
model.fit(data, label)

train_set = lgb.Dataset(data=data, label=label)
model2 = lgb.train({'num_leaves': 31, 'objective': 'multiclass', 'num_class': 4}, train_set=train_set)


def pred(x):
    rs = model.predict_proba(x)
    return rs[::, 1]


# ex = PermutationExplainer(pred, data)
# rs = ex(data[0:1])

# ex2 = TreeExplainer(model2)
# interaction_shap = ex2.shap_interaction_values(data)
#
# arr = data.values
# zeros = np.zeros((1, arr.shape[1]))
# means = np.array([arr.mean(axis=0)])
# ex3 = KernelExplainer(pred, means)
# rs = ex3.shap_values(arr)


