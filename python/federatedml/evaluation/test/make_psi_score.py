import xgboost as xgb
import pandas as pd

df_guest = pd.read_csv('./hr_data_guest_oot2_train_1vs10.csv')
df_guest.columns = ['sid', 'y', 'mock_feature']
df_host = pd.read_csv('./hr_data_host_oot2.csv')

df = df_guest.merge(df_host, on=['sid'], how='inner')

df = df.drop(columns=['sid'])
labels = df['y']
feats = df.drop(columns=['y'])
split_ratio = 0.8
train_label = labels[0: int(len(feats)*split_ratio)]
train = feats[0: int(len(feats)*split_ratio)]
validate = feats[int(len(feats)*split_ratio):]

param = {'max_depth': 4, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

train_mat = xgb.DMatrix(data=train, label=train_label)
validate_mat = xgb.DMatrix(data=validate)
model = xgb.train(dtrain=xgb.DMatrix(data=train, label=train_label), params=param, num_boost_round=50)
rs1 = model.predict(train_mat)
rs2 = model.predict(validate_mat)

score1 = pd.DataFrame(data=rs1, columns=['score'])
score2 = pd.DataFrame(data=rs2, columns=['score'])

score1.to_csv('hr_score_train.csv', index=False)
score2.to_csv('hr_score_validate.csv', index=False)