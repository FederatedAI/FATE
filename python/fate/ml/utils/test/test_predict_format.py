#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from fate.arch import Context
from fate.arch.computing.backends.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.backends.standalone import StandaloneFederation
import pandas as pd
from fate.ml.utils.predict_tools import compute_predict_details, PREDICT_SCORE, LABEL, BINARY, REGRESSION, MULTI
from fate.arch.dataframe import PandasReader
import numpy as np


computing = CSession()
ctx = Context("guest", computing=computing, federation=StandaloneFederation(
    computing, "fed", ("guest", 10000), [("host", 9999)]), )

df = pd.DataFrame()
df['id'] = [i for i in range(50)]
df['sample_id'] = [i for i in range(len(df))]
df[PREDICT_SCORE] = [np.random.random(1)[0] for i in range(len(df))]
df[LABEL] = [np.random.randint(0, 2) for i in range(len(df))]

no_label_df = df.drop([LABEL], axis=1)


df_reg = pd.DataFrame()
df_reg['id'] = [i for i in range(50)]
df_reg['sample_id'] = [i for i in range(len(df_reg))]
df_reg[PREDICT_SCORE] = [np.random.random(1)[0] * 10 for i in range(len(df_reg))]
df_reg[LABEL] = [np.random.random(1)[0] * 10 for i in range(len(df_reg))]

df_multi = pd.DataFrame()
df_multi['id'] = [i for i in range(50)]
df_multi['sample_id'] = [i for i in range(len(df_multi))]
df_multi[PREDICT_SCORE] = [[float(np.random.random(1)[0]) for i in range(3)] for i in range(len(df_multi))]
df_multi[LABEL] = [np.random.randint(0, 3) for i in range(len(df_multi))]

reader = PandasReader(
    sample_id_name='sample_id',
    match_id_name="id",
    dtype="object")
data = reader.to_frame(ctx, df)
data_2 = reader.to_frame(ctx, no_label_df)
data_3 = reader.to_frame(ctx, df_reg)
data_4 = reader.to_frame(ctx, df_multi)


rs = compute_predict_details(data, BINARY, classes=[0, 1], threshold=0.8)
rs_2 = compute_predict_details(data_2, BINARY, classes=[0, 1], threshold=0.3)
rs_3 = compute_predict_details(data_3, REGRESSION)
rs_4 = compute_predict_details(data_4, MULTI, classes=[0, 1, 2])