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
#
import os
import torch
from torch import nn, optim
from torch.nn import Sequential
import pandas as pd
import argparse
from fate_client.pipeline.utils import test_utils
from torch.utils.data import DataLoader, TensorDataset
from fate_client.pipeline.utils.test_utils import JobConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tqdm

seed = 114514
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config="../../config.yaml", param="", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = test_utils.load_job_config(config)

    if isinstance(param, str):
        param = test_utils.JobConfig.load_from_file(param)

    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        print(f"config: {config}")
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    assert isinstance(param, dict)

    epochs = param.get('epochs')
    batch_size = param.get('batch_size')
    in_feat = param.get('in_feat')
    out_feat = param.get('out_feat')
    lr = param.get('lr')
    idx = param.get('idx')
    label_name = param.get('label_name')

    guest_data_table = param.get("data_guest")
    host_data_table = param.get("data_host")

    guest_data = pd.read_csv(os.path.join(data_base_dir, guest_data_table), index_col=idx)
    host_data = pd.read_csv(os.path.join(data_base_dir, host_data_table), index_col=idx)
    
    X = pd.concat([guest_data, host_data], ignore_index=True)
    y = X.pop(label_name).values

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Sequential(
            nn.Linear(in_feat, out_feat),
            nn.ReLU(),
            nn.Linear(out_feat ,1)
        )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm.tqdm(range(epochs)):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        y_train_pred = model(X).numpy()
        
    mse = mean_squared_error(y, y_train_pred)
    rmse = mse ** 0.5
    return {}, {'rmse': rmse}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY PIPELINE JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="")
    args = parser.parse_args()
    main(args.config, args.param)