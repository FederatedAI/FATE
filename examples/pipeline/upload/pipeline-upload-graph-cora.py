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
import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.utils.tools import load_job_config


def main(data_base):
    guest = 9999

    # partition for data storage
    partition = 4

    # table name and namespace, used in FATE job configuration
    guest_data = {
        "feats": "cora_feats_guest",
        "train": "cora_train_guest",
        "val": "cora_val_guest",
        "test": "cora_test_guest",
        "adj": "cora_adj_guest",
        "namespace": f"experiment"}
    host_data = {
        "feats": "cora_feats_host",
        "train": "cora_train_host",
        "val": "cora_val_host",
        "test": "cora_test_host",
        "adj": "cora_adj_host",
        "namespace": f"experiment"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)
    # add upload data info
    # path to csv file(s) to be uploaded, modify to upload designated data
    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_feats_guest.csv"),
                                    table_name=guest_data["feats"],             # table name
                                    namespace=guest_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_train_guest.csv"),
                                    table_name=guest_data["train"],             # table name
                                    namespace=guest_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_val_guest.csv"),
                                    table_name=guest_data["val"],             # table name
                                    namespace=guest_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_test_guest.csv"),
                                    table_name=guest_data["test"],             # table name
                                    namespace=guest_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_adj_guest.csv"),
                                    table_name=guest_data["adj"],             # table name
                                    namespace=guest_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_feats_host.csv"),
                                    table_name=host_data["feats"],             # table name
                                    namespace=host_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_train_host.csv"),
                                    table_name=host_data["train"],             # table name
                                    namespace=host_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_val_host.csv"),
                                    table_name=host_data["val"],             # table name
                                    namespace=host_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_test_host.csv"),
                                    table_name=host_data["test"],             # table name
                                    namespace=host_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "cora4fate/cora_adj_host.csv"),
                                    table_name=host_data["adj"],             # table name
                                    namespace=host_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    # upload data
    pipeline_upload.upload(drop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--base", "-b", type=str, required=True,
                        help="data base, path to directory that contains examples/data")

    args = parser.parse_args()
    main(args.base)
