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

from pipeline.backend.pipeline import PipeLine
from pipeline.utils.tools import load_job_config

def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    backend = config.backend
    work_mode = config.work_mode
    data_base = config.data_base

    # partition for data storage
    partition = 4

    dense_data = {"name": "breast_hetero_guest", "namespace": f"experiment{namespace}"}

    tag_data = {"name": "tag_value_1", "namespace": f"experiment{namespace}"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)
    # add upload data info
    # csv file name from python path & file name
    pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_guest.csv"),
                                    table_name=dense_data["name"],             # table name
                                    namespace=dense_data["namespace"],         # namespace
                                    head=1, partition=partition,               # data info
                                    id_delimiter=",")                          # needed for spark backend

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/tag_value_1000_140.csv"),
                                    table_name=tag_data["name"],
                                    namespace=tag_data["namespace"],
                                    head=0, partition=partition,
                                    id_delimiter=",")
    # upload all data
    pipeline_upload.upload(work_mode=work_mode, backend=backend, drop=1)


if __name__ == "__main__":
    main()
