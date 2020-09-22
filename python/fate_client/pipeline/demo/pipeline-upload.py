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

from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.pipeline import PipeLine

# path to data
# default fate installation path
DATA_BASE = "/data/projects/fate"

# site-package ver
# import site
# DATA_BASE = site.getsitepackages()[0]


def main():
    # parties config
    guest = 9999
    # 0 for eggroll, 1 for spark
    backend = Backend.EGGROLL
    # 0 for standalone, 1 for cluster
    work_mode = WorkMode.STANDALONE
    # use the work mode below for cluster deployment
    # work_mode = WorkMode.CLUSTER

    # partition for data storage
    partition = 4

    dense_data = {"name": "breast_hetero_guest", "namespace": f"experiment"}

    tag_data = {"name": "tag_value_1", "namespace": f"experiment"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)
    # add upload data info
    # csv file name from python path & file name
    pipeline_upload.add_upload_data(file=os.path.join(DATA_BASE, "examples/data/breast_hetero_guest.csv"),
                                    table_name=dense_data["name"],             # table name
                                    namespace=dense_data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.add_upload_data(file=os.path.join(DATA_BASE, "examples/data/tag_value_1000_140.csv"),
                                    table_name=tag_data["name"],
                                    namespace=tag_data["namespace"],
                                    head=0, partition=partition)

    # upload all data
    pipeline_upload.upload(work_mode=work_mode, backend=backend, drop=1)


if __name__ == "__main__":
    main()
