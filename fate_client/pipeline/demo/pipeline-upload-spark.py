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
from pipeline.component.dataio import DataIO
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

# find python path
import site
SITE_PATH = site.getsitepackages()[0]


def main():
    # parties config
    guest = 9999
    host = 9999
    # 0 for eggroll, 1 for spark
    backend = Backend.SPARK
    # 0 for standalone, 1 for cluster
    # work_mode = WorkMode.STANDALONE
    # use the work mode below for cluster deployment
    work_mode = WorkMode.CLUSTER
    # storage engine for uploaded data
    storage_engine = "HDFS"

    # partition for data storage
    partition = 8

    dense_data = {"name": "breast_hetero_guest", "namespace": "experiment"}

    tag_data = {"name": "tag_value_1", "namespace": "experiment"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host)
    # add upload data info
    # csv file name from python path & file name
    pipeline_upload.add_upload_data(file=os.path.join(SITE_PATH, "examples/data/breast_hetero_guest.csv"),
                                    table_name=dense_data["name"],             # table name
                                    namespace=dense_data["namespace"],         # namespace
                                    head=1, partition=partition,               # data info
                                    storage_path="hdfs://mfate-cluster/data")  # storage path

    pipeline_upload.add_upload_data(file=os.path.join(SITE_PATH, "examples/data/tag_value_1000_140.csv"),
                                    table_name=tag_data["name"],
                                    namespace=tag_data["namespace"],
                                    head=0, partition=partition,
                                    storage_path="hdfs://mfate-cluster/data")
    # upload all data
    pipeline_upload.upload(work_mode=work_mode, backend=backend, drop=1, storage_engine=storage_engine)

    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).algorithm_param(table=dense_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role="guest", party_id=guest).algorithm_param(table=tag_data)

    dataio_0 = DataIO(name="dataio_0", with_label=True, label_name="y", output_format="dense",
                      missing_fill=False, outlier_replace=False)

    dataio_1 = DataIO(name="dataio_1", with_label=False, input_format="tag", output_format="dense",
                      tag_with_value=True, delimitor=",")


    pipeline.add_component(reader_0)
    pipeline.add_component(reader_1)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(dataio_1, data=Data(data=reader_1.output.data))

    pipeline.compile()

    pipeline.fit(backend=backend, work_mode=work_mode)


if __name__ == "__main__":
    main()
