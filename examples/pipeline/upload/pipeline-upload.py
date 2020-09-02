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

import argparse
import os
# find python path
import site

from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.reader import Reader
from pipeline.interface.data import Data

from examples.util.config import Config

SITE_PATH = site.getsitepackages()[0]


def main(config="../../config.yaml"):
    # obtain config
    config = Config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    backend = config.backend
    work_mode = config.work_mode

    dense_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    tag_data = {"name": "tag_value_1", "namespace": "experiment"}

    pipeline_upload = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)
    # add upload data info
    # csv file name from python path & file name
    pipeline_upload.add_upload_data(file=os.path.join(SITE_PATH, "examples/data/breast_hetero_guest.csv"),
                                    table_name=dense_data["name"],             # table name
                                    namespace=dense_data["namespace"],         # namespace
                                    head=1, partition=8)
    pipeline_upload.add_upload_data(file=os.path.join(SITE_PATH, "examples/data/tag_value_1000_140.csv"),
                                    table_name=tag_data["name"],
                                    namespace=tag_data["namespace"],
                                    head=0, partition=8)
    # upload all data
    pipeline_upload.upload(work_mode=work_mode, drop=1)

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=dense_data)

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role='guest', party_id=guest).algorithm_param(table=tag_data)

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
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
