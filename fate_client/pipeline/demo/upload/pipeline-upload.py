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

from pipeline.backend.pipeline import PipeLine
from pipeline.component.dataio import DataIO
from pipeline.component.reader import Reader
from pipeline.demo.util.demo_util import Config
from pipeline.interface.data import Data

# find python path
import site
SITE_PATH = site.getsitepackages()[0]


def main(config="../config.yaml"):
    # obtain config
    config = Config(config)
    guest = config.guest
    host = config.host[0]
    backend = config.backend
    work_mode = config.work_mode

    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    pipeline_upload = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)
    # add upload data info
    # csv file name from python path & file name
    pipeline_upload.add_upload_data(os.path.join(SITE_PATH, "examples/data/breast_hetero_guest.csv"),
                             table_name=guest_train_data["name"],             # table name
                             namespace=guest_train_data["namespace"])         # namespace
    pipeline_upload.add_upload_data(os.path.join(SITE_PATH, "examples/data/breast_hetero_host.csv"),
                             table_name=host_train_data["name"],
                             namespace=host_train_data["namespace"])
    # upload all data
    pipeline_upload.upload(work_mode=work_mode, drop=-1)

    pipeline = PipeLine().set_initiator(role='guest', party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).algorithm_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=host).algorithm_param(table=host_train_data)

    dataio_0 = DataIO(name="dataio_0", with_label=False, tag_with_value=True, output_format="dense")

    pipeline.add_component(reader_0)
    pipeline.add_component(dataio_0, data=Data(data=reader_0.output.data))

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
