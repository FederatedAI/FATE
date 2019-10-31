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

from arch.api import session

from arch.api.utils import log_utils, dtable_utils

LOGGER = log_utils.getLogger()


class Download(object):
    def __init__(self):
        self.taskid = ''
        self.tracker = None
        self.parameters = {}

    def run(self, component_parameters=None, args=None):
        self.parameters = component_parameters["DownloadParam"]
        self.parameters["role"] = component_parameters["role"]
        self.parameters["local"] = component_parameters["local"]
        table_name, namespace = dtable_utils.get_table_info(config=self.parameters,
                                                            create=False)
        job_id = "_".join(self.taskid.split("_")[:2])
        session.init(job_id, self.parameters["work_mode"])
        with open(os.path.abspath(self.parameters["output_path"]), "w") as fout:
            data_table = session.get_data_table(name=table_name, namespace=namespace)
            LOGGER.info('===== begin to export data =====')
            lines = 0
            for key, value in data_table.collect():
                if not value:
                    fout.write(key + "\n")
                else:
                    fout.write(key + self.parameters.get("delimitor", ",") + str(value) + "\n")
                lines += 1
                if lines % 2000 == 0:
                    LOGGER.info("===== export {} lines =====".format(lines))
            LOGGER.info("===== export {} lines totally =====".format(lines))
            LOGGER.info('===== export data finish =====')
            LOGGER.info('===== export data file path:{} ====='.format(os.path.abspath(self.parameters["output_path"])))

    def set_taskid(self, taskid):
        self.taskid = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None





