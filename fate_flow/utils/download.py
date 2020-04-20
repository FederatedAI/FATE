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
from fate_flow.entity.metric import Metric, MetricMeta

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
        job_id = self.taskid.split("_")[0]
        session.init(job_id, self.parameters["work_mode"])
        with open(os.path.abspath(self.parameters["output_path"]), "w") as fout:
            data_table = session.get_data_table(name=table_name, namespace=namespace)
            count = data_table.count()
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
                if lines % 10000 == 0:
                    job_info = {'f_progress': lines/count*100//1}
                    self.update_job_status(self.parameters["local"]['role'], self.parameters["local"]['party_id'],
                                           job_info)
            self.update_job_status(self.parameters["local"]['role'],
                                   self.parameters["local"]['party_id'], {'f_progress': 100})
            self.callback_metric(metric_name='data_access',
                                 metric_namespace='download',
                                 metric_data=[Metric("count", data_table.count())])
            LOGGER.info("===== export {} lines totally =====".format(lines))
            LOGGER.info('===== export data finish =====')
            LOGGER.info('===== export data file path:{} ====='.format(os.path.abspath(self.parameters["output_path"])))

    def update_job_status(self, role, party_id, job_info):
        self.tracker.save_job_info(role=role, party_id=party_id, job_info=job_info)

    def set_taskid(self, taskid):
        self.taskid = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return None

    def export_model(self):
        return None

    def callback_metric(self, metric_name, metric_namespace, metric_data):
        self.tracker.log_metric_data(metric_name=metric_name,
                                     metric_namespace=metric_namespace,
                                     metrics=metric_data)
        self.tracker.set_metric_meta(metric_namespace,
                                     metric_name,
                                     MetricMeta(name='download',
                                                metric_type='DOWNLOAD'))





