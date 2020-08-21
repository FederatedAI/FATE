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

from fate_arch.common import log
from fate_flow.entity.metric import Metric, MetricMeta
from fate_arch import storage
from fate_flow.utils import job_utils
from fate_flow.scheduling_apps.client import ControllerClient


LOGGER = log.getLogger()


class Download(object):
    def __init__(self):
        self.taskid = ''
        self.tracker = None
        self.parameters = {}

    def run(self, component_parameters=None, args=None):
        self.parameters = component_parameters["DownloadParam"]
        self.parameters["role"] = component_parameters["role"]
        self.parameters["local"] = component_parameters["local"]
        name, namespace = self.parameters.get("name"), self.parameters.get("namespace")
        with open(os.path.abspath(self.parameters["output_path"]), "w") as fout:
            with storage.Session.build(session_id=job_utils.generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id, suffix="storage", random_end=True),
                                       name=name,
                                       namespace=namespace) as storage_session:
                data_table = storage_session.get_table(namespace=namespace, name=name)
                count = data_table.count()
                LOGGER.info('===== begin to export data =====')
                lines = 0
                job_info = {}
                job_info["job_id"] = self.tracker.job_id
                job_info["role"] = self.tracker.role
                job_info["party_id"] = self.tracker.party_id
                for key, value in data_table.collect():
                    if not value:
                        fout.write(key + "\n")
                    else:
                        fout.write(key + self.parameters.get("delimitor", ",") + str(value) + "\n")
                    lines += 1
                    if lines % 2000 == 0:
                        LOGGER.info("===== export {} lines =====".format(lines))
                    if lines % 10000 == 0:
                        job_info["progress"] = lines/count*100//1
                        ControllerClient.update_job(job_info=job_info)
                job_info["progress"] = 100
                ControllerClient.update_job(job_info=job_info)
                self.callback_metric(metric_name='data_access',
                                     metric_namespace='download',
                                     metric_data=[Metric("count", data_table.count())])
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

    def callback_metric(self, metric_name, metric_namespace, metric_data):
        self.tracker.log_metric_data(metric_name=metric_name,
                                     metric_namespace=metric_namespace,
                                     metrics=metric_data)
        self.tracker.set_metric_meta(metric_namespace,
                                     metric_name,
                                     MetricMeta(name='download',
                                                metric_type='DOWNLOAD'))





