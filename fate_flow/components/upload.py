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
import shutil
import time

from fate_arch.common import log, file_utils
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.utils.job_utils import generate_session_id
from fate_flow.scheduling_apps.client import ControllerClient
from fate_arch import storage

LOGGER = log.getLogger()


class Upload(object):
    def __init__(self):
        self.taskid = ''
        self.tracker = None
        self.MAX_PARTITIONS = 1024
        self.MAX_BYTES = 1024*1024*8
        self.parameters = {}
        self.table = None

    def run(self, component_parameters=None, args=None):
        self.parameters = component_parameters["UploadParam"]
        LOGGER.info(self.parameters)
        self.parameters["role"] = component_parameters["role"]
        self.parameters["local"] = component_parameters["local"]
        job_id = self.taskid.split("_")[0]
        if not os.path.isabs(self.parameters.get("file", "")):
            self.parameters["file"] = os.path.join(file_utils.get_project_base_directory(), self.parameters["file"])
        if not os.path.exists(self.parameters["file"]):
            raise Exception("%s is not exist, please check the configure" % (self.parameters["file"]))
        if not os.path.getsize(self.parameters["file"]):
            raise Exception("%s is an empty file" % (self.parameters["file"]))
        name, namespace = self.parameters.get("name"), self.parameters.get("namespace")
        _namespace, _table_name = self.generate_table_name(self.parameters["file"])
        if namespace is None:
            namespace = _namespace
        if name is None:
            name = _table_name
        read_head = self.parameters['head']
        if read_head == 0:
            head = False
        elif read_head == 1:
            head = True
        else:
            raise Exception("'head' in conf.json should be 0 or 1")
        partitions = self.parameters["partition"]
        if partitions <= 0 or partitions >= self.MAX_PARTITIONS:
            raise Exception("Error number of partition, it should between %d and %d" % (0, self.MAX_PARTITIONS))
        with storage.Session.build(session_id=generate_session_id(self.tracker.task_id, self.tracker.task_version, self.tracker.role, self.tracker.party_id, suffix="storage", random_end=True),
                                   storage_engine=self.parameters["storage_engine"], options=self.parameters.get("options")) as storage_session:
            from fate_arch.storage import EggRollStorageType
            address = storage.StorageTableMeta.create_address(storage_engine=self.parameters["storage_engine"], address_dict={"name": name, "namespace": namespace, "storage_type": EggRollStorageType.ROLLPAIR_LMDB})
            self.parameters["partitions"] = partitions
            self.parameters["name"] = name
            if self.parameters.get("destroy", False):
                LOGGER.info(f"destroy table {name} {namespace}")
                storage_session.get_table(name=name, namespace=namespace).destroy()
            self.table = storage_session.create_table(address=address, **self.parameters)
            data_table_count = self.save_data_table(job_id, name, namespace, head)
        LOGGER.info("------------load data finish!-----------------")
        # rm tmp file
        try:
            if '{}/fate_upload_tmp'.format(job_id) in self.parameters['file']:
                LOGGER.info("remove tmp upload file")
                shutil.rmtree(os.path.join(self.parameters["file"].split('tmp')[0], 'tmp'))
        except:
            LOGGER.info("remove tmp file failed")
        LOGGER.info("file: {}".format(self.parameters["file"]))
        LOGGER.info("total data_count: {}".format(data_table_count))
        LOGGER.info("table name: {}, table namespace: {}".format(name, namespace))

    def set_taskid(self, taskid):
        self.taskid = taskid

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data_table(self, job_id, dst_table_name, dst_table_namespace, head=True):
        input_file = self.parameters["file"]
        count = self.get_count(input_file)
        with open(input_file, 'r') as fin:
            lines_count = 0
            if head is True:
                data_head = fin.readline()
                count -= 1
                self.save_data_header(data_head)
            n = 0
            while True:
                data = list()
                lines = fin.readlines(self.MAX_BYTES)
                if lines:
                    for line in lines:
                        values = line.replace("\n", "").replace("\t", ",").split(",")
                        data.append((values[0], self.list_to_str(values[1:])))
                    lines_count += len(data)
                    save_progress = lines_count/count*100//1
                    job_info = {'progress': save_progress, "job_id": job_id, "role": self.parameters["local"]['role'], "party_id": self.parameters["local"]['party_id']}
                    ControllerClient.update_job(job_info=job_info)
                    self.table.put_all(data)
                    if n == 0:
                        self.table.get_meta().update_metas(part_of_data=data)
                else:
                    self.table.get_meta().update_metas(count=self.table.count(), partitions=self.parameters["partition"])
                    count_actual = self.table.count()
                    self.tracker.log_output_data_info(data_name='upload',
                                                      table_namespace=dst_table_namespace,
                                                      table_name=dst_table_name)

                    self.tracker.log_metric_data(metric_namespace="upload",
                                                 metric_name="data_access",
                                                 metrics=[Metric("count", count_actual)])
                    self.tracker.set_metric_meta(metric_namespace="upload",
                                                 metric_name="data_access",
                                                 metric_meta=MetricMeta(name='upload', metric_type='UPLOAD'))
                    return count_actual
                n += 1

    def save_data_header(self, header_source):
        header_source_item = header_source.split(',')
        self.table.get_meta().update_metas(schema={'header': ','.join(header_source_item[1:]).strip(), 'sid': header_source_item[0]})

    def get_count(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            count = 0
            for line in fp:
                count += 1
        return count

    def list_to_str(self, input_list):
        return ','.join(list(map(str, input_list)))

    def generate_table_name(self, input_file_path):
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        file_name = input_file_path.split(".")[0]
        file_name = file_name.split("/")[-1]
        return file_name, str_time

    def save_data(self):
        return None

    def export_model(self):
        return None
