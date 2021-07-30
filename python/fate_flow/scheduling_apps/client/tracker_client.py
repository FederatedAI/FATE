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
import base64
from typing import List

from fate_arch import storage
from fate_arch.abc import AddressABC
from fate_arch.common import log
from fate_arch.common.base_utils import serialize_b64, deserialize_b64
from fate_flow.entity.types import RetCode, RunParameters
from fate_flow.entity.metric import Metric, MetricMeta
from fate_flow.operation.job_tracker import Tracker
from fate_flow.utils import api_utils

LOGGER = log.getLogger()


class TrackerClient(object):
    def __init__(self, job_id: str, role: str, party_id: int,
                 model_id: str = None,
                 model_version: str = None,
                 component_name: str = None,
                 component_module_name: str = None,
                 task_id: str = None,
                 task_version: int = None,
                 job_parameters: RunParameters = None
                 ):
        self.job_id = job_id
        self.role = role
        self.party_id = party_id
        self.model_id = model_id
        self.model_version = model_version
        self.component_name = component_name if component_name else 'pipeline'
        self.module_name = component_module_name if component_module_name else 'Pipeline'
        self.task_id = task_id
        self.task_version = task_version
        self.job_parameters = job_parameters
        self.job_tracker = Tracker(job_id=job_id, role=role, party_id=party_id, component_name=component_name,
                                   task_id=task_id,
                                   task_version=task_version,
                                   model_id=model_id,
                                   model_version=model_version,
                                   job_parameters=job_parameters)

    def log_job_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        self.log_metric_data_common(metric_namespace=metric_namespace, metric_name=metric_name, metrics=metrics,
                                    job_level=True)

    def log_metric_data(self, metric_namespace: str, metric_name: str, metrics: List[Metric]):
        self.log_metric_data_common(metric_namespace=metric_namespace, metric_name=metric_name, metrics=metrics,
                                    job_level=False)

    def log_metric_data_common(self, metric_namespace: str, metric_name: str, metrics: List[Metric], job_level=False):
        LOGGER.info("Request save job {} task {} {} on {} {} metric {} {} data".format(self.job_id,
                                                                                       self.task_id,
                                                                                       self.task_version,
                                                                                       self.role,
                                                                                       self.party_id,
                                                                                       metric_namespace,
                                                                                       metric_name))
        request_body = {}
        request_body['metric_namespace'] = metric_namespace
        request_body['metric_name'] = metric_name
        request_body['metrics'] = [serialize_b64(metric, to_str=True) for metric in metrics]
        request_body['job_level'] = job_level
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/metric_data/save'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        return response['retcode'] == RetCode.SUCCESS

    def set_job_metric_meta(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta):
        self.set_metric_meta_common(metric_namespace=metric_namespace, metric_name=metric_name, metric_meta=metric_meta,
                                    job_level=True)

    def set_metric_meta(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta):
        self.set_metric_meta_common(metric_namespace=metric_namespace, metric_name=metric_name, metric_meta=metric_meta,
                                    job_level=False)

    def set_metric_meta_common(self, metric_namespace: str, metric_name: str, metric_meta: MetricMeta, job_level=False):
        LOGGER.info("Request save job {} task {} {} on {} {} metric {} {} meta".format(self.job_id,
                                                                                       self.task_id,
                                                                                       self.task_version,
                                                                                       self.role,
                                                                                       self.party_id,
                                                                                       metric_namespace,
                                                                                       metric_name))
        request_body = dict()
        request_body['metric_namespace'] = metric_namespace
        request_body['metric_name'] = metric_name
        request_body['metric_meta'] = serialize_b64(metric_meta, to_str=True)
        request_body['job_level'] = job_level
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/metric_meta/save'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        return response['retcode'] == RetCode.SUCCESS

    def create_table_meta(self, table_meta):
        request_body = dict()
        for k, v in table_meta.to_dict().items():
            if k == "part_of_data":
                request_body[k] = serialize_b64(v, to_str=True)
            elif k == "schema":
                request_body[k] = serialize_b64(v, to_str=True)
            elif issubclass(type(v), AddressABC):
                request_body[k] = v.__dict__
            else:
                request_body[k] = v
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/table_meta/create'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        if response['retcode'] != RetCode.SUCCESS:
            raise Exception(f"create table meta failed:{response['retmsg']}")

    def get_table_meta(self, table_name, table_namespace):
        request_body = {"table_name": table_name, "namespace": table_namespace}
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/table_meta/get'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        if response['retcode'] != RetCode.SUCCESS:
            raise Exception(f"create table meta failed:{response['retmsg']}")
        else:
            data_table_meta = storage.StorageTableMeta(name=table_name,
                                                       namespace=table_namespace, new=True)
            data_table_meta.set_metas(**response["data"])
            data_table_meta.address = storage.StorageTableMeta.create_address(storage_engine=response["data"].get("engine"),
                                                                              address_dict=response["data"].get("address"))
            data_table_meta.part_of_data = deserialize_b64(data_table_meta.part_of_data)
            data_table_meta.schema = deserialize_b64(data_table_meta.schema)
            return data_table_meta

    def save_component_output_model(self, component_model):
        json_body = {"model_id": self.model_id, "model_version": self.model_version, "component_model": component_model}
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/component_model/save'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=json_body)
        if response['retcode'] != RetCode.SUCCESS:
            raise Exception(f"create table meta failed:{response['retmsg']}")

    def read_component_output_model(self, search_model_alias, tracker):
        json_body = {"search_model_alias": search_model_alias, "model_id": self.model_id, "model_version": self.model_version}
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/component_model/get'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=json_body)
        if response['retcode'] != RetCode.SUCCESS:
            raise Exception(f"create table meta failed:{response['retmsg']}")
        else:
            model_buffers = {}
            for model_name, v in response['data'].items():
                model_buffers[model_name] = tracker.pipelined_model.parse_proto_object(buffer_name=v[0], buffer_object_serialized_string=base64.b64decode(v[1].encode()))
            return model_buffers

    def log_output_data_info(self, data_name: str, table_namespace: str, table_name: str):
        LOGGER.info("Request save job {} task {} {} on {} {} data {} info".format(self.job_id,
                                                                                  self.task_id,
                                                                                  self.task_version,
                                                                                  self.role,
                                                                                  self.party_id,
                                                                                  data_name))
        request_body = dict()
        request_body["data_name"] = data_name
        request_body["table_namespace"] = table_namespace
        request_body["table_name"] = table_name
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/output_data_info/save'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        return response['retcode'] == RetCode.SUCCESS

    def get_output_data_info(self, data_name=None):
        LOGGER.info("Request read job {} task {} {} on {} {} data {} info".format(self.job_id,
                                                                                  self.task_id,
                                                                                  self.task_version,
                                                                                  self.role,
                                                                                  self.party_id,
                                                                                  data_name))
        request_body = dict()
        request_body["data_name"] = data_name
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/output_data_info/read'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        if response["retcode"] == RetCode.SUCCESS and "data" in response:
            return response["data"]
        else:
            return None

    def log_component_summary(self, summary_data: dict):
        LOGGER.info("Request save job {} task {} {} on {} {} component summary".format(self.job_id,
                                                                                       self.task_id,
                                                                                       self.task_version,
                                                                                       self.role,
                                                                                       self.party_id))
        request_body = dict()
        request_body["summary"] = summary_data
        response = api_utils.local_api(job_id=self.job_id,
                                       method='POST',
                                       endpoint='/tracker/{}/{}/{}/{}/{}/{}/summary/save'.format(
                                           self.job_id,
                                           self.component_name,
                                           self.task_id,
                                           self.task_version,
                                           self.role,
                                           self.party_id),
                                       json_body=request_body)
        return response['retcode'] == RetCode.SUCCESS
