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

import json
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path

from flow_sdk.client import FlowClient
from pipeline.backend import config as conf
from pipeline.backend.config import IODataType
from pipeline.backend.config import JobStatus
from pipeline.backend.config import StatusCode
from pipeline.utils.logger import LOGGER


class JobInvoker(object):
    def __init__(self):
        self.client = FlowClient(ip=conf.FlowConfig.IP, port=conf.FlowConfig.PORT, version=conf.SERVER_VERSION,
                                 app_key=conf.FlowConfig.APP_KEY, secret_key=conf.FlowConfig.SECRET_KEY)

    def submit_job(self, dsl=None, submit_conf=None, callback_func=None):
        """
        dsl_path = None
        with tempfile.TemporaryDirectory() as job_dir:
            if dsl:
                dsl_path = os.path.join(job_dir, "job_dsl.json")
                # pprint.pprint(dsl)
                LOGGER.debug(f"submit dsl is: \n {json.dumps(dsl, indent=4, ensure_ascii=False)}")
                with open(dsl_path, "w") as fout:
                    fout.write(json.dumps(dsl))
            LOGGER.debug(f"submit conf is: \n {json.dumps(submit_conf, indent=4, ensure_ascii=False)}")
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(submit_conf))

            result = self.client.job.submit(conf_path=submit_path, dsl_path=dsl_path)
            if callback_func is not None:
                callback_func(result)
            try:
                if 'retcode' not in result or result["retcode"] != 0:
                    raise ValueError(f"retcode err")

                if "jobId" not in result:
                    raise ValueError(f"jobID not in result: {result}")

                job_id = result["jobId"]
                data = result["data"]
            except ValueError:
                raise ValueError("job submit failed, err msg: {}".format(result))
        """
        LOGGER.debug(f"submit dsl is: \n {json.dumps(dsl, indent=4, ensure_ascii=False)}")
        LOGGER.debug(f"submit conf is: \n {json.dumps(submit_conf, indent=4, ensure_ascii=False)}")
        result = self.client.job.submit(config_data=submit_conf, dsl_data=dsl)
        if callback_func is not None:
            callback_func(result)
        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError(f"retcode err")

            if "jobId" not in result:
                raise ValueError(f"jobID not in result: {result}")

            job_id = result["jobId"]
            data = result["data"]
        except ValueError:
            raise ValueError("job submit failed, err msg: {}".format(result))
        return job_id, data

    def upload_data(self, submit_conf=None, drop=0):
        """
        with tempfile.TemporaryDirectory() as job_dir:
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(submit_conf))

            result = self.client.data.upload(conf_path=submit_path, verbose=1, drop=drop)
            try:
                if 'retcode' not in result or result["retcode"] != 0:
                    raise ValueError

                if "jobId" not in result:
                    raise ValueError

                job_id = result["jobId"]
                data = result["data"]
            except:
                raise ValueError("job submit failed, err msg: {}".format(result))
        """
        result = self.client.data.upload(config_data=submit_conf, verbose=1, drop=drop)
        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError

            if "jobId" not in result:
                raise ValueError

            job_id = result["jobId"]
            data = result["data"]
        except BaseException:
            raise ValueError("job submit failed, err msg: {}".format(result))
        return job_id, data

    def monitor_job_status(self, job_id, role, party_id):
        party_id = str(party_id)
        start_time = time.time()
        pre_cpn = None
        LOGGER.info(f"Job id is {job_id}\n")
        while True:
            ret_code, ret_msg, data = self.query_job(job_id, role, party_id)
            status = data["f_status"]
            if status == JobStatus.SUCCESS:
                # print("job is success!!!")
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                # sys.stdout.write(f"\n\r")
                LOGGER.info(f"Job is success!!! Job id is {job_id}")
                LOGGER.info(f"Total time: {elapse_seconds}")
                return StatusCode.SUCCESS

            elif status == JobStatus.FAILED:
                # sys.stdout.write(f"\n\r")
                # LOGGER.info(f"\n\r")
                raise ValueError(f"Job is failed, please check out job {job_id} by fate board or fate_flow cli")

            elif status == JobStatus.WAITING:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                # sys.stdout.write(f"\r")
                # sys.stdout.flush()
                LOGGER.info(f"\x1b[80D\x1b[1A\x1b[KJob is still waiting, time elapse: {elapse_seconds}")

            elif status == JobStatus.CANCELED:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                # sys.stdout.write(f"\n\r")
                LOGGER.info(f"Job is canceled, time elapse: {elapse_seconds}\r")
                return StatusCode.CANCELED

            elif status == JobStatus.TIMEOUT:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                # sys.stdout.write(f"\n\r")
                raise ValueError(f"Job is timeout, time elapse: {elapse_seconds}\r")

            elif status == JobStatus.RUNNING:
                ret_code, _, data = self.query_task(job_id=job_id, role=role, party_id=party_id,
                                                    status=JobStatus.RUNNING)
                if ret_code != 0 or len(data) == 0:
                    time.sleep(conf.TIME_QUERY_FREQS)
                    continue

                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                if len(data) == 1:
                    cpn = data[0]["f_component_name"]
                else:
                    cpn = []
                    for cpn_data in data:
                        cpn.append(cpn_data["f_component_name"])

                if cpn != pre_cpn:
                    LOGGER.info(f"\r")
                    pre_cpn = cpn
                # sys.stdout.write(f"\r")
                # sys.stdout.flush()
                LOGGER.info(f"\x1b[80D\x1b[1A\x1b[KRunning component {cpn}, time elapse: {elapse_seconds}")

            else:
                raise ValueError(f"Unknown status: {status}")

            time.sleep(conf.TIME_QUERY_FREQS)

    def query_job(self, job_id, role, party_id):
        party_id = str(party_id)
        result = self.client.job.query(job_id=job_id, role=role, party_id=party_id)
        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError("can not query_job")

            ret_code = result["retcode"]
            ret_msg = result["retmsg"]
            data = result["data"][0]
            return ret_code, ret_msg, data
        except ValueError:
            raise ValueError("query job result is {}, can not parse useful info".format(result))

    def get_output_data_table(self, job_id, cpn_name, role, party_id):
        """

        Parameters
        ----------
        job_id: str
        cpn_name: str
        role: str
        party_id: int

        Returns
        -------
        dict
        single output example:
            {
                table_name: [],
                table_namespace: []

            }
        multiple output example:
            {
            train_data: {
                table_name: [],
                table_namespace: []
                },
            validate_data: {
                table_name: [],
                table_namespace: []
                }
            test_data: {
                table_name: [],
                table_namespace: []
                }
            }
        """
        party_id = str(party_id)
        result = self.client.component.output_data_table(job_id=job_id, role=role,
                                                         party_id=party_id, component_name=cpn_name)
        data = {}
        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError

            if "data" not in result:
                raise ValueError

            all_data = result["data"]
            n = len(all_data)
            # single data table
            if n == 1:
                single_data = all_data[0]
                del single_data["data_name"]
                data = single_data
            # multiple data table
            elif n > 1:
                for single_data in all_data:
                    data_name = single_data["data_name"]
                    del single_data["data_name"]
                    data[data_name] = single_data
            # no data table obtained
            else:
                LOGGER.info(f"No output data table found in {result}")

        except ValueError:
            raise ValueError("Job submit failed, err msg: {}".format(result))
        return data

    def query_task(self, job_id, role, party_id, status=None):
        party_id = str(party_id)
        result = self.client.task.query(job_id=job_id, role=role,
                                        party_id=party_id, status=status)
        try:
            if 'retcode' not in result:
                raise ValueError("Cannot query task status of job {}".format(job_id))

            ret_code = result["retcode"]
            ret_msg = result["retmsg"]

            if ret_code != 0:
                data = None
            else:
                data = result["data"]
            return ret_code, ret_msg, data
        except ValueError:
            raise ValueError("Query task result is {}, cannot parse useful info".format(result))

    def get_output_data(self, job_id, cpn_name, role, party_id, limits=None):
        """

        Parameters
        ----------
        job_id: str
        cpn_name: str
        role: str
        party_id: int
        limits: int, None, default None. Maximum number of lines returned, including header. If None, return all lines.

        Returns
        -------
        dict
        single output example:
            {
                data: [],
                meta: []

            }
        multiple output example:
            {
            train_data: {
                data: [],
                meta: []
                },
            validate_data: {
                data: [],
                meta: []
                }
            test_data: {
                data: [],
                meta: []
                }
            }
        """
        party_id = str(party_id)
        with tempfile.TemporaryDirectory() as job_dir:
            result = self.client.component.output_data(job_id=job_id, role=role, output_path=job_dir,
                                                       party_id=party_id, component_name=cpn_name)
            output_dir = result["directory"]
            n = 0
            for file in os.listdir(output_dir):
                if file.endswith("csv"):
                    n += 1

            if n > 0:
                data_dict = {}
                for data_name in [IODataType.SINGLE, IODataType.TRAIN, IODataType.VALIDATE, IODataType.TEST]:
                    curr_data_dict = JobInvoker.create_data_meta_dict(data_name, output_dir, limits)
                    if curr_data_dict is not None:
                        data_dict[data_name] = curr_data_dict
            # no output data obtained
            else:
                raise ValueError(f"No output data found in directory{output_dir}")
            if len(data_dict) == 1:
                return list(data_dict.values())[0]
            return data_dict

    @staticmethod
    def create_data_meta_dict(data_name, output_dir, limits):
        data_file = f"{data_name}.csv"
        meta_file = f"{data_name}.meta"

        output_data = os.path.join(output_dir, data_file)
        output_meta = os.path.join(output_dir, meta_file)
        if not Path(output_data).resolve().exists():
            return
        data = JobInvoker.extract_output_data(output_data, limits)
        meta = JobInvoker.extract_output_meta(output_meta)
        data_dict = {"data": data, "meta": meta}
        return data_dict

    @staticmethod
    def extract_output_data(output_data, limits):
        data = []
        with open(output_data, "r") as fin:
            for i, line in enumerate(fin):
                if i == limits:
                    break
                data.append(line.strip())
        return data

    @staticmethod
    def extract_output_meta(output_meta):
        with open(output_meta, "r") as fin:
            try:
                meta_dict = json.load(fin)
                meta = meta_dict["header"]
            except ValueError:
                raise ValueError(f"Cannot get output data meta. err msg: ")

        return meta

    def get_model_param(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.output_model(job_id=job_id, role=role,
                                                        party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                raise ValueError(f"job {job_id}, component {cpn_name} has no output model param")
            return result["data"]
        except BaseException:
            raise ValueError("Cannot get output model, err msg: ")
            # raise

    def get_metric(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.metric_all(job_id=job_id, role=role,
                                                      party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                raise ValueError(f"job {job_id}, component {cpn_name} has no output metric")
            return result["data"]
        except BaseException:
            raise ValueError("Cannot get ouput model, err msg: ")
            # raise

    def get_summary(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.get_summary(job_id=job_id, role=role,
                                                       party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                # print("job {}, component {} has no output metric".format(job_id, cpn_name))
                raise ValueError(f"Job {job_id}, component {cpn_name} has no output metric")
            return result["data"]
        except BaseException:
            raise ValueError("Cannot get output model, err msg: ")

    def model_deploy(self, model_id, model_version, cpn_list=None, predict_dsl=None, components_checkpoint=None):
        if cpn_list:
            result = self.client.model.deploy(model_id=model_id, model_version=model_version, cpn_list=cpn_list)
        elif predict_dsl:
            result = self.client.model.deploy(model_id=model_id, model_version=model_version,
                                              predict_dsl=predict_dsl, components_checkpoint=components_checkpoint)
        else:
            result = self.client.model.deploy(model_id=model_id, model_version=model_version,
                                              components_checkpoint=components_checkpoint)

        if result is None or 'retcode' not in result:
            raise ValueError("Call flow deploy is failed, check if fate_flow server is up!")
        elif result["retcode"] != 0:
            raise ValueError("Cannot deploy components, error msg is {}".format(result["retmsg"]))
        else:
            return result["data"]

    def get_predict_dsl(self, model_id, model_version):
        result = self.client.model.get_predict_dsl(model_id=model_id, model_version=model_version)
        if result is None or 'retcode' not in result:
            raise ValueError("Call flow get predict dsl is failed, check if fate_flow server is up!")
        elif result["retcode"] != 0:
            raise ValueError("Cannot get predict dsl, error msg is {}".format(result["retmsg"]))
        else:
            return result["data"]

    def load_model(self, load_conf):
        """
        with tempfile.TemporaryDirectory() as job_dir:
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(load_conf))
        result = self.client.model.load(submit_path)
        """
        result = self.client.model.load(load_conf)
        if result is None or 'retcode' not in result:
            raise ValueError("Call flow load failed, check if fate_flow server is up!")
        elif result["retcode"] != 0:
            raise ValueError("Cannot load model, error msg is {}".format(result["retmsg"]))
        else:
            return result["data"]

    def bind_model(self, bind_conf):
        """
        with tempfile.TemporaryDirectory() as job_dir:
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(bind_conf))
        result = self.client.model.bind(submit_path)
        """
        result = self.client.model.bind(bind_conf)
        if result is None or 'retcode' not in result:
            raise ValueError("Call flow bind failed, check if fate_flow server is up!")
        elif result["retcode"] != 0:
            raise ValueError("Cannot bind model, error msg is {}".format(result["retmsg"]))
        else:
            return result["retmsg"]

    def convert_homo_model(self, convert_conf):
        """
        with tempfile.TemporaryDirectory() as job_dir:
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(convert_conf))
        result = self.client.model.homo_convert(submit_path)
        """
        result = self.client.model.homo_convert(convert_conf)
        if result is None or 'retcode' not in result:
            raise ValueError("Call flow homo convert failed, check if fate_flow server is up!")
        elif result["retcode"] != 0:
            raise ValueError("Cannot convert homo model, error msg is {}".format(result["retmsg"]))
        else:
            return result["data"]
