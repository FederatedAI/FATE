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
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

from flow_sdk.client import FlowClient
from pipeline.backend import config as conf
from pipeline.backend.config import IODataType
from pipeline.backend.config import JobStatus
from pipeline.backend.config import StatusCode
from pipeline.utils.logger import LOGGER


class JobFunc:
    SUBMIT_JOB = "submit_job"
    UPLOAD = "upload"
    COMPONENT_OUTPUT_MODEL = "component_output_model"
    COMPONENT_METRIC = "component_metric_all"
    JOB_STATUS = "query_job"
    TASK_STATUS = "query_task"
    COMPONENT_OUTPUT_DATA = "component_output_data"
    COMPONENT_OUTPUT_DATA_TABLE = "component_output_data_table"
    DEPLOY_COMPONENT = "deo"


class JobInvoker(object):
    def __init__(self):
        self.client = FlowClient()

    @classmethod
    def _run_cmd(cls, cmd, output_while_running=False):
        subp = subprocess.Popen(cmd,
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if not output_while_running:
            stdout, stderr = subp.communicate()
            return stdout.decode("utf-8")
        else:
            for line in subp.stdout:
                if line == "":
                    continue
                else:
                    # print(line.strip())
                    LOGGER.debug(f"{line.strip()}")

    @LOGGER.catch
    def submit_job(self, dsl=None, submit_conf=None):
        dsl_path = None
        with tempfile.TemporaryDirectory() as job_dir:
            if dsl:
                dsl_path = os.path.join(job_dir, "job_dsl.json")
                # pprint.pprint(dsl)
                LOGGER.info(f"submit dsl is: \n {json.dumps(dsl, indent=4, ensure_ascii=False)}")
                with open(dsl_path, "w") as fout:
                    fout.write(json.dumps(dsl))

            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(submit_conf))

            result = self.client.job.submit(conf_path=submit_path, dsl_path=dsl_path)
            try:
                if 'retcode' not in result or result["retcode"] != 0:
                    # LOGGER.opt(exception=True).error(f"retcode err")
                    raise ValueError

                if "jobId" not in result:
                    # LOGGER.opt(exception=True).error(f"jobID not in result: {result}")
                    raise ValueError

                job_id = result["jobId"]
                data = result["data"]
            except ValueError:
                # LOGGER.opt(exception=True).error(f"job submit failed, err msg: {result}")
                # raise
                raise ValueError("job submit failed, err msg: {}".format(result))

        return job_id, data

    @LOGGER.catch
    def upload_data(self, submit_conf=None, drop=0):
        with tempfile.TemporaryDirectory() as job_dir:
            submit_path = os.path.join(job_dir, "job_runtime_conf.json")
            with open(submit_path, "w") as fout:
                fout.write(json.dumps(submit_conf))

            result = self.client.data.upload(conf_path=submit_path, verbose=1, drop=drop)
            try:
                if 'retcode' not in result or result["retcode"] != 0 or result["retcode"] != 0:
                    # LOGGER.opt(exception=True).error(f"retcode err")
                    raise ValueError

                if "jobId" not in result:
                    # LOGGER.opt(exception=True).error(f"jobID not in result: {result}")
                    raise ValueError

                job_id = result["jobId"]
                data = result["data"]
            except ValueError:
                # LOGGER.opt(exception=True).error(f"job submit failed, err msg: {result}")
                # raise
                raise ValueError("job submit failed, err msg: {}".format(result))

        return job_id, data

    @LOGGER.catch
    def monitor_job_status(self, job_id, role, party_id):
        party_id = str(party_id)
        start_time = time.time()
        pre_cpn = None
        # print ("Job id is {}".format(job_id))
        LOGGER.info(f"Job id is {job_id}")
        while True:
            ret_code, ret_msg, data = self.query_job(job_id, role, party_id)
            status = data["f_status"]
            if status == JobStatus.COMPLETE:
                # print("job is success!!!")
                LOGGER.info(f"Job is success!!! Job id is {job_id}")
                return StatusCode.SUCCESS

            if status == JobStatus.FAILED:
                # print("job is failed, please check out job {} by fate board or fate_flow cli".format(job_id))
                LOGGER.error(f"Job is failed, please check out job {job_id} by fate board or fate_flow cli")
                return StatusCode.FAIL

            if status == JobStatus.WAITING:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                # print("job is still waiting, time elapse: {}".format(elapse_seconds), end="\r", flush=True)
                sys.stdout.write(f"Job is still waiting, time elapse: {elapse_seconds}\r")
                sys.stdout.flush()
                # LOGGER.info(f"Job is still waiting, time elapse: {elapse_seconds}")

            if status == JobStatus.RUNNING:
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
                    print("\n", end="\r")
                    sys.stdout.write(f"\n \r")
                    pre_cpn = cpn
                """
                print("Running component {}, time elpase: {}".format(cpn,
                                                                     elapse_seconds), end="\r",
                      flush=True)
                """
                sys.stdout.write(f"Running component {cpn}, time elapse: {elapse_seconds}\r")
                sys.stdout.flush()
                # LOGGER.info(f"Running component {cpn}, time elapse: {elapse_seconds}")

            time.sleep(conf.TIME_QUERY_FREQS)

    @LOGGER.catch
    def query_job(self, job_id, role, party_id):
        party_id = str(party_id)
        result = self.client.job.query(job_id=job_id, role=role, party_id=party_id)
        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError("can not query_job")
                # LOGGER.opt(exception=True).error("can not query_job")

            ret_code = result["retcode"]
            ret_msg = result["retmsg"]
            data = result["data"][0]
            return ret_code, ret_msg, data
        except ValueError:
            raise ValueError("query job result is {}, can not parse useful info".format(result))
            # LOGGER.opt(exception=True).error(f"query job result is {result}, can not parse useful info. err msg: ")

    @LOGGER.catch
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
        #print(f"in get_output_data_table, component {cpn_name} result is {result}")
        data = {}
        try:
            if 'retcode' not in result or result["retcode"] != 0:
                raise ValueError
                # LOGGER.opt(exception=True).error(f"get output data table failed")

            if "data" not in result:
                raise ValueError
                # LOGGER.opt(exception=True).error(f"data not in result: {result}")

            all_data = result["data"]
            n = len(all_data)
            # single data table
            if n == 1:
                single_data = all_data[0]
                # data_name = single_data["data_name"]
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
                # print(f"No output data table found in {result}.")
                LOGGER.info(f"No output data table fount in {result}")

        except ValueError:
            # LOGGER.opt(exception=True).error(f"job submit failed, err msg: {result}")
            # raise
            raise ValueError("Job submit failed, err msg: {}".format(result))
        return data

    @LOGGER.catch
    def query_task(self, job_id, role, party_id, status=None):
        party_id = str(party_id)
        result = self.client.task.query(job_id=job_id, role=role,
                                        party_id=party_id, status=status)
        try:
            if 'retcode' not in result:
                raise ValueError("Cannot query task status of job {}".format(job_id))
                # LOGGER.opt(exception=True).error(f"can not query task status of job {job_id}")

            ret_code = result["retcode"]
            ret_msg = result["retmsg"]

            if ret_code != 0:
                data = None
            else:
                data = result["data"]
            return ret_code, ret_msg, data
        except ValueError:
            # LOGGER.opt(exception=True).error(f"query task result is {result}, can not parse useful info. err msg: ")
            # raise
            raise ValueError("Query task result is {}, cannot parse useful info".format(result))

    @LOGGER.catch
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
            # single output data
            if n == 1:
                data_dict = JobInvoker.create_data_meta_dict(IODataType.SINGLE, output_dir, limits)
            # multiple output data
            elif n > 1:
                data_dict = {}
                for data_name in [IODataType.TRAIN, IODataType.VALIDATE, IODataType.TEST]:
                    curr_data_dict = JobInvoker.create_data_meta_dict(data_name, output_dir, limits)
                    data_dict[data_name] = curr_data_dict
            # no output data obtained
            else:
                LOGGER.error(f"No output data found in directory{output_dir}")
                # print(f"No output data found in directory {output_dir}.")
            return data_dict

    @staticmethod
    def create_data_meta_dict(data_name, output_dir, limits):
        data_file = f"{data_name}.csv"
        meta_file = f"{data_name}.meta"

        output_data = os.path.join(output_dir, data_file)
        output_meta = os.path.join(output_dir, meta_file)
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
        #print(f"{output_data}: {data[:10]}")
        return data

    @staticmethod
    def extract_output_meta(output_meta):
        with open(output_meta, "r") as fin:
            try:
                meta_dict = json.load(fin)
                meta = meta_dict["header"]
            except ValueError:
                # print("Can not get output data meta.")
                LOGGER.error(f"Cannot get output data meta. err msg: ")
                # raise

        # print(f"{output_meta}: {meta}")
        return meta

    def get_model_param(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.output_model(job_id=job_id, role=role,
                                                        party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                # print("job {}, component {} has no output model param".format(job_id, cpn_name))
                LOGGER.error(f"job {job_id}, component {cpn_name} has no output model param")
                return
            return result["data"]
        except:
            # print("Can not get output model, err msg is {}".format(result))
            LOGGER.error("Cannot get output model, err msg: ")
            # raise

    def get_metric(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.metric_all(job_id=job_id, role=role,
                                                      party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                # print("job {}, component {} has no output metric".format(job_id, cpn_name))
                LOGGER.error(f"job {job_id}, component {cpn_name} has no output metric")
                return
            return result["data"]
        except:
            # print("Can not get output model, err msg is {}".format(result))
            LOGGER.error("Cannot get ouput model, err msg: ")
            # raise

    def get_summary(self, job_id, cpn_name, role, party_id):
        result = None
        party_id = str(party_id)
        try:
            result = self.client.component.get_summary(job_id=job_id, role=role,
                                                       party_id=party_id, component_name=cpn_name)
            if "data" not in result:
                # print("job {}, component {} has no output metric".format(job_id, cpn_name))
                LOGGER.error(f"Job {job_id}, component {cpn_name} has no output metric")
                return
            return result["data"]
        except:
            # print("Can not get output model, err msg is {}".format(result))
            LOGGER.error("Cannot get output model, err msg: ")
            # raise

    @LOGGER.catch
    def get_predict_dsl(self, train_dsl, cpn_list, version):
        result = None
        with tempfile.TemporaryDirectory() as job_dir:
            train_dsl_path = os.path.join(job_dir, "train_dsl.json")
            with open(train_dsl_path, "w") as fout:
                fout.write(json.dumps(train_dsl))

            result = self.client.job.generate_dsl(train_dsl_path=train_dsl_path, cpn_list=cpn_list, version=version)

        if result is None or 'retcode' not in result:
            # LOGGER.opt(exception=True).error(f"call flow generate dsl is failed, check if fate_flow server is start!")
            raise ValueError("Call flow generate dsl is failed, check if fate_flow server is start!")
        elif result["retcode"] != 0:
            # LOGGER.opt(exception=True).error(f"Cannot generate predict dsl, error msg is {result['retmsg']}")
            raise ValueError("Cannot generate predict dsl, error msg is {}".format(result["retmsg"]))
        else:
            return result["data"]
