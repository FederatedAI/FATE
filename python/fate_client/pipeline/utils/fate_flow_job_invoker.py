import time
from datetime import timedelta
from .flow_client import FlowClient
from ..conf.env_config import FlowConfig


class FATEFlowJobInvoker(object):
    def __init__(self):
        self._client = FlowClient(ip=FlowConfig.IP, port=FlowConfig.PORT, version=FlowConfig.VERSION)

    def monitor_status(self, job_id, role, party_id):
        start_time = time.time()
        pre_task = None
        print(f"Job id is {job_id}\n")
        while True:
            response_data = self.query_job(job_id, role, party_id)
            status = response_data["status"]
            if status == JobStatus.SUCCESS:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                print(f"Job is success!!! Job id is {job_id}, response_data={response_data}")
                print(f"Total time: {elapse_seconds}")
                break

            elif status == JobStatus.RUNNING:
                code, data = self.query_task(job_id=job_id, role=role, party_id=party_id,
                                             status=JobStatus.RUNNING)

                if code != 0 or len(data) == 0:
                    time.sleep(0.1)
                    continue

                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                if len(data) == 1:
                    task = data[0]["task_name"]
                else:
                    task = []
                    for task_data in data:
                        task.append(task_data["task_name"])

                if task != pre_task:
                    print(f"\r")
                    pre_task = task
                print(f"\x1b[80D\x1b[1A\x1b[KRunning task {task}, time elapse: {elapse_seconds}")

            elif status == JobStatus.WAITING:
                elapse_seconds = timedelta(seconds=int(time.time() - start_time))
                print(f"\x1b[80D\x1b[1A\x1b[KJob is waiting, time elapse: {elapse_seconds}")

            elif status == JobStatus.FAILED:
                raise ValueError(f"Job is failed, please check out job_id={job_id} in fate_flow log directory")

            time.sleep(1)

    def submit_job(self, dag_schema):
        response = self._client.submit_job(dag_schema=dag_schema)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            job_id = response["job_id"]
            model_id = response["data"]["model_id"]
            model_version = response["data"]["model_version"]
            return job_id, model_id, model_version
        except BaseException:
            raise ValueError(f"submit job is failed, response={response}")

    def query_job(self, job_id, role, party_id):
        response = self._client.query_job(job_id, role, party_id)
        try:
            code = response["code"]
            if code != 0:
                raise ValueError(f"Return code {code}!=0")

            data = response["data"][0]
            return data
        except BaseException:
            raise ValueError(f"query job is failed, response={response}")

    def query_task(self, job_id, role, party_id, status):
        response = self._client.query_task(job_id, role, party_id, status)
        try:
            code = response["code"]
            data = response.get("data", [])
            return code, data
        except BaseException:
            raise ValueError(f"query task is failed, response={response}")

    def query_site_info(self):
        response = self._client.query_site_info()
        try:
            party_id = response["data"]["party_id"]
            return '9999'
            # TODO: fix it later
            # return party_id
        except BaseException:
            raise ValueError(f"query site info is failed, response={response}")


class JobStatus(object):
    SUCCESS = "success"
    RUNNING = "running"
    WAITING = "waiting"
    FAILED = "failed"
