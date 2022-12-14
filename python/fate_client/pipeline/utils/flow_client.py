import requests


class Address(object):
    SUBMIT_JOB = "/job/submit"
    QUERY_JOB = "/job/query"
    STOP_JOB = "/job/stop"
    QUERY_TASK = "/job/task/query"

    def __init__(self, server_url):
        self._submit_job_url = server_url + self.SUBMIT_JOB
        self._query_job_url = server_url + self.QUERY_JOB
        self._stop_job_url = server_url + self.STOP_JOB
        self._query_task_url = server_url + self.QUERY_TASK

    @property
    def submit_job_url(self):
        return self._submit_job_url

    @property
    def query_job_url(self):
        return self._query_job_url

    @property
    def stop_job_url(self):
        return self._stop_job_url

    @property
    def query_task_url(self):
        return self._query_task_url


class FlowClient(object):
    def __init__(self, ip, port, version):
        self._server_url = f"http://{ip}:{port}/{version}"
        self._address = Address(server_url=self._server_url)

    def submit_job(self, dag_schema):
        response = requests.post(
            self._address.submit_job_url,
            json={
                "dag_schema": dag_schema
            }
        )

        return response.json()

    def query_job(self, job_id, role, party_id):
        response = requests.post(
            self._address.query_task_url,
            json={
                "job_id": job_id,
                "role": role,
                "party_id": party_id
            }
        )

        return response.json()

    def stop_job(self, job_id):
        response = requests.post(
            self._address.stop_job_url,
            json={
                "job_id": job_id
            }
        )

        return response.json()

    def query_task(self, job_id, role, party_id, status):
        response = requests.post(
            self._address.query_task_url,
            json={
                "job_id": job_id,
                "role": role,
                "party_id": party_id,
                "status": status
            }
        )

        return response.json()
