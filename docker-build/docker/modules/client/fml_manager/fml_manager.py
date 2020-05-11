#
#  Copyright 2020 The KubeFATE Authors. All Rights Reserved.
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
#
import json
import os
import tarfile
import requests
import base64
import random
import time

from contextlib import closing
from arch.api.utils import file_utils
from arch.api.utils.core_utils import get_lan_ip


class FMLManager:
    def __init__(self, server_conf="/data/projects/fate/python/arch/conf/server_conf.json", log_path="./"):
        self.server_conf = file_utils.load_json_conf(server_conf)
        self.ip = self.server_conf.get("servers").get("fateflow").get("host")
        self.serving_uri = self.server_conf.get("servings")
        self.http_port = self.server_conf.get("servers").get("fateflow").get("http.port")
        self.server_url = "http://{}:{}/{}".format(self.ip, self.http_port, "v1")
        if self.server_conf.get("servers").get("fml_agent"):
            fml_agent_host = self.server_conf.get("servers").get("fml_agent").get("host")
            fml_agent_port = self.server_conf.get("servers").get("fml_agent").get("port")
            self.fml_agent_url = "http://{}:{}/{}".format(fml_agent_host, fml_agent_port, "api")
        self.log_path = log_path

    # Job management
    def submit_job(self, dsl, config):
        post_data = {'job_dsl': dsl,
                     'job_runtime_conf': config}
        response = requests.post("/".join([self.server_url, "job", "submit"]), json=post_data)

        return self.prettify(response)

    def submit_job_by_files(self, dsl_path, config_path):
        config_data = {}
        if config_path:
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise Exception('Conf cannot be null.')
        dsl_data = {}
        if dsl_path:
            dsl_path = os.path.abspath(dsl_path)
            with open(dsl_path, 'r') as f:
                dsl_data = json.load(f)
        else:
            raise Exception('DSL_path cannot be null.')


        return self.submit_job(dsl_data, config_data)

    def query_job(self, query_conditions):
        response = requests.post("/".join([self.server_url, "job", "query"]), json=query_conditions)
        return self.prettify(response)

    def query_job_conf(self, query_conditions):
        response = requests.post("/".join([self.server_url, "job", "config"]), json=query_conditions)
        return self.prettify(response)

    def stop_job(self, job_id):
        post_data = {
            'job_id': job_id
        }
        response = requests.post("/".join([self.server_url, "job", "stop"]), json=post_data)
        return self.prettify(response)

    def update_job(self, job_id, role, party_id, notes):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "notes": notes
        }
        response = requests.post("/".join([self.server_url, "job", "update"]), json=post_data)
        return self.prettify(response)

    def fetch_job_log(self, job_id):
        data = {
            "job_id": job_id
        }

        tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
        extract_dir = os.path.join(self.log_path, 'job_{}_log'.format(job_id))
        with closing(requests.get("/".join([self.server_url, "job", "log"]), json=data,
                                      stream=True)) as response:
            if response.status_code == 200:
                self.__download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                response = {'retcode': 0,
                            'directory': extract_dir,
                            'retmsg': 'download successfully, please check {} directory, file name is {}'.format(extract_dir, tar_file_name)}

                return self.prettify(response, True)
            else:
                return self.prettify(response, True)

    # Data management
    def load_data(self, url, namespace, table_name, work_mode, head, partition, api_version="1.4"):
        if api_version == "1.4":
            temp_file = None
            if url.startswith("http://") or url.startswith("https://"):
                downloader = HttpDownloader(url)
                temp_file = downloader.download_to(file_utils.get_project_base_directory())
                url = temp_file

            post_data = {
                "namespace": namespace,
                "table_name": table_name,
                "work_mode": work_mode,
                "head": head,
                "partition": partition
            }
            data_files = {
                "file": open(url, "rb")
            }
            response = requests.post("/".join([self.server_url, "data", "upload"]), data=post_data, files=data_files)

            if temp_file is not None and os.path.exists(temp_file):
                print("Delete temp file...")
                os.remove(temp_file)
        else:
            post_data = {
                "file": url,
                "namespace": namespace,
                "table_name": table_name,
                "work_mode": work_mode,
                "head": head,
                "partition": partition
            }
            response = requests.post("/".join([self.server_url, "data", "upload"]), json=post_data)

        return self.prettify(response)

    def query_data(self, job_id, limit):
        post_data = {
            "job_id": job_id,
            "limit": limit
        }

        response = requests.post("/".join([self.server_url, "data", "upload", "history"]), json=post_data)

        return self.prettify(response)

    # The data is download to fateflow. FATE not ready to download to local.
    def download_data(self, namespace, table_name, filename, work_mode, delimitor, output_folder = "./"):
        if not hasattr(self, "fml_agent_url"):
            raise Exception("Cannot download data! Please start fml_agent in fateflow.")

        DEFAULT_DATA_FOLDER = "/data/projects/fate/python/fml_agent/data"
        output_path = "{}/{}".format(DEFAULT_DATA_FOLDER, filename)
        post_data = {
            "namespace": namespace,
            "table_name": table_name,
            "work_mode": work_mode,
            "delimitor": delimitor,
            "output_path": output_path
        }
        response = requests.post("/".join([self.server_url, "data", "download"]), json=post_data)

        if response.status_code == 200:
            output = json.loads(response.content)
            job_id = output["jobId"]
            query_condition = {
                "job_id":job_id
            }
            for i in range(500):
                time.sleep(1)
                status = self.query_job(query_condition).json()["data"][0]["f_status"]
                if status == "failed":
                    print("Failed")
                    raise Exception("Failed to download data.")
                if status == "success":
                    with closing(requests.get("/".join([self.fml_agent_url, "download", filename]), stream=True)) as response:
                        if response.status_code == 200:
                            output = "{}/{}".format(output_folder, filename)
                            self.__download_data_from_request(response, output)
                            response = {
                            'retcode': 0,
                            'retmsg': 'download successfully, please check {} directory, file name is {}'.format(output_folder, filename)}
                            return self.prettify(response, True)
                    break

        response = {
            'retcode': 1,
            'retmsg': 'Download failed'
        }

        return self.prettify(response, True)

    # Model management
    def load_model(self, initiator_party_id, federated_roles, work_mode, model_id, model_version):
        post_data = {
            "initiator": {
                "party_id": initiator_party_id,
                "role": "guest"
            },
            "role": federated_roles,
            "job_parameters": {
                "work_mode": work_mode,
                "model_id": model_id,
                "model_version": model_version
            }
        }
        response = requests.post("/".join([self.server_url, "model", "load"]), json=post_data)

        return self.prettify(response)

    def bind_model(self, service_id, initiator_party_id, federated_roles, work_mode, model_id, model_version):
        if self.serving_uri == nil:
            raise Exception('Federated Serving is not deployed or not correctly configured yet. ')
        post_data = {
            "service_id": service_id,
            "initiator": {
                "party_id": initiator_party_id,
                "role": "guest"
            },
            "role": federated_roles,
            "job_parameters": {
                "work_mode": work_mode,
                "model_id": model_id,
                "model_version": model_version
            },
            "servings": self.serving_uri
        }

        response = requests.post("/".join([self.server_url, "model", "bind"]), json=post_data)
        return self.prettify(response)

    def print_model_version(self, role, party_id, model_id):
        namespace = "#".join([role, party_id, model_id])
        post_data = {
            "namespace": namespace
        }

        response = requests.post("/".join([self.server_url, "model", "version"]), json=post_data)

        return self.prettify(response, True)

    def model_output(self, role, party_id, model_id, model_version, model_component):
        namespace = "#".join([role, party_id, model_id])
        post_data = {
            "name": model_version,
            "namespace": namespace
        }
        response = requests.post("/".join([self.server_url, "model", "transfer"]), json=post_data)
        model = json.loads(response.content)
        if model["data"] != "":
            en_model_metadata = model["data"]["%sMeta" % model_component]
            en_model_parameters = model["data"]["%sParam" % model_component]

        model = {
            "metadata": en_model_metadata,
            "parameters": en_model_parameters
        }

        return self.prettify(model, True)

    def offline_predict_on_dataset(self, is_vertical, initiator_party_role, initiator_party_id, work_mode, model_id, model_version, federated_roles, guest_data_name = "", guest_data_namespace = "", host_data_name = "", host_data_namespace = ""):
        if is_vertical:
            print("This API is not support vertical federated machine learning yet. ")
            return

        # For predict job, dsl is empty dict.
        dsl = {}

        config = {
            "initiator": {
                "role": initiator_party_role,
                "party_id": initiator_party_id
            },
            "job_parameters": {
                "work_mode": work_mode,
                "job_type": "predict",
                "model_id": model_id,
                "model_version": model_version
            },
            "role": federated_roles,
            "role_parameters": {}
        }

        if guest_data_name != "" or guest_data_namespace != "":
            if initiator_party_role != "guest":
                raise Exception("Initiator not has data sets.")

            guest_parameters = {
                "args": {
                    "data": {
                        "eval_data": [{"name": guest_data_name, "namespace": guest_data_namespace}]
                    }
                }
            }

            config["role_parameters"]["guest"] = guest_parameters

        if host_data_name != "" or host_data_namespace != "":
            host_parameters = {
                "args": {
                    "data": {
                        "eval_data": [{"name": host_data_name, "namespace": host_data_namespace}]
                    }
                }
            }
            config["role_parameters"]["host"] = guest_parameters

        return self.submit_job(dsl, config)

    # Task
    def query_task(self, query_conditions):
        response = requests.post("/".join([self.server_url, "job", "task", "query"]), json=query_conditions)
        return self.prettify(response)

    # Tracking
    def track_job_data(self, job_id, role, party_id):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id
        }

        response = requests.post("/".join([self.server_url, "tracking", "job", "data_view"]), json=post_data)
        return self.prettify(response, True)

    def track_component_all_metric(self, job_id, role, party_id, component_name):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "component_name": component_name
        }

        response = requests.post("/".join([self.server_url, "tracking", "component", "metric", "all"]), json=post_data)
        return self.prettify(response, True)

    def track_component_metric_type(self, job_id, role, party_id, component_name):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "component_name": component_name
        }

        response = requests.post("/".join([self.server_url, "tracking", "component", "metrics"]), json=post_data)
        return self.prettify(response, True)

    """
    metric_name and metric_namespace can be found in API track_component_metric_type
    e.g. response = manager.track_component_metric_type(jobId, "guest", "10000", "homo_lr_0") 
        {
            "data": {
                "train": [
                    "loss"
                ]
            },
            "retcode": 0,
            "retmsg": "success"
        }

        The metric_name is "loss" and metric_namespace is "train"
    """
    def track_component_metric_data(self, job_id, role, party_id, component_name, metric_name, metric_namespace):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "component_name": component_name,
            "metric_name": metric_name,
            "metric_namespace": metric_namespace
        }

        response = requests.post("/".join([self.server_url, "tracking", "component", "metric_data"]), json=post_data)
        return self.prettify(response, True)

    def track_component_parameters(self, job_id, role, party_id, component_name):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "component_name": component_name
        }

        response = requests.post("/".join([self.server_url, "tracking", "component", "parameters"]), json=post_data)
        return self.prettify(response, True)

    def track_component_output_model(self, job_id, role, party_id, component_name):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "component_name": component_name
        }

        response = requests.post("/".join([self.server_url, "tracking", "component", "output", "model"]), json=post_data)
        return self.prettify(response, True)

    def track_component_output_data(self, job_id, role, party_id, component_name):
        post_data = {
            "job_id": job_id,
            "role": role,
            "party_id": party_id,
            "component_name": component_name
        }

        response = requests.post("/".join([self.server_url, "tracking", "component", "output", "data"]), json=post_data)
        return self.prettify(response, True)

    # Utils
    def prettify(self, response, verbose=False):
        if verbose:
            if isinstance(response, requests.Response):
                if response.status_code == 200:
                    print("Success!")
                print(json.dumps(response.json(), indent=4, ensure_ascii=False))
            else:
                print(response)

        return response

    def __download_data_from_request(self, http_response, output):
     with open(output, 'wb') as fw:
        for chunk in http_response.iter_content(1024):
            if chunk:
                fw.write(chunk)

    def __download_from_request(self, http_response, tar_file_name, extract_dir):
        with open(tar_file_name, 'wb') as fw:
            for chunk in http_response.iter_content(1024):
                if chunk:
                    fw.write(chunk)
        tar = tarfile.open(tar_file_name, "r:gz")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, extract_dir)
        tar.close()
        os.remove(tar_file_name)

class HttpDownloader:
    def __init__(self, url):
        self.url = url

    def download_to(self, path_to_save):
        r = requests.get(self.url, allow_redirects=True)
        filename = __get_filename_from_cd(r.headers.get('content-disposition'))
        temp_file_to_write = os.path.join(file_utils.get_project_base_directory(), filename)
        open(temp_file_to_write, 'wb').write(r.content)

        return temp_file_to_write

    def __get_filename_from_cd(cd):
        """
        Get filename from content-disposition
        """
        if not cd:
            return None
        fname = re.findall('filename=(.+)', cd)
        if len(fname) == 0:
            return None
        return fname[0]
