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
import sys
import requests
import json
from fate_flow.settings import API_VERSION, HTTP_PORT

fate_flow_server_host = 'http://127.0.0.1:{}/{}'.format(HTTP_PORT, API_VERSION)
job_id = sys.argv[1]
role = sys.argv[2]
party_id = int(sys.argv[3])
base_request_data = {'job_id': job_id, 'role': role, 'party_id': party_id}
print('job id is {}'.format(job_id))
# data view
print('job data view')
response = requests.post('{}/tracking/job/data_view'.format(fate_flow_server_host), json=base_request_data)
print(response.json())
response = requests.post('{}/job/data/view/query'.format(fate_flow_server_host), json=base_request_data)
print(response.json())

# dependency
print('dependency')
response = requests.post('{}/pipeline/dag/dependency'.format(fate_flow_server_host),
                         json={'job_id': job_id, 'role': role, 'party_id': party_id})
dependency_response = response.json()
print(json.dumps(dependency_response))
print()
for component_name in dependency_response['data']['component_list']:
    print('component name is {}'.format(component_name))
    base_request_data['component_name'] = component_name
    # metrics
    print('metrics')
    response = requests.post('{}/tracking/component/metrics'.format(fate_flow_server_host), json=base_request_data)
    print(response.json())
    print('metrics return {}'.format(response.json()))
    print()
    if response.json()['retcode'] == 0 and response.json()['retmsg'] != "no data":
        for metric_namespace, metric_names in response.json()['data'].items():
            base_request_data['metric_namespace'] = metric_namespace
            for metric_name in metric_names:
                base_request_data['metric_name'] = metric_name
                response = requests.post('{}/tracking/component/metric_data'.format(fate_flow_server_host), json=base_request_data)
                if response.json()['retcode'] == 0 :
                    print('{} {} metric data:'.format(metric_namespace, metric_name))
                    print(response.json())
                else:
                    print('{} {} no metric data!'.format(metric_namespace, metric_name))
                print()

    # parameters
    print('parameters')
    response = requests.post('{}/tracking/component/parameters'.format(fate_flow_server_host), json=base_request_data)
    print(response.json())
    print('parameters retcode {}'.format(response.json()['retcode']))
    # model
    print('output model')
    response = requests.post('{}/tracking/component/output/model'.format(fate_flow_server_host), json=base_request_data)
    print(response.json())
    print('output model retcode {}'.format(response.json()['retcode']))
    # data
    print('output data')
    response = requests.post('{}/tracking/component/output/data'.format(fate_flow_server_host), json=base_request_data)
    print(response.json())
    print('output data retcode {}'.format(response.json()['retcode']))
    print()
    response = requests.post('{}/tracking/component/output/data/table'.format(fate_flow_server_host), json=base_request_data)
    print(response.json())
