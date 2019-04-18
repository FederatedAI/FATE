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
import grpc
from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
from arch.task_manager.settings import logger


def load_model(config_data):
    for serving in config_data.get('servings'):
        with grpc.insecure_channel(serving) as channel:
            stub = model_service_pb2_grpc.ModelServiceStub(channel)
            request = model_service_pb2.PublishRequest()
            request.myPartyId = config_data.get("my_party_id")
            for party_id, model in config_data.get("models").items():
                request.models[int(party_id)].name = model["name"]
                request.models[int(party_id)].namespace = model["namespace"]
            response = stub.publishLoad(request)
            logger.info("party_id: {}, serving server: {}, load status: {}".format(request.myPartyId, serving, response.statusCode))


def publish_online(config_data):
    for serving in config_data.get('servings'):
        with grpc.insecure_channel(serving) as channel:
            stub = model_service_pb2_grpc.ModelServiceStub(channel)
            request = model_service_pb2.PublishRequest()
            request.myPartyId = config_data.get("partyId")
            for party_id, model in config_data.get("models").items():
                request.models[int(party_id)].name = model["name"]
                request.models[int(party_id)].namespace = model["namespace"]
                response = stub.publishOnline(request)
                print(response)
