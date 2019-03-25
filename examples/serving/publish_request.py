import grpc
import time
import sys

from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
import threading

def run(model_id):
    with grpc.insecure_channel('localhost:7777') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_service_pb2.PublishRequest()
        request.commitId = model_id
        request.sceneId = "50000"
        request.myPartyId = "9999"
        request.partnerPartyId = "10000"
        request.myRole = "host"
        response = stub.publishLoad(request)
        print(response)
    with grpc.insecure_channel('localhost:7778') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_service_pb2.PublishRequest()
        request.commitId = model_id
        request.sceneId = "50000"
        request.myPartyId = "10000"
        request.partnerPartyId = "9999"
        request.myRole = "guest"
        response = stub.publishLoad(request)
        response = stub.publishOnline(request)
        print(response)


if __name__ == '__main__':
    run(sys.argv[1])
