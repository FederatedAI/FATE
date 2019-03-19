import grpc
import time
import sys

from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
import threading

def run(model_id):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_service_pb2.PublishRequest()
        request.modelId = model_id
        response = stub.publishLoad(request)
        request.partnerPartyId = "100001"
        request.sceneId = "500001"
        request.myRole = "guest"
        response = stub.publishOnline(request)
        request.myRole = "host"
        response = stub.publishOnline(request)


if __name__ == '__main__':
    run(sys.argv[1])
