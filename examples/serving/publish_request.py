import grpc
import time
import sys

from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
import threading

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_service_pb2.PublishRequest()
        request.sceneId = "50000"
        request.myPartyId = "9999"
        request.partnerPartyId = "10000"
        request.myRole = "host"
        request.commitId = "1271cc984cb211e99267acde48001122"
        response = stub.publishLoad(request)
        print(response)
        response = stub.publishOnline(request)
        print(response)


if __name__ == '__main__':
    run()
