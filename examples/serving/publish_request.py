import grpc
import time
import sys

from arch.api.proto import model_service_pb2
from arch.api.proto import model_service_pb2_grpc
import threading

def run(model_id):
    with grpc.insecure_channel('localhost:8001') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_service_pb2.PublishRequest()
        request.myPartyId = 10000
        request.models[9999].name = "hetero_lr_host_model_hetero_logistic_regression_example_standalone_20190322185246"
        request.models[9999].namespace = "hetero_lr"
        request.models[10000].name = "hetero_lr_guest_model_hetero_logistic_regression_example_standalone_20190322185246"
        request.models[10000].namespace = "hetero_lr"
        response = stub.publishLoad(request)
        print(response)
        response = stub.publishOnline(request)
        print(response)

    with grpc.insecure_channel('localhost:8000') as channel:
        stub = model_service_pb2_grpc.ModelServiceStub(channel)
        request = model_service_pb2.PublishRequest()
        request.myPartyId = 9999
        request.models[9999].name = "hetero_lr_host_model_hetero_logistic_regression_example_standalone_20190322185246"
        request.models[9999].namespace = "hetero_lr"
        request.models[10000].name = "hetero_lr_guest_model_hetero_logistic_regression_example_standalone_20190322185246"
        request.models[10000].namespace = "hetero_lr"
        response = stub.publishLoad(request)
        print(response)


if __name__ == '__main__':
    #run(sys.argv[1])
    run("")
