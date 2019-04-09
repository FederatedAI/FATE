import grpc
import time
import json
import sys

from arch.api.proto import inference_service_pb2
from arch.api.proto import inference_service_pb2_grpc
import threading

def run(address):
    ths = []
    with grpc.insecure_channel(address) as channel:
        for i in range(1):
            th = threading.Thread(target=send, args=(channel, ))
            ths.append(th)
        st = int(time.time())
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        et = int(time.time())

def send(channel):
    stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
    request = inference_service_pb2.InferenceRequest()
    request.meta.sceneId = 50000
    request.meta.myPartyId = 10000
    request.meta.partnerPartyId = 9999
    request.meta.myRole = 'guest'

    request.model.name = "hetero_lr_guest_model"
    request.model.namespace = "hetero_lr"

    data = {}
    data["123456"] = {}
    data["123456"]["fid1"] = 5.1
    data["123456"]["fid2"] = 6.2
    data["123456"]["fid3"] = 7.6

    request.data = json.dumps(data).encode(encoding="utf-8")
    response = stub.predict(request)
    print(response)


if __name__ == '__main__':
    run(sys.argv[1])
