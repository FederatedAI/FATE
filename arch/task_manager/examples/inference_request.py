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
    request.local.role = 'guest'
    request.local.partyId = 9999
    request.role['guest'].partyId.append(9999)
    request.role['host'].partyId.append(10000)
    request.role['arbiter'].partyId.append(10000)
    request.sceneId = 50000

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
