import grpc
import time
import json
import sys
import uuid

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
    request = inference_service_pb2.InferenceMessage()
    request_data = dict()
    request_data['sceneid'] = 50000
    #request_data['modelNamespace'] = '50000_guest_9999_10000-9999-10000_model'  #  You can specify the model namespace this way
    #request_data['modelName'] = 'acd3e1807a1211e9969aacde48001122' #  You can specify the model name this way
    request_data['seqno'] = uuid.uuid1().hex
    request_data['caseid'] = uuid.uuid1().hex

    feature_data = {}
    feature_data["phone_num"] = "18576637870"
    feature_data["device_id"] = "xxxxxxxxxx"
    feature_data["fid1"] = 5.1
    feature_data["fid2"] = 6.2
    feature_data["fid3"] = 7.6
    request_data['featureData'] = feature_data

    request.data = json.dumps(request_data).encode(encoding="utf-8")
    response = stub.inference(request)
    print(response)


if __name__ == '__main__':
    run(sys.argv[1])
