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
            th = threading.Thread(target=send, args=(channel,))
            ths.append(th)
        st = int(time.time())
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        et = int(time.time())


def process_response(call_future):
    print(call_future.result())


def send(channel):
    stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
    request = inference_service_pb2.InferenceMessage()
    request_data = dict()
    request_data['serviceId'] = 'xxxxxxxxx'
    request_data['applyId'] = ''
    # request_data['modelId'] = 'arbiter-10000#guest-10000#host-10000#model'  #  You can specify the model id this way
    # request_data['modelVersion'] = 'acd3e1807a1211e9969aacde48001122' #  You can specify the model version this way
    request_data['caseid'] = uuid.uuid1().hex

    feature_data = dict()
    feature_data['fid1'] = 5.1
    feature_data['fid2'] = 6.2
    feature_data['fid3'] = 7.6
    request_data['featureData'] = feature_data
    request_data['sendToRemoteFeatureData'] = feature_data

    print(json.dumps(request_data, indent=4))

    request.body = json.dumps(request_data).encode(encoding='utf-8')
    print(stub.inference(request))


if __name__ == '__main__':
    run(sys.argv[1])
