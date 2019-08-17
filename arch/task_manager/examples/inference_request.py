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
    request_data['appid'] = "9999"
    # request_data['modelId'] = '50000_guest_9999_10000-9999-10000_model'  #  You can specify the model id this way
    # request_data['modelVersion'] = 'acd3e1807a1211e9969aacde48001122' #  You can specify the model version this way
    request_data['caseid'] = uuid.uuid1().hex
    #request_data['caseid'] = '2d2926107fa411e9a06a00e04c6c66f9'

    feature_data = {}
    feature_data["device_id"] = 'e4ceacfe-ea49-49cb-9e48-a124235f897f'
    feature_data["fid1"] = 5.1
    feature_data["fid2"] = 6.2
    feature_data["fid3"] = 7.6
    request_data['featureData'] = feature_data

    print(json.dumps(request_data, indent=4))

    request.body = json.dumps(request_data).encode(encoding="utf-8")
    # print(stub.inference(request))
    print(stub.startInferenceJob(request))
    """
    call_future = stub.inference.future(request)
    call_future.add_done_callback(process_response)
    time.sleep(5)
    """

    get_result_request = inference_service_pb2.InferenceMessage()
    get_result_request_data = dict()
    get_result_request_data['appid'] = "9999"
    # get_result_request_data['caseid'] = uuid.uuid1().hex
    get_result_request_data['caseid'] = request_data['caseid']
    get_result_request.body = json.dumps(get_result_request_data).encode(encoding="utf-8")
    time.sleep(5)
    print(stub.getInferenceResult(get_result_request))




if __name__ == '__main__':
    run(sys.argv[1])
