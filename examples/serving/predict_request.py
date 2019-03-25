import grpc
import time

from arch.api.proto import prediction_service_pb2
from arch.api.proto import prediction_service_pb2_grpc
import threading

def run():
    ths = []
    with grpc.insecure_channel('localhost:7778') as channel:
        for i in range(1):
            th = threading.Thread(target=send, args=(channel, ))
            ths.append(th)
        st = int(time.time())
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        et = int(time.time())
        print(et - st)

def send(channel):
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = prediction_service_pb2.PredictRequest()
    request.meta.sceneId = '50000'
    request.meta.myPartyId = '10000'
    request.meta.partnerPartyId = "9999"
    request.meta.myRole = 'guestUser'

    request.data["123456"].floatData["k1"] = 5
    request.data["123456"].floatData["k2"] = 3
    request.data["123456"].floatData["k3"] = 4

    response = stub.predict(request)
    print(response)
    #print("%d Client received: %s" % (threading.currentThread().ident, request.msg))
    """
    for request in stub.GetListMsg(msg_pb2.MsgRequest(name='world', id=5)):
        print("%d Client received: %s" % (threading.currentThread().ident, request.msg))
    """


if __name__ == '__main__':
    run()
