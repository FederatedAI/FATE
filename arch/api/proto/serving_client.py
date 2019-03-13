import grpc
import time

import serving_pb2
import serving_pb2_grpc
import threading

def run():
    ths = []
    with grpc.insecure_channel('localhost:50051') as channel:
        for i in range(10000):
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
    stub = serving_pb2_grpc.PredictionServiceStub(channel)
    request = serving_pb2.PredictRequest()
    request.modelId = "1212121"

    request.data["123456"].floatData["v0"] = 10
    request.data["123456"].floatData["v1"] = 10
    request.data["123456"].stringData["v2"] = "1234"

    response = stub.predict(request)
    #print(response)
    #print("%d Client received: %s" % (threading.currentThread().ident, request.msg))
    """
    for request in stub.GetListMsg(msg_pb2.MsgRequest(name='world', id=5)):
        print("%d Client received: %s" % (threading.currentThread().ident, request.msg))
    """


if __name__ == '__main__':
    run()
