import pandas as pd
import numpy as np
import uuid
import time
import json
from numbers import Number
from arch.api import federation
from arch.api import session
from arch.api import RuntimeInstance
from arch.api.standalone.federation import FederationRuntime
from arch.api.utils import file_utils
from federatedml.secureprotol.fate_paillier import *
from federatedml.secureprotol.encrypt import *
from federatedml.util.param_checker import AllChecker
from contrib.fate_script import fate_script
from contrib.fate_script.blas.blas import *


def test_plain_lr():    
    from sklearn.datasets import make_moons
    import functools
    # 修改flow_id 否则内存表可能被覆盖
    session.init(mode=0)
    ns = str(uuid.uuid1())

    X = session.table('testX7', ns, partition=2)
    Y = session.table('testY7', ns, partition=2)

    b = np.array([0])
    eta = 1.2
    max_iter = 10

    total_num = 500

    _x, _y = make_moons(total_num, noise=0.25,random_state=12345)
    for i in range(np.shape(_y)[0]):
        X.put(i, _x[i])
        Y.put(i, _y[i])

    print(len([y for y in Y.collect()]))

    current_milli_time = lambda: int(round(time.time() * 1000))

    start = current_milli_time()
    #shape_w = [1, np.shape(_x)[1]]
    shape_w = [np.shape(_x)[1]]
    w = np.ones(shape_w)

    print(w)
    X = TensorInEgg(None,None,X)
    Y = TensorInEgg(None,None,Y)
    w = TensorInPy(None,None,w)
    b = TensorInPy(None, None, b)

    # lr = LR(shape_w)
    # lr.train(X, Y)
    itr = 0
    while itr < max_iter:
        H = 1 / X
        H = 1.0 / (1 + ((X @ w + b) * -1).map(np.exp))
        R = H - Y

        gradient_w = (R * X).sum() / total_num
        gradient_b = R.sum() / total_num
        w = w - eta * gradient_w
        b = b - eta * gradient_b
        print("aaa",w,b)
        # self.plot(itr)
        itr += 1

    print("train total time: {}".format(current_milli_time() - start))
    _x_test, _y_test = make_moons(50,random_state=12345)
    _x_test = TensorInPy(None,None, _x_test)
    y_pred = 1.0 / (1 + ((_x_test @ w + b) * -1).map(np.exp))
    from sklearn import metrics

    auc = metrics.roc_auc_score(_y_test, y_pred.store.reshape(50))
    print("auc: {}".format(auc))


def test_paillier_lr():    
    from sklearn.datasets import make_moons
    import functools

    
    #cipher = PaillierEncrypt()
    cipher = PaillierEncrypt()
    cipher.generate_key()

    # 修改flow_id 否则内存表可能被覆盖
    session.init(mode=0)
    ns = str(uuid.uuid1())
    p = True
    X_G = session.table('testX7', ns, partition=2, persistent=p)
    X_H = session.table('testX7_2', ns, partition=2, persistent=p)
    Y = session.table('testY7', ns, partition=2, persistent=p)

    b = np.array([0])
    eta = 1.2
    max_iter = 2 #00
    #max_iter = 100

    total_num = 500
    import pandas as pd
    data_H = pd.read_csv("/data/projects/qijun/fate/python/examples/data/breast_a.csv").values 
    data_G = pd.read_csv("/data/projects/qijun/fate/python/examples/data/breast_b.csv").values 
    print("shape",data_H.shape,data_G.shape,np.shape(data_H)[0])
    #_x, _y = make_moons(total_num, noise=0.25,random_state=12345)
    #for i in range(np.shape(_y)[0]):
    for i in range(np.shape(data_H)[0]):
        X_G.put(data_G[i][0], data_G[i][2:])
        X_H.put(data_H[i][0], data_H[i][1:])
        Y.put(data_G[i][0], 1 if data_G[i][1] == 1 else -1)
        #X_G.put(i, _x[i][:1])
        #X_H.put(i, _x[i][1:])
        #Y.put(i, _y[i])

    print(len([y for y in Y.collect()]))

    current_milli_time = lambda: int(round(time.time() * 1000))

    start = current_milli_time()
    #shape_w_G = [1, data_G.shape[1] - 2]
    shape_w_G = [data_G.shape[1] - 2]
    #shape_w_H = [1, data_H.shape[1] - 1]
    shape_w_H = [data_H.shape[1] - 1]
    w_G = np.zeros(shape_w_G)
    w_H = np.zeros(shape_w_H)
    #print("shape_w_H:{}".format(shape_w_H))
    #print("w_H:{}".format(w_H))
    #print(w_G,w_H)
    X_G = TensorInEgg(cipher,None,X_G)
    X_H = TensorInEgg(cipher,None,X_H)
    Y = TensorInEgg(cipher,None,Y)
    w_G = TensorInPy(cipher,None,w_G)
    w_H = TensorInPy(cipher,None,w_H)
    b = TensorInPy(cipher, None, b)
    #print("[Debug]W_G before iter:", w_G)

    # lr = LR(shape_w)
    # lr.train(X, Y)
    itr = 0
    pre_loss_A = None
    learning_rate=0.15
    while itr < max_iter:
        ep_time = current_milli_time()
        fw_H = X_H @ w_H
        #print("fw_H:", list(fw_H.store.collect()))
        #print("[Debug]w_H:{}, X_H:{}".format(w_H, X_H))
        enc_fw_H = fw_H.encrypt()
        enc_fw_square_H = (fw_H ** 2).encrypt()
        #print("[Debug]enc_fw_H:", enc_fw_H.store)
        #print("[Debug]enc_fw_square_H:", enc_fw_square_H.store)
        fw_G = X_G @ w_G  
        #print("fw_G:", list(fw_G.store.collect()))
        enc_fw_G = fw_G.encrypt()
        enc_fw_square_G = (fw_G ** 2).encrypt()

        enc_agg_wx_G = enc_fw_G + enc_fw_H 
        print("enc_fw_H", list(enc_fw_H.store.collect())[:10])
        print("enc_fw_G", list(enc_fw_H.store.collect())[:10])
        enc_agg_wx_square_G = enc_fw_square_G + enc_fw_square_H + 2 * fw_G * enc_fw_H
        print("enc_fw_square_H:", list(enc_fw_square_H.store.collect())[:10])
        print("enc_agg_wx_square_G:", list(enc_agg_wx_square_G.store.collect())[:10])

        enc_fore_grad_G = 0.25 * enc_agg_wx_G - 0.5 * Y
        print("enc_fore_grad_G:", list(enc_fore_grad_G.store.collect())[:10]) 
        enc_grad_G = (X_G * enc_fore_grad_G).mean()
        enc_grad_H = (X_H * enc_fore_grad_G).mean()
        print("iter:{} enc_grad_G:{}".format(itr, enc_grad_G))
        print("iter:{} enc_grad_H:{}".format(itr, enc_grad_H))
        grad_A = enc_grad_G.hstack(enc_grad_H)
        print("iter:{} grad_A:{}".format(itr, grad_A))
        #grad_b = enc_fore_grad_G.mean() * 0.2
        learning_rate *= 0.999
        optim_grad_A = grad_A * learning_rate 
        print("iter:{}, optim_grad_A:{}".format(itr, optim_grad_A))
        optim_grad_G,optim_grad_H = optim_grad_A.decrypt().split(shape_w_G[0])
        print("iter:{} optim_grad_G:{}".format(itr, optim_grad_G))
        print("iter:{} optim_grad_H:{}".format(itr, optim_grad_H))
        #print("[Debug]w_G:", w_G)
        #print("[Debug]w_H:", w_H)
        w_G = w_G - optim_grad_G
        w_H = w_H - optim_grad_H
        print("[Debug]iter:{} w_G:{}".format(itr, w_G))
        print("[Debug]iter:{} w_H:{}".format(itr, w_H))

        #b = b - grad_b
        print("enc_agg_wx_G:", list(enc_agg_wx_G.store.collect())[:10])
        enc_half_ywx_G = 0.5 * enc_agg_wx_G * Y

        print("enc_half_ywx_G", list(enc_half_ywx_G.store.collect())[:10])
        enc_loss_G = (enc_half_ywx_G * -1 + enc_agg_wx_square_G / 8 + np.log(2)).mean()
        print("enc_loss_G:", enc_loss_G)
        #todo: tensor compare
        loss_A = enc_loss_G.decrypt().store
        print("loss_A:{}, pre_loss_A:{}".format(loss_A, pre_loss_A))
        tmp = 99999 if pre_loss_A is None else loss_A - pre_loss_A
        #print("aaa:",b,w_G,w_H,loss_A,pre_loss_A,tmp, abs(tmp),current_milli_time() - ep_time)
        if pre_loss_A is not None and abs(loss_A - pre_loss_A) < 1e-4:
            break
        pre_loss_A = loss_A
        itr += 1

    print("train total time: {}".format(current_milli_time() - start))
    #_x_test, _y_test = make_moons(50,random_state=12345)
    #_x_test = TensorInPy(None,None, _x_test.transpose())
    _x_test = X_G.hstack(X_H)
    _y_test = np.array(list(Y.store.collect()))[:,1]
    w = w_G.hstack(w_H)
    print("w:", w)
    #print("_x_test:", list(_x_test.store.collect())[:10])
    #print("w * _x_test:", list((w *_x_test).store.collect()))
    y_pred = 1.0 / (1 + ((_x_test @ w + b) * -1).map(np.exp))
    y_pred = np.array(list(y_pred.store.collect()))[:,1]
    #print("blas:y_pred", y_pred)
    from sklearn import metrics

    auc = metrics.roc_auc_score(_y_test, y_pred)
    print("auc: {}".format(auc))

   
if __name__ == '__main__':
    #test_plain_lr()
    test_paillier_lr()
