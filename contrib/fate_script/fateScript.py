import sys
from sklearn import metrics
from contrib.fate_script import fate_script
from contrib.fate_script.blas.blas import *
from contrib.fate_script.utils.fate_script_transfer_variable import *
__site__ = sys.argv[1]
__job_id__ = sys.argv[2]
__conf_path = sys.argv[3]
__work_mode = sys.argv[4]
fate_script.init(__job_id__, __conf_path, int(__work_mode))
transfer_variable = HeteroLRTransferVariable()
fate_script.init_encrypt_operator()
if __site__ == "A":
    paillier_pubkey =fate_script.init_public_key()

    fate_script.remote(paillier_pubkey, name =transfer_variable.paillier_pubkey.name, tag = (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), role = 'D', idx = 0) # from A
    print("A finish remote paillier_pubkey(name:{}, tag:{}) to D".format(transfer_variable.paillier_pubkey.name, transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(paillier_pubkey, name =transfer_variable.paillier_pubkey.name, tag = (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), role = 'H', idx = 0) # from A
    print("A finish remote paillier_pubkey(name:{}, tag:{}) to H".format(transfer_variable.paillier_pubkey.name, transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(paillier_pubkey, name =transfer_variable.paillier_pubkey.name, tag = (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from A
    print("A finish remote paillier_pubkey(name:{}, tag:{}) to G".format(transfer_variable.paillier_pubkey.name, transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(paillier_pubkey, name =transfer_variable.paillier_pubkey.name, tag = (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), role = 'E', idx = 0) # from A
    print("A finish remote paillier_pubkey(name:{}, tag:{}) to E".format(transfer_variable.paillier_pubkey.name, transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]))
if __site__ == "H":
    paillier_pubkey=fate_script.get(transfer_variable.paillier_pubkey.name, (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), idx = 0) # to H
    print("H finish getting paillier_pubkey from A")
    fate_script.get_public_key(paillier_pubkey)
if __site__ == "H":
    pub_key =paillier_pubkey
if __site__ == "D":
    paillier_pubkey=fate_script.get(transfer_variable.paillier_pubkey.name, (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), idx = 0) # to D
    print("D finish getting paillier_pubkey from A")
    fate_script.get_public_key(paillier_pubkey)
if __site__ == "D":
    pub_key =paillier_pubkey
if __site__ == "G":
    paillier_pubkey=fate_script.get(transfer_variable.paillier_pubkey.name, (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
    print("G finish getting paillier_pubkey from A")
    fate_script.get_public_key(paillier_pubkey)
if __site__ == "G":
    pub_key =paillier_pubkey
if __site__ == "E":
    paillier_pubkey=fate_script.get(transfer_variable.paillier_pubkey.name, (transfer_variable.paillier_pubkey.name + '.' + str(__job_id__)[-14:]), idx = 0) # to E
    print("E finish getting paillier_pubkey from A")
    fate_script.get_public_key(paillier_pubkey)
if __site__ == "E":
    pub_key =paillier_pubkey
if __site__ == "H":
    X =fate_script.get_lr_x_table("/data/projects/fate/python/examples/data/breast_a.csv")
if __site__ == "H":
    W=fate_script.get_lr_w("/data/projects/fate/python/examples/data/breast_a.csv")
if __site__ == "G":
    X =fate_script.get_lr_x_table("/data/projects/fate/python/examples/data/breast_b.csv")
if __site__ == "G":
    W=fate_script.get_lr_w("/data/projects/fate/python/examples/data/breast_b.csv")
if __site__ == "G":
    Y =fate_script.get_lr_y_table("/data/projects/fate/python/examples/data/breast_b.csv")
if __site__ == "G":
    shape_w =fate_script.get_lr_shape_w("/data/projects/fate/python/examples/data/breast_b.csv")

    fate_script.remote(shape_w, name =transfer_variable.shape_w.name, tag = (transfer_variable.shape_w.name + '.' + str(__job_id__)[-14:]), role = 'A', idx = 0) # from G
    print("G finish remote shape_w(name:{}, tag:{}) to A".format(transfer_variable.shape_w.name, transfer_variable.shape_w.name + '.' + str(__job_id__)[-14:]))
if __site__ == "A":
    ml_conf =fate_script.init_ml_conf()

    fate_script.remote(ml_conf, name =transfer_variable.ml_conf.name, tag = (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), role = 'H', idx = 0) # from A
    print("A finish remote ml_conf(name:{}, tag:{}) to H".format(transfer_variable.ml_conf.name, transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(ml_conf, name =transfer_variable.ml_conf.name, tag = (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), role = 'D', idx = 0) # from A
    print("A finish remote ml_conf(name:{}, tag:{}) to D".format(transfer_variable.ml_conf.name, transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(ml_conf, name =transfer_variable.ml_conf.name, tag = (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from A
    print("A finish remote ml_conf(name:{}, tag:{}) to G".format(transfer_variable.ml_conf.name, transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(ml_conf, name =transfer_variable.ml_conf.name, tag = (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), role = 'R', idx = 0) # from A
    print("A finish remote ml_conf(name:{}, tag:{}) to R".format(transfer_variable.ml_conf.name, transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]))

    fate_script.remote(ml_conf, name =transfer_variable.ml_conf.name, tag = (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), role = 'E', idx = 0) # from A
    print("A finish remote ml_conf(name:{}, tag:{}) to E".format(transfer_variable.ml_conf.name, transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]))
if __site__ == "H":
    ml_conf=fate_script.get(transfer_variable.ml_conf.name, (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), idx = 0) # to H
    print("H finish getting ml_conf from A")
if __site__ == "H":
    ml_conf =ml_conf
if __site__ == "D":
    ml_conf=fate_script.get(transfer_variable.ml_conf.name, (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), idx = 0) # to D
    print("D finish getting ml_conf from A")
if __site__ == "D":
    ml_conf =ml_conf
if __site__ == "G":
    ml_conf=fate_script.get(transfer_variable.ml_conf.name, (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
    print("G finish getting ml_conf from A")
if __site__ == "G":
    ml_conf =ml_conf
if __site__ == "E":
    ml_conf=fate_script.get(transfer_variable.ml_conf.name, (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), idx = 0) # to E
    print("E finish getting ml_conf from A")
if __site__ == "E":
    ml_conf =ml_conf
if __site__ == "R":
    ml_conf=fate_script.get(transfer_variable.ml_conf.name, (transfer_variable.ml_conf.name + '.' + str(__job_id__)[-14:]), idx = 0) # to R
    print("R finish getting ml_conf from A")
if __site__ == "R":
    ml_conf =ml_conf
if __site__ == "A":
    is_stopped =False
if __site__ == "A":
    pre_loss =None
for iter_idx in range(ml_conf.iter_num):
    for batch_idx in range(ml_conf.batch_num):
        if __site__ == "H":
            forward =X @ W
        if __site__ == "H":
            _enc_forward = fate_script.tensor_encrypt(forward)

            fate_script.remote(_enc_forward, name =transfer_variable._enc_forward.name, tag = (transfer_variable._enc_forward.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from H
            print("H finish remote _enc_forward(name:{}, tag:{}) to G".format(transfer_variable._enc_forward.name, transfer_variable._enc_forward.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "H":
            _enc_forward_square = fate_script.tensor_encrypt(forward**2)

            fate_script.remote(_enc_forward_square, name =transfer_variable._enc_forward_square.name, tag = (transfer_variable._enc_forward_square.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from H
            print("H finish remote _enc_forward_square(name:{}, tag:{}) to G".format(transfer_variable._enc_forward_square.name, transfer_variable._enc_forward_square.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "G":
            _enc_forward=fate_script.get(transfer_variable._enc_forward.name, (transfer_variable._enc_forward.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
            print("G finish getting _enc_forward from H")
        if __site__ == "G":
            _enc_forward_h =_enc_forward
        if __site__ == "G":
            _enc_forward_square=fate_script.get(transfer_variable._enc_forward_square.name, (transfer_variable._enc_forward_square.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
            print("G finish getting _enc_forward_square from H")
        if __site__ == "G":
            _enc_forward_square_h =_enc_forward_square
        if __site__ == "G":
            forward =X @ W
        if __site__ == "G":
            _enc_forward = fate_script.tensor_encrypt(forward)
        if __site__ == "G":
            _enc_forward_square = fate_script.tensor_encrypt(forward**2)
        if __site__ == "G":
            _enc_agg_wx =_enc_forward + _enc_forward_h
        if __site__ == "G":
            _enc_agg_wx_square =_enc_forward_square + _enc_forward_square_h + 2 * forward * _enc_forward_h
        if __site__ == "G":
            _enc_fore_gradient =0.25 * _enc_agg_wx - 0.5 * Y

            fate_script.remote(_enc_fore_gradient, name =transfer_variable._enc_fore_gradient.name, tag = (transfer_variable._enc_fore_gradient.name + '.' + str(__job_id__)[-14:]), role = 'H', idx = 0) # from G
            print("G finish remote _enc_fore_gradient(name:{}, tag:{}) to H".format(transfer_variable._enc_fore_gradient.name, transfer_variable._enc_fore_gradient.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "G":
            _enc_grad_G =(X * _enc_fore_gradient).mean()

            fate_script.remote(_enc_grad_G, name =transfer_variable._enc_grad_G.name, tag = (transfer_variable._enc_grad_G.name + '.' + str(__job_id__)[-14:]), role = 'A', idx = 0) # from G
            print("G finish remote _enc_grad_G(name:{}, tag:{}) to A".format(transfer_variable._enc_grad_G.name, transfer_variable._enc_grad_G.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "H":
            _enc_fore_gradient=fate_script.get(transfer_variable._enc_fore_gradient.name, (transfer_variable._enc_fore_gradient.name + '.' + str(__job_id__)[-14:]), idx = 0) # to H
            print("H finish getting _enc_fore_gradient from G")
        if __site__ == "H":
            _enc_grad_H =(X * _enc_fore_gradient).mean()

            fate_script.remote(_enc_grad_H, name =transfer_variable._enc_grad_H.name, tag = (transfer_variable._enc_grad_H.name + '.' + str(__job_id__)[-14:]), role = 'A', idx = 0) # from H
            print("H finish remote _enc_grad_H(name:{}, tag:{}) to A".format(transfer_variable._enc_grad_H.name, transfer_variable._enc_grad_H.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "A":
            _enc_grad_G=fate_script.get(transfer_variable._enc_grad_G.name, (transfer_variable._enc_grad_G.name + '.' + str(__job_id__)[-14:]), idx = 0) # to A
            print("A finish getting _enc_grad_G from G")
        if __site__ == "A":
            _enc_grad_g =_enc_grad_G
        if __site__ == "A":
            _enc_grad_H=fate_script.get(transfer_variable._enc_grad_H.name, (transfer_variable._enc_grad_H.name + '.' + str(__job_id__)[-14:]), idx = 0) # to A
            print("A finish getting _enc_grad_H from H")
        if __site__ == "A":
            _enc_grad_h =_enc_grad_H
        if __site__ == "A":
            grad = fate_script.tensor_decrypt(_enc_grad_g.hstack(_enc_grad_h))
        if __site__ == "A":
            (ml_conf.learning_rate) =(ml_conf.learning_rate) * 0.999
        if __site__ == "A":
            optim_grad =grad * ml_conf.learning_rate
        if __site__ == "A":
            shape_w=fate_script.get(transfer_variable.shape_w.name, (transfer_variable.shape_w.name + '.' + str(__job_id__)[-14:]), idx = 0) # to A
            print("A finish getting shape_w from G")
        if __site__ == "A":
            shape_w =shape_w
        if __site__ == "A":
            optim_grad_g =optim_grad.split(shape_w[0])[0]

            fate_script.remote(optim_grad_g, name =transfer_variable.optim_grad_g.name, tag = (transfer_variable.optim_grad_g.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from A
            print("A finish remote optim_grad_g(name:{}, tag:{}) to G".format(transfer_variable.optim_grad_g.name, transfer_variable.optim_grad_g.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "A":
            optim_grad_h =optim_grad.split(shape_w[0])[1]

            fate_script.remote(optim_grad_h, name =transfer_variable.optim_grad_h.name, tag = (transfer_variable.optim_grad_h.name + '.' + str(__job_id__)[-14:]), role = 'H', idx = 0) # from A
            print("A finish remote optim_grad_h(name:{}, tag:{}) to H".format(transfer_variable.optim_grad_h.name, transfer_variable.optim_grad_h.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "G":
            optim_grad_g=fate_script.get(transfer_variable.optim_grad_g.name, (transfer_variable.optim_grad_g.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
            print("G finish getting optim_grad_g from A")
        if __site__ == "G":
            optim_grad_G =optim_grad_g
        if __site__ == "H":
            optim_grad_h=fate_script.get(transfer_variable.optim_grad_h.name, (transfer_variable.optim_grad_h.name + '.' + str(__job_id__)[-14:]), idx = 0) # to H
            print("H finish getting optim_grad_h from A")
        if __site__ == "H":
            optim_grad_H =optim_grad_h
        if __site__ == "G":
            W =W - optim_grad_G
        if __site__ == "H":
            W =W - optim_grad_H
        if __site__ == "G":
            _enc_half_ywx =0.5 * _enc_agg_wx * Y
        if __site__ == "G":
            _enc_loss =(_enc_half_ywx * (-1) + _enc_agg_wx_square / 8 + np.log(2)).mean()

            fate_script.remote(_enc_loss, name =transfer_variable._enc_loss.name, tag = (transfer_variable._enc_loss.name + '.' + str(__job_id__)[-14:]), role = 'A', idx = 0) # from G
            print("G finish remote _enc_loss(name:{}, tag:{}) to A".format(transfer_variable._enc_loss.name, transfer_variable._enc_loss.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "A":
            _enc_loss=fate_script.get(transfer_variable._enc_loss.name, (transfer_variable._enc_loss.name + '.' + str(__job_id__)[-14:]), idx = 0) # to A
            print("A finish getting _enc_loss from G")
        if __site__ == "A":
            loss = fate_script.tensor_decrypt(_enc_loss)
        if __site__ == "A":
            if pre_loss is not None and abs(loss.store - pre_loss.store) < ml_conf.eps :
                is_stopped = True
        if __site__ == "A":
            if pre_loss is None or abs(loss.store - pre_loss.store) >= ml_conf.eps :
                if __site__ == "A":
                    is_stopped =False

                    fate_script.remote(is_stopped, name =transfer_variable.is_stopped.name, tag = (transfer_variable.is_stopped.name + '.' + str(__job_id__)[-14:]), role = 'H', idx = 0) # from A
                    print("A finish remote is_stopped(name:{}, tag:{}) to H".format(transfer_variable.is_stopped.name, transfer_variable.is_stopped.name + '.' + str(__job_id__)[-14:]))

                    fate_script.remote(is_stopped, name =transfer_variable.is_stopped.name, tag = (transfer_variable.is_stopped.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from A
                    print("A finish remote is_stopped(name:{}, tag:{}) to G".format(transfer_variable.is_stopped.name, transfer_variable.is_stopped.name + '.' + str(__job_id__)[-14:]))
        if __site__ == "A":
            pre_loss =loss
        if __site__ == "A":
            if is_stopped:
                break
        if __site__ == "G":
            is_stopped=fate_script.get(transfer_variable.is_stopped.name, (transfer_variable.is_stopped.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
            print("G finish getting is_stopped from A")
        if __site__ == "G":
            is_stopped =is_stopped
        if __site__ == "G":
            if is_stopped:
                break
        if __site__ == "H":
            is_stopped=fate_script.get(transfer_variable.is_stopped.name, (transfer_variable.is_stopped.name + '.' + str(__job_id__)[-14:]), idx = 0) # to H
            print("H finish getting is_stopped from A")
        if __site__ == "H":
            is_stopped =is_stopped
        if __site__ == "H":
            if is_stopped:
                break
        if __site__ == "A":
            if iter_idx >= 2:
                break
        if __site__ == "G":
            if iter_idx >= 2:
                break
        if __site__ == "H":
            if iter_idx >= 2:
                break
if __site__ == "H":
    Z =X @ W

    fate_script.remote(Z, name =transfer_variable.Z.name, tag = (transfer_variable.Z.name + '.' + str(__job_id__)[-14:]), role = 'G', idx = 0) # from H
    print("H finish remote Z(name:{}, tag:{}) to G".format(transfer_variable.Z.name, transfer_variable.Z.name + '.' + str(__job_id__)[-14:]))
if __site__ == "G":
    Z=fate_script.get(transfer_variable.Z.name, (transfer_variable.Z.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
    print("G finish getting Z from H")
if __site__ == "G":
    Z_h =Z
if __site__ == "G":
    Z =X @ W
if __site__ == "G":
    Z_agg =Z + Z_h
if __site__ == "G":
    Y_test =np.array(list(Y.store.collect()))[:,1]
if __site__ == "G":
    Y_pred =1.0 / (1 + (Z_agg * -1).map(np.exp))
if __site__ == "G":
    Y_pred =np.array(list(Y_pred.store.collect()))[:,1]
if __site__ == "G":
    auc =metrics.roc_auc_score(Y_test, Y_pred)
if __site__ == "G":
    print("auc: {}".format(auc))
