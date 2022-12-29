from federatedml.nn.backend.torch import nn, init
import json
from federatedml.nn.backend.torch import serialization as s
import torch as t
from federatedml.nn.backend.torch.import_hook import fate_torch_hook
from federatedml.nn.backend.torch.cust import CustModel

fate_torch_hook(t)

cust_resnet = CustModel(name='resnet')
transformer = nn.Transformer()
seq = nn.Sequential(
    nn.Linear(10, 10),
    CustModel(name='lr', param={'input_size': 2}),
    CustModel(name='mf', param={'u_num': 100, 'i_num': 100, 'embd_dim': 32}),
    CustModel(name='resnet'),
    transformer,
)
nn_define_json = json.dumps(seq.to_dict(), indent=3)
nn_define = seq.to_dict()
recover_seq = s.recover_sequential_from_dict(nn_define)
