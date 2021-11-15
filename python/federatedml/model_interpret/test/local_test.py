import pandas as pd
import numpy as np
from fate_arch.session import computing_session as session
from federatedml.feature.instance import Instance


session.init('1', 0)


def get_breast_guest():
    f = open("/home/cwj/FATE/standalone-fate-master-1.4.5/examples/data/breast_hetero_guest.csv", 'r')
    data = f.read()
    data = data.split('\n')

    header = data[0].split(',')

    convert_data = []
    for d in data[1:-1]:
        d_list = d.split(',')
        d_list = [float(i) for i in d_list]
        convert_data.append(d_list)

    data_inst_list = []

    for d in convert_data:
        id_ = int(d[0])
        label = int(d[1])
        data_inst_list.append((id_, Instance(features=np.array(d[2:]), label=label)))

    table = session.parallelize(data_inst_list, partition=16, include_key=True)
    table.schema['header'] = header[2:]
    return table


def get_breast_host():
    f = open("/home/cwj/FATE/standalone-fate-master-1.4.5/examples/data/breast_hetero_host.csv", 'r')
    data = f.read()
    data = data.split('\n')

    header = data[0].split(',')

    convert_data = []
    for d in data[1:-1]:
        d_list = d.split(',')
        d_list = [float(i) for i in d_list]
        convert_data.append(d_list)

    data_inst_list = []

    for d in convert_data:
        id_ = int(d[0])
        data_inst_list.append((id_, Instance(features=np.array(d[1:]))))

    table = session.parallelize(data_inst_list, partition=16, include_key=True)
    table.schema['header'] = header[1:]
    return table

def test_speed():

    import copy
    import time
    a = np.zeros(20)
    s = time.time()
    for i in range(100000):
        copy.deepcopy(a)
    e = time.time()
    print(e-s)

    s = time.time()
    for i in range(100000):
        b = np.array(a)
    e = time.time()
    print(e-s)


if __name__ == '__main__':
    test_speed()
    a = np.zeros(10)
    b = np.array(a)