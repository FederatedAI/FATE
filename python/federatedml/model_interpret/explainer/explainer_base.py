import numpy as np
from federatedml.util import consts


def data_inst_table_to_arr(data_inst, take_num=500):

    take_rs = data_inst.take(take_num)
    header = data_inst.schema['header']
    ids = []
    data_list = []
    for id_, inst in take_rs:
        ids.append(id_)
        data_list.append(inst.features)
    data_arr = np.array(data_list)
    return ids, header, data_arr


class Explainer(object):

    def __init__(self, role, flow_id):
        assert role in [consts.GUEST, consts.HOST, consts.ARBITER]
        self.role = role
        self.flow_id = flow_id

    def init_model(self, *args):
        pass

    def init_background_data(self, *args):
        pass

    def explain(self, data_inst, n=500):
        pass

    def explain_row(self, *args):
        pass

    def explain_interaction(self, data_inst, n=500):
        pass

