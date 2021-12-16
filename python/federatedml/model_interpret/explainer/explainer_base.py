import numpy as np
from federatedml.util import consts
from federatedml.util import LOGGER
from fate_arch.session import computing_session as session


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


def take_inst_in_sorted_order(data_inst, take_num=500, ret_arr=True):

    generator = data_inst.collect()
    header = data_inst.schema['header']
    ids = [i[0] for i in generator]
    sorted_id = sorted(ids)
    target_id = sorted_id[0:take_num]
    join_table = session.parallelize([(id_, 1) for id_ in target_id], partition=data_inst.partitions, include_key=True)
    result = data_inst.join(join_table, lambda x1, x2: x1)
    collect_result = list(result.collect())
    collect_result = sorted(collect_result, key=lambda x: x[0])

    if ret_arr:
        ids = []
        data_list = []
        for id_, inst in collect_result:
            ids.append(id_)
            data_list.append(inst.features)
        data_arr = np.array(data_list)

        return ids, header, data_arr

    else:
        return collect_result


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

