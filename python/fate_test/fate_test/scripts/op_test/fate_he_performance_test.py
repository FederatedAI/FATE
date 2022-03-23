import numpy as np
from prettytable import PrettyTable, ORGMODE
from fate_test.scripts.op_test.performance_assess import Metric
from operator import add, mul


class PaillierAssess(object):
    def __init__(self, method, data_num, test_round):
        from federatedml.secureprotol.fate_paillier import PaillierKeypair
        self.public_key, self.private_key = PaillierKeypair.generate_keypair()
        self.method = method
        self.data_num = data_num
        self.test_round = test_round
        self.float_data_x, self.encrypt_float_data_x, self.int_data_x, self.encrypt_int_data_x = self._get_data()
        self.float_data_y, self.encrypt_float_data_y, self.int_data_y, self.encrypt_int_data_y = self._get_data()

    def _get_data(self, type_int=True, type_float=True):
        if self.method == "Paillier":
            key = self.public_key
        else:
            key = None
        encrypt_float_data = []
        encrypt_int_data = []
        float_data = np.random.uniform(-1e9, 1e9, size=self.data_num)
        int_data = np.random.randint(-1000, 1000, size=self.data_num)
        if type_float:
            for i in float_data:
                encrypt_float_data.append(key.encrypt(i))
        if type_int:
            for i in int_data:
                encrypt_int_data.append(key.encrypt(i))
        return float_data, encrypt_float_data, int_data, encrypt_int_data

    def output_table(self):
        table = PrettyTable()
        table.set_style(ORGMODE)
        table.field_names = [self.method, "One time consumption", f"{self.data_num} times consumption",
                             "relative acc", "log2 acc", "operations per second", "plaintext consumption per second"]

        metric = Metric(data_num=self.data_num, test_round=self.test_round)

        table.add_row(metric.encrypt(self.float_data_x, self.public_key.encrypt))
        decrypt_data = [self.private_key.decrypt(i) for i in self.encrypt_float_data_x]
        table.add_row(metric.decrypt(self.encrypt_float_data_x, self.float_data_x, decrypt_data,
                                     self.private_key.decrypt))

        real_data = list(map(add, self.float_data_x, self.float_data_y))
        encrypt_data = list(map(add, self.encrypt_float_data_x, self.encrypt_float_data_y))
        self.binary_op(table, metric, self.encrypt_float_data_x, self.encrypt_float_data_y,
                       self.int_data_x, self.int_data_y, real_data, encrypt_data,
                       add, "float add")

        real_data = list(map(add, self.int_data_x, self.int_data_y))
        encrypt_data = list(map(add, self.encrypt_int_data_x, self.encrypt_int_data_y))
        self.binary_op(table, metric, self.encrypt_int_data_x, self.encrypt_int_data_y,
                       self.int_data_x, self.int_data_y, real_data, encrypt_data,
                       add, "int add")

        real_data = list(map(mul, self.float_data_x, self.float_data_y))
        encrypt_data = list(map(mul, self.encrypt_float_data_x, self.float_data_y))
        self.binary_op(table, metric, self.encrypt_float_data_x, self.float_data_y,
                       self.float_data_x, self.float_data_y, real_data, encrypt_data,
                       mul, "float mul")

        real_data = list(map(mul, self.int_data_x, self.int_data_y))
        encrypt_data = list(map(mul, self.encrypt_int_data_x, self.int_data_y))
        self.binary_op(table, metric, self.encrypt_int_data_x, self.int_data_y,
                       self.float_data_x, self.float_data_y, real_data, encrypt_data,
                       mul, "int mul")

        return table.get_string(title=f"{self.method} Computational performance")

    def binary_op(self, table, metric, encrypt_data_x, encrypt_data_y, raw_data_x, raw_data_y,
                  real_data, encrypt_data, op, op_name):
        decrypt_data = [self.private_key.decrypt(i) for i in encrypt_data]
        table.add_row(metric.binary_op(encrypt_data_x, encrypt_data_y,
                                       raw_data_x, raw_data_y,
                                       real_data, decrypt_data,
                                       op, op_name))
