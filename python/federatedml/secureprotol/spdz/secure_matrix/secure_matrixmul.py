import numpy as np

from fate_arch.common import Party
from fate_arch.session import is_table
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.secureprotol.spdz.tensor.fixedpoint_endec import FixedPointEndec
from federatedml.transfer_variable.transfer_class.sshe_model_transfer_variable import SSHEModelTransferVariable


class SecureMatrixMul(object):
    def __init__(self):
        self.transfer_variable = SSHEModelTransferVariable()
        self.role = None

    @classmethod
    def table_dot(cls, a_table, b_table):
        def _table_dot_func(it):
            ret = None
            for _, (x, y) in it:
                if ret is None:
                    ret = np.tensordot(x, y, [[], []])
                else:
                    ret += np.tensordot(x, y, [[], []])
            return ret

        return a_table.join(b_table, lambda x, y: [x, y]) \
            .applyPartitions(lambda it: _table_dot_func(it)) \
            .reduce(lambda x, y: x + y)

    @classmethod
    def dot(cls, matrix, y):
        def _vec_dot(x, y):
            res = np.dot(x, y)
            if not isinstance(res, np.ndarray):
                res = np.array([res])
            return res

        if isinstance(y, np.ndarray):
            res = matrix.mapValues(_vec_dot)
            return res

        elif is_table(y):
            res = cls.table_dot(matrix, y)
            return res
        else:
            raise ValueError(f"type={type(y)}")

    def secure_matrix_mul(self, matrix, cipher=None, suffix=tuple()):
        curt_suffix = ("secure_matrix_mul",) + suffix
        dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST

        if cipher is not None:
            if isinstance(matrix, fixedpoint_table.FixedPointTensor):
                de_matrix = self.fixpoint_encoder.decode(matrix.value)
                encrypt_mat = cipher.distribute_encrypt(de_matrix)
            else:
                encrypt_mat = cipher.recursive_encrypt(matrix.value)

            # remote encrypted matrix;
            self.transfer_variable.share_matrix.remote(encrypt_mat, role=dest_role, idx=0, suffix=curt_suffix)

            share_tensor = ShareMarixinSecretSharingWithHe.from_source(tensor_name, source, cipher, q_field, encoder)

            return share_tensor

        else:
            share = self.transfer_variable.share_matrix.get(role=dest_role, idx=0,
                                                            suffix=curt_suffix)
            LOGGER.debug(f"Make share tensor")
            # share dtype : PaillerTensor;
            res = cls.dot(matrix, share)
            share_tensor = ShareMarixinSecretSharingWithHe.from_source(tensor_name, res, cipher, q_field, encoder)

            return share_tensor


class ShareMarixinSecretSharingWithHe(object):
    def __init__(self):
        pass

    @classmethod
    def from_source(cls, tensor_name, source, cipher, q_field, encoder, fixedpoint_numpy=True):
        if is_table(source):
            random_tensor = fixedpoint_table.PaillierFixedPointTensor.from_source(tensor_name, source,
                                                          encoder=encoder,
                                                          q_field=q_field)
            return random_tensor

        elif isinstance(source,  np.ndarray):
            random_tensor = fixedpoint_numpy.PaillierFixedPointTensor.from_source(tensor_name, source,
                                                                          encoder=encoder,
                                                                          q_field=q_field)
            return random_tensor

        elif isinstance(source, Party):
            if fixedpoint_numpy:
                share_tensor = fixedpoint_numpy.PaillierFixedPointTensor.from_source(tensor_name, source,
                                                                         encoder=encoder,
                                                                         q_field=q_field)
            else:
                share_tensor = fixedpoint_table.PaillierFixedPointTensor.from_source(tensor_name, source,
                                                                             encoder=encoder,
                                                                             q_field=q_field)

            if isinstance(share_tensor.value, numpy.ndarray):
                share = cipher.recursive_decrypt(share_tensor.value)
                share = encoder.encode(share)
                return fixedpoint_numpy.FixedPointTensor(value=share,
                                                         q_field=q_field,
                                                         endec=encoder)
            elif is_table(share_tensor.value):
                share = cipher.distribute_decrypt(share_tensor.value)
                share = encoder.encode(share)
                return fixedpoint_table.FixedPointTensor(share, q_field=q_field, encoder=encoder)

            else:
                raise ValueError(f"type={type(share_tensor.value)}")

        else:
            raise ValueError(f"type={type(source)}")
