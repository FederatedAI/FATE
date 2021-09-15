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

    def _dot(self, array):
        def _dot(x):
            res = fate_operator.vec_dot(x, array)
            if not isinstance(res, np.ndarray):
                res = np.array([res])
            return res

        if isinstance(share, np.ndarray):
            xy = self._dot(matrix, share)
        else:
            share_tensor = fixedpoint_table.PaillierFixedPointTensor(
                share, q_field=matrix.q_field, encoder=matrix.endec)
            xy = matrix.dot_local(share_tensor)
        LOGGER.debug(f"Finish dot")

        return self._boxed(self.value.mapValues(_dot))

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
            xy = self._dot(matrix, share)
            share_tensor = ShareMarixinSecretSharingWithHe.from_source(tensor_name, xy, cipher, q_field, encoder)

            return share_tensor


class ShareMarixinSecretSharingWithHe(object):
    def __init__(self):
        pass

    @classmethod
    def from_source(cls, tensor_name, source, cipher, q_field, encoder, fixedpoint_numpy=True):
        if isinstance(source, fixedpoint_table.FixedPointTensor):
            random_tensor = fixedpoint_table.FixedPointTensor.from_source(tensor_name, source,
                                                          encoder=encoder,
                                                          q_field=q_field)
            return random_tensor

        elif isinstance(source,  fixedpoint_numpy.FixedPointTensor):
            random_tensor = fixedpoint_numpy.FixedPointTensor.from_source(tensor_name, source,
                                                                          encoder=encoder,
                                                                          q_field=q_field)
            return random_tensor

        elif isinstance(source, Party):
            if fixedpoint_numpy:
                share_tensor = fixedpoint_numpy.FixedPointTensor.from_source(tensor_name, source,
                                                                         encoder=encoder,
                                                                         q_field=q_field)
            else:
                share_tensor = fixedpoint_table.FixedPointTensor.from_source(tensor_name, source,
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
