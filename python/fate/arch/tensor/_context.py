from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator, Tuple, TypeVar, overload

import torch
from typing_extensions import Literal


class CipherKind(Enum):
    PHE = 1
    PHE_PAILLIER = 2


class Device(Enum):
    CPU = 1
    GPU = 2
    FPGA = 3
    CPU_Intel = 4


class Distributed(Enum):
    NONE = 1
    EGGROLL = 2
    SPARK = 3


T = TypeVar("T")


class _ContextInside:
    def __init__(self, cpn_input) -> None:
        self._push_gc_dict = {}
        self._pull_gc_dict = {}

        self._flowid = None

        self._roles = cpn_input.roles
        self._job_parameters = cpn_input.job_parameters
        self._parameters = cpn_input.parameters
        self._flow_feeded_parameters = cpn_input.flow_feeded_parameters

        self._device = Device.CPU
        self._distributed = Distributed.EGGROLL


class Context:
    def __init__(self, inside: _ContextInside, namespace: str) -> None:
        self._inside = inside

    @classmethod
    def from_cpn_input(cls, cpn_input):
        states = _ContextInside(cpn_input)
        namespace = "fate"
        return Context(states, namespace)

    @property
    def distributed(self) -> Distributed:
        return self._inside._distributed

    def current_namespace(self):
        return self._namespace_state.get_namespce()

    @overload
    def keygen(
        self, kind: Literal[CipherKind.PHE], key_length: int
    ) -> Tuple["PHEEncryptor", "PHEDecryptor"]:
        ...

    @overload
    def keygen(self, kind: CipherKind, **kwargs) -> Any:
        ...

    def keygen(self, kind, key_length: int, **kwargs):
        # TODO: exploring expansion eechanisms
        if kind == CipherKind.PHE or kind == CipherKind.PHE_PAILLIER:
            if self.distributed == Distributed.NONE:
                if self.device == Device.CPU:
                    from .impl.tensor.multithread_cpu_tensor import (
                        PaillierPHECipherLocal,
                    )

                    encryptor, decryptor = PaillierPHECipherLocal().keygen(
                        key_length=key_length
                    )
                    return PHEEncryptor(encryptor), PHEDecryptor(decryptor)
            if self.distributed == Distributed.EGGROLL:
                if self.device == Device.CPU:
                    from .impl.tensor.distributed import PaillierPHECipherDistributed

                    encryptor, decryptor = PaillierPHECipherDistributed().keygen(
                        key_length=key_length
                    )
                    return PHEEncryptor(encryptor), PHEDecryptor(decryptor)

        raise NotImplementedError(
            f"keygen for kind<{kind}>-distributed<{self.distributed}>-device<{self.device}> is not implemented"
        )

    def random_tensor(self, shape, num_partition=1) -> "FPTensor":
        if self.distributed == Distributed.NONE:
            return FPTensor(self, torch.rand(shape))
        else:
            from fate.arch.tensor.impl.tensor.distributed import FPTensorDistributed

            from ..session import computing_session

            parts = []
            first_dim_approx = shape[0] // num_partition
            last_part_first_dim = shape[0] - (num_partition - 1) * first_dim_approx
            assert first_dim_approx > 0
            for i in range(num_partition):
                if i == num_partition - 1:
                    parts.append(
                        torch.rand(
                            (
                                last_part_first_dim,
                                *shape[1:],
                            )
                        )
                    )
                else:
                    parts.append(torch.rand((first_dim_approx, *shape[1:])))
            return FPTensor(
                self,
                FPTensorDistributed(
                    computing_session.parallelize(
                        parts, include_key=False, partition=num_partition
                    )
                ),
            )

    def create_tensor(self, tensor: torch.Tensor) -> "FPTensor":

        return FPTensor(self, tensor)

    @contextmanager
    def sub_namespace(self, namespace):
        """
        into sub_namespace ``, suffix federation namespace with `namespace`

        Examples:
        ```
        with ctx.sub_namespace("fit"):
            ctx.push(..., trans_key, obj)

        with ctx.sub_namespace("predict"):
            ctx.push(..., trans_key, obj2)
        ```
        `obj1` and `obj2` are pushed with different namespace
        without conflic.
        """

        prev_namespace_state = self._namespace_state

        # into subnamespace
        self._namespace_state = NamespaceState(
            self._namespace_state.sub_namespace(namespace)
        )

        # return sub_ctx
        # ```python
        # with ctx.sub_namespace(xxx) as sub_ctx:
        #     ...
        # ```
        #
        yield self

        # restore namespace state when leaving with context
        self._namespace_state = prev_namespace_state

    @overload
    @contextmanager
    def iter_namespaces(
        self, start: int, stop: int, *, prefix_name=""
    ) -> Generator[Generator["Context", None, None], None, None]:
        ...

    @overload
    @contextmanager
    def iter_namespaces(
        self, stop: int, *, prefix_name=""
    ) -> Generator[Generator["Context", None, None], None, None]:
        ...

    @contextmanager
    def iter_namespaces(self, *args, prefix_name=""):
        assert 0 < len(args) <= 2, "position argument should be 1 or 2"
        if len(args) == 1:
            start, stop = 0, args[0]
        if len(args) == 2:
            start, stop = args[0], args[1]

        prev_namespace_state = self._namespace_state

        def _state_iterator() -> Generator[Context, None, None]:
            for i in range(start, stop):
                # the tags in the iteration need to be distinguishable
                template_formated = f"{prefix_name}iter_{i}"
                self._namespace_state = IterationState(
                    prev_namespace_state.sub_namespace(template_formated)
                )
                yield self

        # with context returns iterator of Contexts
        # namespaec state inside context is changed alone with iterator comsued
        yield _state_iterator()

        # restore namespace state when leaving with context
        self._namespace_state = prev_namespace_state
