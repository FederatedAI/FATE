import json
from contextlib import contextmanager
from enum import Enum
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import typing

import torch
from fate_arch.common import Party
from fate_arch.federation.transfer_variable import IterationGC
from fate_arch.session import get_session
from typing_extensions import Literal


from ._parties import Parties, PreludeParty
from .abc.tensor import PHEDecryptorABC, PHEEncryptorABC, PHETensorABC
from ._federation import FederationDeserializer


class NamespaceState:
    def __init__(self, namespace) -> None:
        self._namespace = namespace

    def get_namespce(self) -> str:
        return self._namespace

    def sub_namespace(self, namespace):
        return f"{self._namespace}.{namespace}"


class FitState(NamespaceState):
    ...


class PredictState(NamespaceState):
    ...


class IterationState(NamespaceState):
    ...


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


class Future:
    """
    `get` maybe async in future, in this version,
    we wrap obj to support explicit typing and check
    """

    def __init__(self, inside) -> None:
        self._inside = inside

    def unwrap_tensor(self) -> "FPTensor":

        assert isinstance(self._inside, FPTensor)
        return self._inside

    def unwrap_phe_encryptor(self) -> "PHEEncryptor":
        assert isinstance(self._inside, PHEEncryptor)
        return self._inside

    def unwrap_phe_tensor(self) -> "PHETensor":

        assert isinstance(self._inside, PHETensor)
        return self._inside

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> T:
        if check is not None and not check(self._inside):
            raise TypeError(f"`{self._inside}` check failed")
        return self._inside


class Futures:
    def __init__(self, insides) -> None:
        self._insides = insides

    def unwrap_tensors(self) -> List["FPTensor"]:

        for t in self._insides:
            assert isinstance(t, FPTensor)
        return self._insides

    def unwrap_phe_tensors(self) -> List["PHETensor"]:

        for t in self._insides:
            assert isinstance(t, PHETensor)
        return self._insides

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> List[T]:
        if check is not None:
            for i, t in enumerate(self._insides):
                if not check(t):
                    raise TypeError(f"{i}th element `{self._insides}` check failed")
        return self._insides


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

    @property
    def device(self):
        return self._device

    @property
    def is_guest(self):
        return self._roles["local"]["role"] == "guest"

    @property
    def is_host(self):
        return self._roles["local"]["role"] == "host"

    @property
    def is_arbiter(self):
        return self._roles["local"]["role"] == "arbiter"

    @property
    def party(self):
        role = self._roles["local"]["role"]
        party_id = self._roles["local"]["party_id"]
        return Party(role, party_id)

    def get_or_set_push_gc(self, key):
        if key not in self._push_gc_dict:
            self._push_gc_dict[key] = IterationGC()
        return self._push_gc_dict[key]

    def get_or_set_pull_gc(self, key):
        if key not in self._push_gc_dict:
            self._pull_gc_dict[key] = IterationGC()
        return self._pull_gc_dict[key]

    def describe(self):
        return dict(
            party=self.party,
            job_parameters=self._job_parameters,
            parameters=self._parameters,
            flow_feeded_parameters=self._flow_feeded_parameters,
        )


class Context:
    def __init__(self, inside: _ContextInside, namespace: str) -> None:
        self._inside = inside
        self._namespace_state = NamespaceState(namespace)

    @classmethod
    def from_cpn_input(cls, cpn_input):
        states = _ContextInside(cpn_input)
        namespace = "fate"
        return Context(states, namespace)

    def describe(self):
        return json.dumps(dict(states=self._inside.describe(),))

    @property
    def party(self):
        return self._inside.party

    @property
    def role(self):
        return self.party.role

    @property
    def party_id(self):
        return self.party.party_id

    @property
    def is_guest(self):
        return self._inside.is_guest

    @property
    def is_host(self):
        return self._inside.is_guest

    @property
    def is_arbiter(self):
        return self._inside.is_guest

    @property
    def device(self) -> Device:
        return self._inside.device

    @property
    def distributed(self) -> Distributed:
        return self._inside._distributed

    def current_namespace(self):
        return self._namespace_state.get_namespce()

    def push(self, target: Parties, key: str, value):
        return self._push(target.get_parties(), key, value)

    def pull(self, source: Parties, key: str,) -> Future:
        return Future(self._pull(source.get_parties(), key)[0])

    def pulls(self, source: Parties, key: str) -> Futures:
        return Futures(self._pull(source.get_parties(), key))

    def _push(self, parties: typing.List[Party], key, value):
        if hasattr(value, "__federation_hook__"):
            value.__federation_hook__(self, key, parties)
        else:
            get_session().federation.remote(
                v=value,
                name=key,
                tag=self.current_namespace(),
                parties=parties,
                gc=self._inside.get_or_set_push_gc(key),
            )

    def _pull(self, parties: typing.List[Party], key):
        raw_values = get_session().federation.get(
            name=key,
            tag=self.current_namespace(),
            parties=parties,
            gc=self._inside.get_or_set_pull_gc(key),
        )
        values = []
        for party, raw_value in zip(parties, raw_values):
            if isinstance(raw_value, FederationDeserializer):
                values.append(raw_value.do_deserialize(self, party))
            else:
                values.append(raw_value)
        return values

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
            from fate_arch.session import computing_session
            from fate_arch.tensor.impl.tensor.distributed import FPTensorDistributed

            parts = []
            first_dim_approx = shape[0] // num_partition
            last_part_first_dim = shape[0] - (num_partition - 1) * first_dim_approx
            assert first_dim_approx > 0
            for i in range(num_partition):
                if i == num_partition - 1:
                    parts.append(torch.rand((last_part_first_dim, *shape[1:],)))
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


class PHEEncryptor:
    def __init__(self, encryptor: PHEEncryptorABC) -> None:
        self._encryptor = encryptor

    def encrypt(self, tensor: "FPTensor"):

        return PHETensor(tensor._ctx, self._encryptor.encrypt(tensor._tensor))


class PHEDecryptor:
    def __init__(self, decryptor: PHEDecryptorABC) -> None:
        self._decryptor = decryptor

    def decrypt(self, tensor: "PHETensor") -> "FPTensor":

        return FPTensor(tensor._ctx, self._decryptor.decrypt(tensor._tensor))


class FPTensor:
    def __init__(self, ctx: Context, tensor) -> None:
        self._ctx = ctx
        self._tensor = tensor

    @property
    def shape(self):
        return self._tensor.shape

    def __add__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__add__"):
            return NotImplemented
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__radd__"):
            return self.__add__(other)
        return self._binary_op(other, self._tensor.__add__)

    def __sub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__sub__"):
            return NotImplemented
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__rsub__"):
            return self.__mul__(-1).__add__(other)
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__mul__"):
            return NotImplemented
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__rmul__"):
            return self.__mul__(other)
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: "FPTensor") -> "FPTensor":
        if not hasattr(self._tensor, "__matmul__"):
            return NotImplemented
        if isinstance(other, FPTensor):
            return FPTensor(self._ctx, self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: "FPTensor") -> "FPTensor":
        if not hasattr(self._tensor, "__rmatmul__"):
            return NotImplemented
        if isinstance(other, FPTensor):
            return FPTensor(self._ctx, self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def _binary_op(self, other, func):
        if isinstance(other, FPTensor):
            return FPTensor(self._ctx, func(other._tensor))
        elif isinstance(other, (int, float)):
            return FPTensor(self._ctx, func(other))
        else:
            return NotImplemented

    @property
    def T(self):
        return FPTensor(self._ctx, self._tensor.T)

    def __federation_hook__(self, ctx, key, parties):
        deserializer = FPTensorFederationDeserializer(key)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._tensor)


class PHETensor:
    def __init__(self, ctx: Context, tensor: PHETensorABC) -> None:
        self._tensor = tensor
        self._ctx = ctx

    @property
    def shape(self):
        return self._tensor.shape

    def __add__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__radd__)

    def __sub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._ctx, self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._ctx, self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def T(self) -> "PHETensor":
        return PHETensor(self._ctx, self._tensor.T())

    @overload
    def decrypt(self, decryptor: "PHEDecryptor") -> FPTensor:
        ...

    @overload
    def decrypt(self, decryptor) -> Any:
        ...

    def decrypt(self, decryptor):
        return decryptor.decrypt(self)

    def _binary_op(self, other, func):
        if isinstance(other, (PHETensor, FPTensor)):
            return PHETensor(self._ctx, func(other._tensor))
        elif isinstance(other, (int, float)):
            return PHETensor(self._ctx, func(other))
        return NotImplemented

    def __federation_hook__(self, ctx, key, parties):
        deserializer = PHETensorFederationDeserializer(key)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._tensor)


class PHETensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key) -> None:
        self.table_key = self.make_frac_key(key, "tensor")

    def do_deserialize(self, ctx: Context, party: Party) -> PHETensor:
        tensor = ctx._pull([party], self.table_key)[0]
        return PHETensor(ctx, tensor)


class FPTensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key) -> None:
        self.table_key = self.make_frac_key(key, "tensor")

    def do_deserialize(self, ctx: Context, party: Party) -> FPTensor:
        tensor = ctx._pull([party], self.table_key)[0]
        return FPTensor(ctx, tensor)
