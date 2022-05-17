import typing

from ._tensor import FPTensor


class LabeledDataloaderWrapper:
    """
    wrapper to transform data_instance to tensor-frendly Dataloader
    """

    def __init__(
        self,
        data_instance,
        max_iter,
        batch_size=-1,
        with_intercept=False,
        shuffle=False,
    ):
        ...

    def __iter__(self) -> "LabeledDataloaderWrapper":
        ...

    def __next__(self) -> typing.Tuple[FPTensor, FPTensor]:
        ...


class UnlabeledDataloaderWrapper:
    """
    wrapper to transform data_instance to tensor-frendly Dataloader
    """

    def __init__(
        self,
        data_instance,
        max_iter,
        batch_size=-1,
        with_intercept=False,
        shuffle=False,
    ):
        ...

    def __iter__(self) -> "UnlabeledDataloaderWrapper":
        ...

    def __next__(self) -> FPTensor:
        ...
