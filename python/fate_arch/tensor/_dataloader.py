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

    @property
    def shape(self) -> typing.Tuple[int, int]:
        ...

    def next_batch(self) -> typing.Tuple[FPTensor, FPTensor]:
        ...

    def has_next(self) -> bool:
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

    @property
    def shape(self) -> typing.Tuple[int, int]:
        ...

    def next_batch(self) -> FPTensor:
        ...

    def has_next(self) -> bool:
        ...
