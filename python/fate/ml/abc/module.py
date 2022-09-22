from abc import abstractmethod
from typing import Optional

from federatedml.util import consts

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver
from fate.interface import Module as ModuleProtocol

from .param import MLParam


class Module(ModuleProtocol):
    mode: str

    @abstractmethod
    def __init__(self, params: MLParam) -> None:
        ...

    @abstractmethod
    def fit(
        self,
        ctx: Context,
        train_data: Dataframe,
        validate_data: Optional[Dataframe] = None,
    ) -> None:
        ...

    @abstractmethod
    def transform(self, ctx: Context, transform_data: Dataframe) -> Dataframe:
        ...

    @abstractmethod
    def predict(self, ctx: Context, predict_data: Dataframe) -> Dataframe:
        ...

    @abstractmethod
    @classmethod
    def load_model(cls, ctx: Context, loader: ModelsLoader) -> "Module":
        ...

    @abstractmethod
    def save_model(self, ctx: Context, saver: ModelsSaver) -> None:
        ...


class HomoModule(Module):
    mode = consts.HOMO


class HeteroModule(Module):
    mode = consts.HETERO
