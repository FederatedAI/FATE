from typing import Optional

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver
from fate.interface import Module as ModuleProtocol


class Module(ModuleProtocol):
    mode: str

    def fit(
        self,
        ctx: Context,
        train_data: Dataframe,
        validate_data: Optional[Dataframe] = None,
    ) -> None:
        ...

    def transform(self, ctx: Context, transform_data: Dataframe) -> Dataframe:
        ...

    def predict(self, ctx: Context, predict_data: Dataframe) -> Dataframe:
        ...

    @classmethod
    def load_model(cls, ctx: Context, loader: ModelsLoader) -> "Module":
        ...

    def save_model(self, ctx: Context, saver: ModelsSaver) -> None:
        ...


class HomoModule(Module):
    mode = "HOMO"


class HeteroModule(Module):
    mode = "HETERO"
