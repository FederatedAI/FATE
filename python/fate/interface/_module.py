from typing import List, Optional, Protocol

from ._context import Context
from ._data_io import Dataframe
from ._model_io import ModelsLoader, ModelsSaver
from ._param import Params


class Module(Protocol):
    mode: str

    def __init__(self, params: Params) -> None:
        ...

    def fit(
        self,
        ctx: Context,
        train_data: Dataframe,
        validate_data: Optional[Dataframe] = None,
    ) -> None:
        ...

    def transform(self, ctx: Context, transform_data: Dataframe) -> List[Dataframe]:
        ...

    def predict(self, ctx: Context, predict_data: Dataframe) -> Dataframe:
        ...

    @classmethod
    def load_model(cls, ctx: Context, loader: ModelsLoader) -> "Module":
        ...

    def save_model(self, ctx: Context, saver: ModelsSaver) -> None:
        ...
