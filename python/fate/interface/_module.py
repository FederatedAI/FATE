from typing import List, Optional, Protocol, Union

from ._context import Context
from ._data_io import Dataframe
from ._param import Params


class Model(Protocol):
    ...


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
    def from_model(cls, model: Union[dict, Model]) -> "Module":
        ...

    def get_model(self) -> Union[dict, Model]:
        ...
