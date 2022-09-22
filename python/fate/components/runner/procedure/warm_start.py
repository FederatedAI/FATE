from typing import List

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from .procedure import Procedure
from .train_with_validate import TrainWithValidate
from .train_without_validate import TrainWithoutValidate


class WarmStart(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return models_loader.has_model and datasets.has_train_data

    @classmethod
    def run(
        cls,
        ctx: Context,
        cpn: Module,
        params: Params,
        datasets: Datasets,
        models_loader: ModelsLoader,
        models_saver: ModelsSaver,
    ) -> List[Dataframe]:
        cpn.load_model(ctx, models_loader)
        if TrainWithValidate.is_fulfilled(params, datasets, models_loader):
            TrainWithValidate.run(
                ctx, cpn, params, datasets, models_loader, models_saver
            )
        elif TrainWithoutValidate.is_fulfilled(params, datasets, models_loader):
            TrainWithoutValidate.run(
                ctx, cpn, params, datasets, models_loader, models_saver
            )
        raise RuntimeError(f"invalid warmstart situation")
