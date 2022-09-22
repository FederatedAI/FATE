from typing import List

from fate.interface import Module, Params

from ..context import ComponentContext
from ..parser.data import Datasets
from ..parser.model import ModelsLoader
from .cross_validation import CrossValidation, Dataframe, ModelsSaver
from .feature_engineering_fit import FeatureEngineeringFit
from .feature_engineering_transform import FeatureEngineeringTransform
from .predict import Predict
from .procedure import Procedure
from .skip_run import SkipRun
from .stepwise import Stepwise
from .train_with_validate import TrainWithValidate
from .train_without_validate import TrainWithoutValidate
from .warm_start import WarmStart


class Dispatcher:
    procedures: List[Procedure] = [
        SkipRun,
        CrossValidation,
        Stepwise,
        WarmStart,
        TrainWithValidate,
        TrainWithoutValidate,
        Predict,
        FeatureEngineeringFit,
        FeatureEngineeringTransform,
    ]

    @classmethod
    def dispatch_run(
        cls,
        ctx: ComponentContext,
        cpn: Module,
        params: Params,
        datasets: Datasets,
        model_loader: ModelsLoader,
        model_saver: ModelsSaver,
        retry: bool,
    ) -> List[Dataframe]:
        if (
            retry
            and (checkpoint := ctx.checkpoint_manager.get_latest_checkpoint())
            is not None
        ):
            return WarmStart.run(ctx, cpn, params, datasets, checkpoint, model_saver)
        else:
            for procedure_cls in cls.procedures:
                if procedure_cls.is_fulfilled(params, datasets, model_loader):
                    return procedure_cls.run(
                        ctx, cpn, params, datasets, model_loader, model_saver
                    )
            raise RuntimeError(f"nothing dispatched")


__all__ = [
    "CrossValidation",
    "FeatureEngineeringTransform",
    "FeatureEngineeringFit",
    "Predict",
    "Stepwise",
    "TrainWithValidate",
    "TrainWithoutValidate",
    "WarmStart",
    "SkipRun",
    "Dispatcher",
]
