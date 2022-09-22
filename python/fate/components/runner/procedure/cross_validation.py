from typing import List

from federatedml.model_selection.k_fold import KFold

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from .procedure import Procedure


class CrossValidation(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return params.is_need_cv

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
        kflod_obj = KFold()
        params.cv_param.role = ctx.role
        params.cv_param.mode = cpn.mode
        output_data = kflod_obj.run(params.cv_param, datasets.train_data, cpn, False)
        return output_data
