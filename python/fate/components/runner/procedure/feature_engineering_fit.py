from typing import List

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from .procedure import Procedure


class FeatureEngineeringFit(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return datasets.has_data and models_loader.has_model

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
        if models_loader.has_model or models_loader.has_isometric_model:
            cpn.load_model(ctx, models_loader)
        data = cpn.extract_data(datasets.data)
        with ctx.sub_ctx("fit") as subctx:
            return cpn.fit(subctx, data)
