from typing import List

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from ..utils import set_predict_data_schema, union_data
from .procedure import Procedure


class TrainWithoutValidate(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return datasets.has_train_data and (not datasets.has_validate_data)

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
        with ctx.sub_ctx("fit") as subctx:
            cpn.fit(subctx, datasets.train_data)
        with ctx.sub_ctx("predict") as subctx:
            predict_on_train_data = cpn.predict(subctx, datasets.validate_data)
        union_output = union_data([predict_on_train_data], ["train"])
        return set_predict_data_schema(union_output, datasets.schema)
