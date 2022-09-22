from typing import List

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from ..utils import set_predict_data_schema, union_data
from .procedure import Procedure


class Predict(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return (not datasets.has_train_data) and datasets.has_test_data

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
        with ctx.sub_ctx("predict") as subctx:
            predict_on_test_data = cpn.predict(subctx, datasets.test_data)
        union_output = union_data([predict_on_test_data], ["predict"])
        return set_predict_data_schema(union_output, datasets.schema)
