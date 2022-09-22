from typing import List

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from .procedure import Procedure


class SkipRun(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return not params.is_need_run

    def run(
        self,
        ctx: Context,
        cpn: Module,
        params: Params,
        datasets: Datasets,
        models_loader: ModelsLoader,
        models_saver: ModelsSaver,
    ) -> List[Dataframe]:
        if isinstance(datasets.data, dict) and len(datasets.data) >= 1:
            return list(datasets.data.values())[0]
        return []
