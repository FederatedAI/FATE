from typing import List

from fate.interface import Context, Dataframe, ModelsLoader, ModelsSaver, Module, Params

from ..parser.data import Datasets
from .procedure import Procedure


class Stepwise(Procedure):
    @classmethod
    def is_fulfilled(
        cls, params: Params, datasets: Datasets, models_loader: ModelsLoader
    ) -> bool:
        return params.is_need_stepwise

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
        cpn.disable_callback_loss()  # TODO: cpn need aware about stepwise
        if cpn.mode == consts.HETERO:
            step_obj = HeteroStepwise()
        else:
            raise ValueError("stepwise currently only support Hetero mode.")

        def _get_stepwise_param(model):
            model.model_param.stepwise_param.role = model.role
            model.model_param.stepwise_param.mode = model.mode
            return model.model_param.stepwise_param

        stepwise_param = _get_stepwise_param(cpn)
        step_obj.run(stepwise_param, datasets.train_data, datasets.validate_data, cpn)
        pred_result = HeteroStepwise.predict(datasets.train_data, cpn)
        union_output = union_data([pred_result], ["train"])
        return set_predict_data_schema(union_output, datasets.schema)
