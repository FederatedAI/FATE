import logging
import typing

from fate.components.core import ARBITER, GUEST, HOST, Role, cpn, params

if typing.TYPE_CHECKING:
    from fate.arch import Context

logger = logging.getLogger(__name__)


@cpn.component(roles=[GUEST, HOST, ARBITER])
def cv_test(ctx, role):
    ...


@cv_test.cross_validation()
def cv(
    ctx: "Context",
    role: Role,
    num_fold: cpn.parameter(type=params.conint(gt=1), desc="parameter", optional=False),
    dataframe_input: cpn.dataframe_input(roles=[GUEST, HOST]),
    json_model_outputs: cpn.json_model_outputs(roles=[GUEST, HOST]),
    dataframe_outputs: cpn.dataframe_outputs(roles=[GUEST, HOST]),
):
    # split data
    for cv_ctx, (train_data, validata_data) in ctx.on_cross_validations.ctxs_zip(
        split_data(dataframe_input.read(), num_fold)
    ):
        # train model
        model = FakeTrainer()
        model.fit(cv_ctx, train_data=train_data)
        # predict model
        predict_result = model.predict(cv_ctx, validata_data=validata_data)
        # evaluate model
        evaluation_result = fake_evaluation(cv_ctx, predict_result=predict_result)
        next(json_model_outputs).write(data=model.get_model(), metadata=evaluation_result)
        cv_ctx.metrics.log_auc("fold_auc", evaluation_result["auc"])
        cv_ctx.metrics.log_roc("fold_roc", [0.1, 0.2, 0.3, 0.4, 0.5])
        next(dataframe_outputs).write(df=predict_result)


def split_data(dataframe_input, num_fold):
    """fake split data"""
    for i in range(num_fold):
        yield dataframe_input, dataframe_input


class FakeTrainer:
    def __init__(self, **kwargs):
        self.model = {
            "data": "fake_model",
        }

    def fit(self, ctx, train_data, **kwargs):
        for i, sub_ctx in ctx.on_iterations.ctxs_range(5):
            sub_ctx.metrics.log_auc("auc", i / 5)

    def predict(self, ctx, validata_data, **kwargs):
        return validata_data

    def get_model(self):
        return self.model


def fake_evaluation(ctx, **kwargs):
    return {"auc": 0.9}
