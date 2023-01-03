from fate.components import (
    GUEST,
    HOST,
    ARBITER,
    DatasetArtifact,
    Input,
    Output,
    Role,
    cpn,
    ClassificationMetrics,
)
from fate.ml.evaluation import BinaryEvaluator


@cpn.component(roles=[GUEST, HOST, ARBITER])
@cpn.artifact("input_data", type=Input[DatasetArtifact], roles=[GUEST, HOST, ARBITER])
@cpn.parameter("eval_type", type=str, default="binary", optional=True)
@cpn.artifact("output_metric", type=Output[ClassificationMetrics], roles=[GUEST, HOST, ARBITER])
def evaluation(
    ctx,
    role: Role,
    input_data,
    eval_type,
    output_metric
):
    evaluate(ctx, input_data, eval_type, output_metric)


def evaluate(ctx, input_data, eval_type, output_metric):
    data = ctx.reader(input_data).read_dataframe().data
    y_true = data.label.tolist()
    y_pred = data.predict_score.values.tolist()

    if eval_type == "binary":
        ctx.metrics.handler.register_metrics(auc=ctx.writer(output_metric))
        evaluator = BinaryEvaluator()
        evaluator.fit(ctx, y_true, y_pred)


