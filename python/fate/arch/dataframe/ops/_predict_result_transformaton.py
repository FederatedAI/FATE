import functools
import pandas as pd
import torch


def transform_to_predict_result(ctx, predict_score, data_type="train", task_type="binary", classes=None,
                                threshold=0.5):
    """
    """
    transform_header = _predict_header_transform(task_type)
    if task_type == "regression":
        ...
    elif task_type == "binary":
        if predict_score.is_distributed:
            predict_score = predict_score.storage.blocks.mapValues(lambda t: t.to_local().data)
        else:
            predict_score_local = predict_score.storage.data
            predict_score = ctx.computing.parallelize(
                [predict_score_local],
                include_key=False,
                partition=1
            )

        to_predict_result_func = functools.partial(_predict_score_to_binary_result,
                                                   header=transform_header,
                                                   threshold=threshold,
                                                   classes=classes,
                                                   data_type=data_type)
        predict_result = predict_score.mapValues(to_predict_result_func)

        return predict_result, transform_header

    elif task_type == "multi":
        ...


def _predict_header_transform(task_type):
    if task_type in ["regression", "binary", "multi"]:
        return ["predict_result",
                "predict_score",
                "predict_detail",
                "type"]
    elif task_type == "cluster":
        ...
    else:
        ...


def _predict_score_to_binary_result(score_block, header, threshold=0.5, classes=None, data_type="train"):
    df = pd.DataFrame(score_block.tolist())
    if classes is None:
        classes = [0, 1]

    def _convert(score_series):
        score = score_series[0]
        result = 1 if score > threshold else 0
        return classes[result], score, {classes[result]: score, classes[1 - result]: 1 - score}, data_type

    df = df.apply(_convert, axis=1, result_type="expand")
    df.columns = header

    return df
