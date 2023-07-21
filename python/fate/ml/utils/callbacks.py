from fate.ml.evaluation.tool import get_metric_names, get_specified_metrics
from fate.ml.abc.module import Module


class CallbackParam(object):

    def __init__(self, 
                 callback_types: list, 
                 metrics: list, 
                 evaluation_freq: int = None, 
                 early_stopping_rounds: int = None,
                 checkpoint_freq: int = None,
                 use_first_metric: bool = False) -> None:

        if not isinstance(callback_types, list) or len(callback_types) == 0:
            raise ValueError("callback_types must be a list with at least one type.")

        if not isinstance(metrics, list) or len(metrics) == 0:
            raise ValueError("metrics must be a list with at least one metric.")
            
        for param, param_name in [(evaluation_freq, "evaluation_freq"), 
                                  (early_stopping_rounds, "early_stopping_rounds"), 
                                  (checkpoint_freq, "checkpoint_freq")]:
            if param is not None and (not isinstance(param, int) or param <= 0):
                raise ValueError(f"{param_name} must be a positive integer or None.")
            
        if not isinstance(use_first_metric, bool):
            raise ValueError("use_first_metric must be a boolean.")

        self.callback_types = callback_types
        self.metrics = metrics
        self.evaluation_freq = evaluation_freq
        self.early_stopping_rounds = early_stopping_rounds
        self.checkpoint_freq = checkpoint_freq
        self.use_first_metric = use_first_metric

    def __str__(self) -> str:
        return (f'Callback types: {self.callback_types}, '
                f'Metrics: {self.metrics}, '
                f'Evaluation frequency: {self.evaluation_freq}, '
                f'Early stopping rounds: {self.early_stopping_rounds}, '
                f'Use first metric for early stopping: {self.use_first_metric}, '
                f'Checkpoint frequency: {self.checkpoint_freq}')




class Callbacks(object):

    def __init__(self, model: Module, callback_params) -> None:
        pass

    def on_train_begin(self, ctx):
        pass

    def on_train_end(self, ctx):
        pass

    def on_epoch_begin(self, ctx, epoch):
        pass

    def on_epoch_end(self, ctx, epoch):
        pass

    def on_batch_begin(self, ctx, batch_index):
        pass

    def on_batch_end(self, ctx, batch_index):
        pass

    def need_stop(self, ctx):
        pass

    def get_best_model(self):
        pass