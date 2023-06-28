from fate.arch import Context
from typing import Optional
import logging

logger = logging.getLogger(__name__)



class AutoSuffix(object):

    """
    A auto suffix that will auto increase count
    """

    def __init__(self, suffix_str=""):
        self._count = 0
        self.suffix_str = suffix_str

    def __call__(self):
        concat_suffix = self.suffix_str + "_" + str(self._count)
        self._count += 1
        return concat_suffix


class Aggregator:

    def __init__(self, ctx: Context, aggregator_name: Optional[str] = None):
        self.ctx = ctx
        if aggregator_name is not None:
            agg_name = "_" + aggregator_name
        else:
            agg_name = ""
        self.suffix = {
            "local_loss": AutoSuffix("local_loss" + agg_name),
            "agg_loss": AutoSuffix("agg_loss" + agg_name),
            "local_model": AutoSuffix("local_model" + agg_name),
            "agg_model": AutoSuffix("agg_model" + agg_name),
            "converge_status": AutoSuffix("converge_status" + agg_name),
            "local_weight": AutoSuffix("local_weight" + agg_name),
            "computed_weight": AutoSuffix("agg_weight" + agg_name),
        }

    def model_aggregation(self, *args, **kwargs):
        raise NotImplementedError("model_aggregation should be implemented in subclass")

    def loss_aggregation(self, *args, **kwargs):
        raise NotImplementedError("loss_aggregation should be implemented in subclass")
