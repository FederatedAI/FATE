import numpy as np

from federatedml.feature.feature_selection.model_adapter import isometric_model
from federatedml.feature.feature_selection.model_adapter.adapter_base import BaseAdapter
from federatedml.util import consts


class PSIAdapter(BaseAdapter):

    def convert(self, model_meta, model_param):

        psi_scores = dict(model_param.total_score)

        col_names, values = [], []
        for name in psi_scores:
            col_names.append(name)
            values.append(psi_scores[name])

        single_info = isometric_model.SingleMetricInfo(
            values=np.array(values),
            col_names=col_names
        )
        result = isometric_model.IsometricModel()
        result.add_metric_value(metric_name=consts.PSI, metric_info=single_info)
        return result
