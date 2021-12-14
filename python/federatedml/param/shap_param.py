from federatedml.param.base_param import BaseParam
from federatedml.util import consts


class SHAPParam(BaseParam):

    """
        Parameter for SHAP which offers explains for federated models.
        SHAP will automatically selects SHAP algorithms for input models:

        Homo/Hetero Tree Models use TreeSHAP
        Homo/Hetero Linear Models use KernelSHAP
        Homo/Hetero NN Models use KernelSHAP

        SHAP will explain all input instances for tree models using TreeSHAP.
        For other models, due to the slow interpretation speed of KernelSHAP, by default we will
        explain first 500 instances taken from the dataset

        Parameters
        ----------

        reference_vec_type : str, 'zeros', 'median', 'average' are accepted, default is zeros.

                             defines how to generate reference vec when running KernelSHAP.
                             values in reference vector are used to replace 'missing features' when evaluating the
                             expected prediction of a certain feature subset.

                             'zeros': reference vec are all zeros
                             'median': vector values are the median of features computed from the input dataset
                             'average': vector values are the average of features computed from the input dataset

        explain_all_host_feature : bool, default is False.

                                   If False, every host party will be regarded as one 'federated feature'.
                                   If True, Explain all host features for hetero federated models.

                                   Noted that for the concern of protecting privacy, explaining all host features is NOT
                                   SUPPORTED IN HETERO LINEAR MODEL!
        """

    def __init__(self, reference_vec_type=consts.ZEROS, explain_all_host_feature=False):
        super(SHAPParam, self).__init__()
        self.reference_type = reference_vec_type
        self.explain_all_host_feature = explain_all_host_feature

    def check(self):

        self.check_valid_value(self.reference_type, 'reference vec type', [consts.ZEROS, consts.MEDIAN, consts.AVERAGE])
        self.check_boolean(self.explain_all_host_feature, 'explain all host feature')



