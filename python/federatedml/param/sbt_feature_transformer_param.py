from federatedml.param.base_param import BaseParam


class SBTTransformerParam(BaseParam):

    def __init__(self, dense_format=True):

        """
        Parameters
        ----------
        dense_format: bool
            return data in dense vec if True, otherwise return in sparse vec
        """
        super(SBTTransformerParam, self).__init__()
        self.dense_format = dense_format

    def check(self):
        self.check_boolean(self.dense_format, 'SBTTransformer')
