from pipeline.param.base_param import BaseParam


class SBTTransformerParam(BaseParam):

    def __init__(self, dense_format=True):

        """
        Args:
            dense_format: return data in dense vec, otherwise return in sparse vec
        """
        super(SBTTransformerParam, self).__init__()
        self.dense_format = dense_format

    def check(self):
        self.check_boolean(self.dense_format, 'SBTTransformer')
