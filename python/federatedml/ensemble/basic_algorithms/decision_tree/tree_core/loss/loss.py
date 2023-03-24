from federatedml.statistic.data_overview import with_weight

class Loss(object):

    @staticmethod
    def initialize(y):
        raise NotImplementedError()

    @staticmethod
    def predict(value):
        raise NotImplementedError()

    @staticmethod
    def compute_loss(y, y_pred, sample_weights=None):
        raise NotImplementedError()

    @staticmethod
    def compute_grad(y, y_pred):
        raise NotImplementedError()

    @staticmethod
    def compute_hess(y, y_pred):
        raise NotImplementedError()
    
    @staticmethod
    def reduce(sample_loss, sample_weights=None):
        
        from federatedml.util import LOGGER
        if sample_weights is not None and with_weight(sample_weights):
            # apply sample weights
            sample_loss = sample_loss.join(sample_weights, lambda x1, x2: (x1[0] * x2.weight, x1[1] * x2.weight))
        loss_sum, sample_num = sample_loss.reduce(lambda tuple1, tuple2: (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1]))
        return loss_sum / sample_num