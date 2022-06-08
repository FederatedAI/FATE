from federatedml.util import LOGGER
from federatedml.framework.homo.blocks import secure_sum_aggregator, loss_scatter, has_converged


class HomoBoostArbiterAggregator(object):

    def __init__(self, ):
        """
        Args:
            transfer_variable:
            converge_type: see federatedml/optim/convergence.py
            tolerate_val:
        """
        self.loss_scatter = loss_scatter.Server()
        self.has_converged = has_converged.Server()

    def aggregate_loss(self, suffix):
        global_loss = self.loss_scatter.weighted_loss_mean(suffix=suffix)
        return global_loss

    def broadcast_converge_status(self, func, loss, suffix):
        is_converged = func(*loss)
        self.has_converged.remote_converge_status(is_converged, suffix=suffix)
        LOGGER.debug('convergence status sent with suffix {}'.format(suffix))
        return is_converged


class HomoBoostClientAggregator(object):

    def __init__(self, ):
        self.loss_scatter = loss_scatter.Client()
        self.has_converged = has_converged.Client()

    def send_local_loss(self, loss, sample_num, suffix):
        self.loss_scatter.send_loss(loss=(loss, sample_num), suffix=suffix)
        LOGGER.debug('loss sent with suffix {}'.format(suffix))

    def get_converge_status(self, suffix):
        converge_status = self.has_converged.get_converge_status(suffix)
        return converge_status
