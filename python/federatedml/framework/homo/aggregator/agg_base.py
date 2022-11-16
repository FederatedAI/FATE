from federatedml.util import consts
from federatedml.util import LOGGER
from pathlib import Path
from federatedml.model_base import Metric
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables

"""
Aggregator registration
"""

_AGGREGATOR_PAIR_DICT = {}
_AGGREGATOR_CLASS_MAP = {}
_ml_base = Path(__file__).resolve().parent.parent.parent


class AggregatorPair(object):

    def __init__(self):
        self._client_cls = None
        self._server_cls = None

    def set_client_class(self, client):
        if self._client_cls is not None:
            raise ValueError('client class is already set, cur client cls is {}, new cls is {}'.
                             format(self._client_cls, client))
        self._client_cls = client

    def set_server_class(self, server):
        if self._server_cls is not None:
            raise ValueError('server class is already set, cur server cls is {}, new cls is {}'.
                             format(self._server_cls, server))
        self._server_cls = server

    def get_server_class(self):
        return self._server_cls

    def get_client_class_name(self):
        return self._client_cls.__name__

    def __repr__(self):
        return "<client:{}, server:{}>".format(self._client_cls, self._server_cls)


# add client class to global class dict
def _add_client(cls, aggregator_name):

    if aggregator_name not in _AGGREGATOR_PAIR_DICT:
        _AGGREGATOR_PAIR_DICT[aggregator_name] = AggregatorPair()
    _AGGREGATOR_PAIR_DICT[aggregator_name].set_client_class(cls)


# add server class to global class dict
def _add_server(cls, aggregator_name):

    if aggregator_name not in _AGGREGATOR_PAIR_DICT:
        _AGGREGATOR_PAIR_DICT[aggregator_name] = AggregatorPair()
    _AGGREGATOR_PAIR_DICT[aggregator_name].set_server_class(cls)


# A decorator that registers new aggregator client
def aggregator_client(aggregator_name):

    def aggregator_decorator(cls):
        _add_client(cls, aggregator_name)
        return cls

    return aggregator_decorator


# A decorator that registers new aggregator server
def aggregator_server(aggregator_name):

    def aggregator_decorator(cls):
        _add_server(cls, aggregator_name)
        return cls

    return aggregator_decorator


def get_aggregator_pairs():
    return _AGGREGATOR_PAIR_DICT


def get_aggregator_class_map():

    for k, v in _AGGREGATOR_PAIR_DICT.items():
        _AGGREGATOR_CLASS_MAP[v.get_client_class_name()] = v.get_server_class()

    return _AGGREGATOR_CLASS_MAP


"""
Base Aggregator
"""


class AggBaseTransVar(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.client_to_server = self._create_variable(
            "client_to_server", src=["host"], dst=["arbiter"]
        )
        self.server_to_client = self._create_variable(
            "server_to_client", dst=["host"], src=["arbiter"]
        )
        self.agg_round = self._create_variable("agg_round", src=["host"], dst=["arbiter"])


class AggregatorClassConfirmTransVar(BaseTransferVariables):

    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.client_agg_class = self._create_variable(
            "client_agg_class", src=["host"], dst=["arbiter"]
        )


def arbiter_get_client_agg_class():
    trans_var = AggregatorClassConfirmTransVar()
    agg_class_str_list = trans_var.client_agg_class.get(idx=-1)
    assert len(set(agg_class_str_list)) == 1, 'clients need to use same aggregator, but got aggregator list {}'\
        .format(agg_class_str_list)
    server_class = get_aggregator_class_map()[agg_class_str_list[0]]
    LOGGER.debug('get server aggregator class {}'.format(server_class))
    return server_class


def reformat_suffix(suffix):

    if isinstance(suffix, list):
        return list(suffix)

    if not isinstance(suffix, tuple):
        return tuple(suffix)


@aggregator_client('base')
class AggregatorBaseClient(object):

    inform_server_aggregator_type = True

    @classmethod
    def disable_inform_server_aggregator_type(cls):
        cls.inform_server_aggregator_type = False

    def __init__(self, max_aggregate_round: int):
        """
        aggregate_round: int > 0, to inform the aggregator the rounds of aggregation
        inform_server_aggregator_type: bool, inform arbiter the aggregator type by sending aggregator type to server
                             IF YOU ARE DEVELOPING CUSTOM-HOMO-NN, THIS OPTION MUST BE True
        """
        self._transfer_var = AggBaseTransVar()
        self.max_aggregate_round = max_aggregate_round
        self._agg_class_confirm_transvar = AggregatorClassConfirmTransVar()

        if self.inform_server_aggregator_type:
            # send aggregator type to inform arbiter to get the corresponding server class
            self._agg_class_confirm_transvar.client_agg_class.remote(type(self).__name__)
            LOGGER.debug('sync done, send class name to server')

        assert isinstance(max_aggregate_round, int), 'round num need to be an integer'
        if self.max_aggregate_round <= 0:
            raise ValueError('round number is an int >=0, but got {}'.format(max_aggregate_round))
        self.max_aggregate_round = max_aggregate_round
        self._transfer_var.agg_round.remote(self.max_aggregate_round, suffix=('agg_round', ))

    def send_to_server(self, obj, suffix):
        """
        Sending an obj to server side, everytime calling this function, suffix must be unique.
        """
        self._transfer_var.client_to_server.remote(obj, suffix=suffix)

    def get_from_server(self, suffix):
        """
        Getting an obj from sever side
        """
        return self._transfer_var.server_to_client.get(idx=0, role=consts.ARBITER, suffix=suffix)

    def aggregate(self, *args, **kwargs):
        raise NotImplementedError('This function need to be implemented')


@aggregator_server('base')
class AggregatorBaseServer(object):

    def __init__(self):
        self._transfer_var = AggBaseTransVar()
        self.max_agg_round = self._transfer_var.agg_round.get(idx=-1, suffix=('agg_round',))[0]
        self.loss_history = []
        self._tracker = None
        LOGGER.info('aggregator base init done, agg round is {}'.format(self.max_agg_round))

    def save_loss(self, loss):
        self.loss_history.append(loss)
        
    def set_tracker(self, tracker):
        self._tracker = tracker

    def get_agg_round(self):
        if self.max_agg_round is None:
            raise ValueError('communication round not initialized, need to get this variable from guest')
        return self.max_agg_round
    
    def callback_loss(self, loss: float, idx: int):
        if self._tracker is not None:
            self._tracker.log_metric_data(
                metric_name="loss",
                metric_namespace="train",
                metrics=[Metric(idx, loss)],
            )        

    def send_to_clients(self, obj, suffix, client_idx=-1):
        """
        Sending an obj to all clients,
        client_idx: int, or list of integer,
                    -1: send to all client
                    int >= 0: send to specified client,
                         in the range of client numbers( if there are 3 clients, available idx is 0,1,2 for example)
                    list: send to several specified client
        """
        self._transfer_var.server_to_client.remote(obj, suffix=suffix, idx=client_idx)

    def get_from_clients(self, suffix, client_idx=-1):
        """
        Getting objs from all clients, return list
        client_idx: int, or list of integer,
            -1: send to all client
            int >= 0: send to specified client,
                 in the range of client numbers( if there are 3 clients, available idx is 0,1,2 for example)
            list: send to several specified client
        """
        return self._transfer_var.client_to_server.get(suffix=suffix, idx=client_idx)

    def aggregate(self):
        raise NotImplementedError("This function need to be implemented")


if __name__ == '__main__':
    pass
