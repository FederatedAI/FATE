from federatedml.framework.homo.blocks import ServerCommunicator, ClientCommunicator
from federatedml.util import consts


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


class AggregatorBaseClient(object):

    def __init__(self, communicate_match_suffix: str = None, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST)):
        """Base class of client aggregator

        Parameters
        ----------
        communicate_match_suffix : str, you can give a unique name to aggregator, to avoid reusing of same transfer variable tag,
                          To make sure that client and server can communicate correctly,
                          the server-side and client-side aggregators need to have the same suffix
        """
        self.communicator = ClientCommunicator(prefix=communicate_match_suffix, server=server, clients=clients)
        self.suffix = {}

    def _get_suffix(self, var_name, user_suffix=tuple()):

        assert var_name in self.suffix, 'var name {} not found in suffix list'.format(
            var_name)
        if user_suffix is not None and not isinstance(user_suffix, tuple):
            raise ValueError('suffix must be None, tuples contains str or number. got {} whose type is {}'.format(
                user_suffix, type(user_suffix)))
        if user_suffix is None or len(user_suffix) == 0:
            return self.suffix[var_name]()
        else:
            return (var_name, ) + user_suffix

    def send(self, obj, suffix):
        self.communicator.send_obj(obj, suffix=suffix)

    def get(self, suffix):
        return self.communicator.get_obj(suffix=suffix)


class AggregatorBaseServer(object):

    def __init__(self, communicate_match_suffix=None, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST)):
        """Base class of server aggregator

        Parameters
        ----------
        communicate_match_suffix : str, you can give a unique name to aggregator, to avoid reusing of same transfer variable tag,
                          To make sure that client and server can communicate correctly,
                          the server-side and client-side aggregators need to have the same suffix
        """
        self.communicator = ServerCommunicator(prefix=communicate_match_suffix, server=server, clients=clients)
        self.suffix = {}

    def _get_suffix(self, var_name, user_suffix=tuple()):

        assert var_name in self.suffix, 'var name {} not found in suffix list'.format(
            var_name)
        if user_suffix is not None and not isinstance(user_suffix, tuple):
            raise ValueError('suffix must be None, tuples contains str or number. got {} whose type is {}'.format(
                user_suffix, type(user_suffix)))
        if user_suffix is None or len(user_suffix) == 0:
            return self.suffix[var_name]()
        else:
            return (var_name, ) + user_suffix

    def broadcast(self, obj, suffix, party_idx=-1):
        self.communicator.broadcast_obj(obj, suffix=suffix, party_idx=party_idx)

    def collect(self, suffix, party_idx=-1):
        objs = self.communicator.get_obj(suffix=suffix, party_idx=party_idx)
        return objs
