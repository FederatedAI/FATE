from federatedml.framework.homo.blocks import RandomPaddingCipherClient, RandomPaddingCipherServer, PadsCipher, RandomPaddingCipherTransVar
from federatedml.framework.homo.aggregator.aggregator_base import AggregatorBaseClient, AutoSuffix, AggregatorBaseServer
import numpy as np
from federatedml.framework.weights import Weights, NumpyWeights
from federatedml.util import LOGGER
import torch as t
from typing import Union, List
from fate_arch.computing._util import is_table
from federatedml.util import consts


AGG_TYPE = ['weighted_mean', 'sum', 'mean']


class SecureAggregatorClient(AggregatorBaseClient):

    def __init__(self, secure_aggregate=True, aggregate_type='weighted_mean', aggregate_weight=1.0,
                 communicate_match_suffix=None, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), lm_aggregate=False):

        super(SecureAggregatorClient, self).__init__(
            communicate_match_suffix=communicate_match_suffix, clients=clients, server=server)
        self.secure_aggregate = secure_aggregate
        self.suffix = {
            "local_loss": AutoSuffix("local_loss"),
            "agg_loss": AutoSuffix("agg_loss"),
            "local_model": AutoSuffix("local_model"),
            "agg_model": AutoSuffix("agg_model"),
            "converge_status": AutoSuffix("converge_status")
        }

        # init secure aggregate random padding:
        if self.secure_aggregate:
            self._random_padding_cipher: PadsCipher = RandomPaddingCipherClient(
                trans_var=RandomPaddingCipherTransVar(
                    prefix=communicate_match_suffix,
                    clients=clients,
                    server=server)).create_cipher()
            LOGGER.info('initialize secure aggregator done')

        # compute weight
        assert aggregate_type in AGG_TYPE, 'aggregate type must in {}'.format(
            AGG_TYPE)
        if aggregate_type == 'weighted_mean':
            aggregate_weight = aggregate_weight
        elif aggregate_type == 'mean':
            aggregate_weight = 1

        self.send(aggregate_weight, suffix=('agg_weight', ))
        self._weight = aggregate_weight / \
            self.get(suffix=('agg_weight', ))[0]  # local weight / total weight

        if aggregate_type == 'sum':  # reset _weight
            self._weight = 1

        self._set_table_amplify_factor = False
        self._lm_aggregate = lm_aggregate

        LOGGER.debug('aggregate compute weight is {}'.format(self._weight))

    def _handle_table_data(self, model):

        model = model.mapValues(lambda x: x * self._weight)
        if self.secure_aggregate:
            if not self._set_table_amplify_factor:
                self._random_padding_cipher.set_amplify_factor(
                    consts.SECURE_AGG_AMPLIFY_FACTOR)
            model = self._random_padding_cipher.encrypt_table(model)
        return model

    def _process_model(self, model):

        to_agg = None

        if isinstance(model, np.ndarray) or isinstance(model, Weights):
            if isinstance(model, np.ndarray):
                to_agg = NumpyWeights(model * self._weight)
            else:
                to_agg = model * self._weight

            if self.secure_aggregate:
                to_agg: Weights = to_agg.encrypted(
                    self._random_padding_cipher)
            return to_agg

        # is FATE distrubed Table
        elif is_table(model):
            return self._handle_table_data(model)

        if isinstance(model, t.nn.Module):
            if self._lm_aggregate:  
                # if model is large, cannot send large object, need to break into key/value
                # and aggregate them in the format of distributed table
                from fate_arch.session import computing_session as session
                parameters = list(model.named_parameters())
                tmp_list = [(k, v.cpu().detach().numpy()) for k, v in parameters if v.requires_grad]
                table = session.parallelize(tmp_list, include_key=True, partition=4)
                return self._handle_table_data(table)
            else:
                parameters = list(model.parameters())
                tmp_list = [[np.array(p.cpu().detach().tolist()) for p in parameters if p.requires_grad]]
                LOGGER.debug('Aggregate trainable parameters: {}/{}'.format(len(tmp_list[0]), len(parameters)))
        elif isinstance(model, t.optim.Optimizer):
            tmp_list = [[np.array(p.cpu().detach().tolist()) for p in group["params"]]
                        for group in model.param_groups]
        elif isinstance(model, list):
            for p in model:
                assert isinstance(
                    p, np.ndarray), 'expecting List[np.ndarray], but got {}'.format(p)
            tmp_list = [model]

        if self.secure_aggregate:
            to_agg = [
                [
                    NumpyWeights(
                        arr *
                        self._weight).encrypted(
                        self._random_padding_cipher) for arr in arr_list] for arr_list in tmp_list]
        else:
            to_agg = [[arr * self._weight for arr in arr_list]
                      for arr_list in tmp_list]

        return to_agg

    def _recover_model(self, model, agg_model):

        if isinstance(model, np.ndarray):
            return agg_model.unboxed
        elif isinstance(model, Weights):
            return agg_model
        elif is_table(agg_model):
            if self._lm_aggregate and isinstance(model, t.nn.Module):
                # recover weights from table
                parameters = dict(agg_model.collect())
                for k, v in model.named_parameters():
                    if k in parameters and v.requires_grad:
                        v.data.copy_(t.Tensor(parameters[k]))
                return model
            else:
                return agg_model
        else:
            if self.secure_aggregate:
                agg_model = [[np_weight.unboxed for np_weight in arr_list]
                             for arr_list in agg_model]

            if isinstance(model, t.nn.Module):
                for agg_p, p in zip(agg_model[0], [p for p in model.parameters() if p.requires_grad]):
                    p.data.copy_(t.Tensor(agg_p))

                return model
            elif isinstance(model, t.optim.Optimizer):
                for agg_group, group in zip(agg_model, model.param_groups):
                    for agg_p, p in zip(agg_group, group["params"]):
                        p.data.copy_(t.Tensor(agg_p))
                return model
            else:
                return agg_model

    def send_loss(self, loss, suffix=tuple()):
        suffix = self._get_suffix('local_loss', suffix)
        assert isinstance(loss, float) or isinstance(
            loss, np.ndarray), 'illegal loss type {}, loss should be a float or a np array'.format(type(loss))
        self.send(loss * self._weight, suffix)

    def send_model(self,
                   model: Union[np.ndarray,
                                Weights,
                                List[np.ndarray],
                                t.nn.Module,
                                t.optim.Optimizer],
                   suffix=tuple()):
        """Sending model to arbiter for aggregation

        Parameters
        ----------
        model : model can be:
                A numpy array
                A Weight instance(or subclass of Weights), see federatedml.framework.weights
                List of numpy array
                A pytorch model, is the subclass of torch.nn.Module
                A pytorch optimizer, will extract param group from this optimizer as weights to aggregate
        suffix : sending suffix, by default tuple(), can be None or tuple contains str&number. If None, will automatically generate suffix
        """
        suffix = self._get_suffix('local_model', suffix)
        # judge model type
        to_agg_model = self._process_model(model)
        self.send(to_agg_model, suffix)

    def get_aggregated_model(self, suffix=tuple()):
        suffix = self._get_suffix("agg_model", suffix)
        return self.get(suffix)[0]

    def get_aggregated_loss(self, suffix=tuple()):
        suffix = self._get_suffix("agg_loss", suffix)
        return self.get(suffix)[0]

    def get_converge_status(self, suffix=tuple()):
        suffix = self._get_suffix("converge_status", suffix)
        return self.get(suffix)[0]

    def model_aggregation(self, model, suffix=tuple()):
        self.send_model(model, suffix=suffix)
        agg_model = self.get_aggregated_model(suffix=suffix)
        return self._recover_model(model, agg_model)

    def loss_aggregation(self, loss, suffix=tuple()):
        self.send_loss(loss, suffix=suffix)
        converge_status = self.get_converge_status(suffix=suffix)
        return converge_status


class SecureAggregatorServer(AggregatorBaseServer):

    def __init__(
        self,
        secure_aggregate=True,
        communicate_match_suffix=None,
        server=(
            consts.ARBITER,
        ),
        clients=(
            consts.GUEST,
            consts.HOST)
            ):

        super(SecureAggregatorServer, self).__init__(
            communicate_match_suffix=communicate_match_suffix, clients=clients, server=server)
        self.suffix = {
            "local_loss": AutoSuffix("local_loss"),
            "agg_loss": AutoSuffix("agg_loss"),
            "local_model": AutoSuffix("local_model"),
            "agg_model": AutoSuffix("agg_model"),
            "converge_status": AutoSuffix("converge_status")
        }
        self.secure_aggregate = secure_aggregate
        if self.secure_aggregate:
            RandomPaddingCipherServer(trans_var=RandomPaddingCipherTransVar(
                prefix=communicate_match_suffix, clients=clients, server=server)).exchange_secret_keys()
            LOGGER.info('initialize secure aggregator done')

        agg_weights = self.collect(suffix=('agg_weight', ))
        sum_weights = 0
        for i in agg_weights:
            sum_weights += i
        self.broadcast(sum_weights, suffix=('agg_weight', ))

    def aggregate_model(self, suffix=None, party_idx=-1):

        # get suffix
        suffix = self._get_suffix('local_model', suffix)
        # recv params for aggregation
        models = self.collect(suffix=suffix, party_idx=party_idx)
        agg_result = None

        # Aggregate Weights or Numpy Array
        if isinstance(models[0], Weights):
            agg_result = models[0]
            for w in models[1:]:
                agg_result += w

        # Aggregate Table
        elif is_table(models[0]):
            agg_result = models[0]
            for table in models[1:]:
                agg_result = agg_result.join(table, lambda x1, x2: x1 + x2)
            return agg_result

        # Aggregate numpy groups
        elif isinstance(models[0], list):
            # aggregation
            agg_result = models[0]
            # aggregate numpy model weights from all clients
            for params_group in models[1:]:
                for agg_params, params in zip(
                        agg_result, params_group):
                    for agg_p, p in zip(agg_params, params):
                        # agg_p: NumpyWeights or numpy array
                        agg_p += p

        if agg_result is None:
            raise ValueError(
                'can not aggregate receive model, format is illegal: {}'.format(models))

        return agg_result

    def broadcast_model(self, model, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('agg_model', suffix)
        self.broadcast(model, suffix=suffix, party_idx=party_idx)

    def aggregate_loss(self, suffix=tuple(), party_idx=-1):

        # get loss
        suffix = self._get_suffix('local_loss', suffix)
        losses = self.collect(suffix, party_idx=party_idx)
        # aggregate loss
        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss += loss

        return total_loss

    def broadcast_loss(self, loss_sum, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('agg_loss', suffix)
        self.broadcast(loss_sum, suffix=suffix, party_idx=party_idx)

    def model_aggregation(self, suffix=tuple(), party_idx=-1):
        agg_model = self.aggregate_model(suffix=suffix, party_idx=party_idx)
        self.broadcast_model(agg_model, suffix=suffix, party_idx=party_idx)
        return agg_model

    def broadcast_converge_status(self, converge_status, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('converge_status', suffix)
        self.broadcast(converge_status, suffix=suffix, party_idx=party_idx)

    def loss_aggregation(self, check_converge=False, converge_func=None, suffix=tuple(), party_idx=-1):
        agg_loss = self.aggregate_loss(suffix=suffix, party_idx=party_idx)
        if check_converge:
            converge_status = converge_func(agg_loss)
        else:
            converge_status = False
        self.broadcast_converge_status(
            converge_status, suffix=suffix, party_idx=party_idx)
        return agg_loss, converge_status
