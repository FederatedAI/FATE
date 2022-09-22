from dataclasses import dataclass
from typing import List, Type

from fate.interface import CpnOutput, Module

from .context import ComponentContext, Namespace
from .parser.checkpoint import CheckpointManager
from .parser.data import Datasets
from .parser.model import PBModelsLoader, PBModelsSaver
from .parser.param import Params
from .parser.tracker import Tracker
from .procedure import Dispatcher


class Runner:
    def __init__(self, cpn_class: Type[Module], cpn_param: Params):
        self.cpn_class = cpn_class
        self.cpn_param = cpn_param

    def run(self, cpn_input, retry: bool) -> CpnOutput:

        # params
        params = self.cpn_param
        params.update(cpn_input.parameters)

        # instance cpn
        cpn = self.cpn_class(params)

        # datasets
        datasets = Datasets.parse(cpn_input.datasets, cpn)

        # model loader and saver
        models_loader = PBModelsLoader.parse(cpn_input.models)
        models_saver = PBModelsSaver()

        # create checkpoint manager
        checkpoint_manager = CheckpointManager.parse(cpn_input.checkpoint_manager)

        # init context
        tracker = Tracker.parse(cpn_input.tracker)
        role = cpn_input.roles["local"]["role"]
        party_id = cpn_input.roles["local"]["party_id"]
        namespace = Namespace()
        ctx = ComponentContext(role, party_id, tracker, checkpoint_manager, namespace)

        # dispatch to concrate procedure and run
        # notice that `Dispatcher` has `short circuit` strategy,
        # which means fisrt `Procedure` that is fulfilled returned
        data_outputs = Dispatcher.dispatch_run(
            ctx, cpn, params, datasets, models_loader, models_saver, retry
        )
        ctx.summary.save()

        @dataclass
        class _CpnOutput(CpnOutput):
            data: list
            model: dict
            cache: List[tuple]

        return _CpnOutput(data_outputs, models_saver.models, ctx.cache.cache)
