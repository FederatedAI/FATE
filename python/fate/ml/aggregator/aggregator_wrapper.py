#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import torch.distributed as dist
from transformers.training_args import TrainingArguments
from fate.ml.aggregator import AggregatorType, aggregator_map


logger = logging.getLogger(__name__)


class AggregatorClientWrapper(object):
    def __init__(
            self,
            ctx,
            aggregate_type,
            aggregator_name,
            aggregator,
            sample_num,
            args: TrainingArguments,
            master_rank=0
    ):
        assert aggregator in {
            item.value for item in AggregatorType
        }, f"aggregator should be one of {{item.value for item in AggregatorType}}, but got {aggregator}"

        self._args = args
        self._world_size = args.world_size
        self._local_rank = args.local_rank
        self._master_rank = master_rank
        self._aggregator = None

        if self._is_master():
            client_class = aggregator_map[aggregator][0]
            logger.info(f"Using {aggregator} aggregator")
            ctx.arbiter.put("agg_type", aggregator)

            aggregator = client_class(
                ctx, aggregate_type=aggregate_type, aggregator_name=aggregator_name, sample_num=sample_num
            )

            self._aggregator = aggregator

    def model_aggregation(self, ctx, model):
        self._gather_model(model)

        if self._is_master():
            self._aggregator.model_aggregation(ctx, model)

        self._sync_model(model)

    def _is_master(self):
        return self._world_size <= 1 or self._local_rank == self._master_rank

    def _sync_model(self, model, src=0, sync_trainable_only=True):
        if self._world_size <= 1:
            return

        if self._local_rank == self._master_rank:
            for p in model.parameters():
                if (not sync_trainable_only) or (sync_trainable_only and p.requires_grad):
                    scatter_list = [p.data for _ in range(self._world_size)]
                    dist.scatter(p.data, scatter_list, async_op=False)
        else:
            for p in model.parameters():
                if (not sync_trainable_only) or (sync_trainable_only and p.requires_grad):
                    dist.scatter(p.data, src=src, async_op=False)

    def _gather_model(self, model):
        if self._world_size <= 1:
            return

        if not self._args.deepspeed or not self._args.hf_deepspeed_config.is_zero3():
            return

        while hasattr(model, "module"):
            model = model.module

        for _, p in model.named_parameters():
            p.all_gather()


class AggregatorServerWrapper(object):
    def __init__(self, ctx):
        aggregator = [ctx.guest.get("agg_type")]
        aggregator.extend(ctx.hosts.get("agg_type"))
        aggregator = set(aggregator)
        if len(aggregator) > 1:
            raise ValueError("Aggregator type should be the same between clients, but got {}".format(aggregator))
        aggregator = aggregator.pop()
        aggregator_name = "fedavg"
        aggregator_server = aggregator_map[aggregator][1]
        logger.info(f"Using {aggregator} aggregator")
        aggregator = aggregator_server(ctx, aggregator_name=aggregator_name)

        self._aggregator = aggregator

    def model_aggregation(self, ctx):
        self._aggregator.model_aggregation(ctx)
