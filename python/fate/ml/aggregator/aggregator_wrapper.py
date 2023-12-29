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
import numpy as np
import torch
import torch.distributed as dist
from transformers.training_args import TrainingArguments
from fate.ml.aggregator import AggregatorType, aggregator_map


logger = logging.getLogger(__name__)


class AggregatorClientWrapper(object):
    def __init__(
        self, ctx, aggregate_type, aggregator_name, aggregator, sample_num, args: TrainingArguments, master_rank=0
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
            logger.info(f"Using {aggregator} aggregator, rank={args.local_rank}")
            ctx.arbiter.put("agg_type", aggregator)

            aggregator = client_class(
                ctx, aggregate_type=aggregate_type, aggregator_name=aggregator_name, sample_num=sample_num
            )

            self._aggregator = aggregator

    def model_aggregation(self, ctx, model):
        logger.info(f"begin to agg model")
        self._gather_model(model)

        if self._is_master():
            self._aggregator.model_aggregation(ctx, model)

        self._sync_model(model)
        logger.info(f"begin to agg model")

    def loss_aggregation(self, ctx, loss):
        logger.info(f"begin to agg loss")
        loss = self._gather_loss(loss)

        logger.info(f"end to gather loss")

        if isinstance(loss, (int, float)):
            loss_type = "single"
        else:
            loss_type = "multi"

        if self._is_master():
            loss = self._aggregator.loss_aggregation(ctx, loss)
            self._sync_loss(loss, loss_type=loss_type)
        else:
            loss = self._sync_loss(loss, loss_type=loss_type)

        logger.info(f"end to agg loss")
        return loss

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
        if not self._args.deepspeed or not self._args.hf_deepspeed_config.is_zero3():
            return

        while hasattr(model, "module"):
            model = model.module

        for _, p in model.named_parameters():
            p.all_gather()

    def _gather_loss(self, loss):
        if self._world_size <= 1:
            return loss

        loss = torch.Tensor([loss]).cuda(self._args.device)
        if self._is_master():
            loss_list = [loss for _ in range(self._world_size)]
            dist.gather(loss, gather_list=loss_list, async_op=False)
            loss_sum = None
            for _l in loss_list:
                if loss_sum is None:
                    loss_sum = _l.item()
                else:
                    loss_sum += _l.item()
            return loss_sum / self._world_size
        else:
            dist.gather(loss, dst=0, async_op=False)
            return loss

    def _sync_loss(self, loss, loss_type):
        if self._world_size <= 1:
            return

        if loss_type == "single":
            if isinstance(loss, (int, float)):
                loss = [loss]
            if isinstance(loss, (list, np.ndarray)):
                loss = np.array(loss)

        if self._is_master():
            loss = torch.Tensor(loss).cuda(self._args.device)
            loss_list = [loss for _ in range(self._world_size)]
            dist.scatter(loss, loss_list, async_op=False)
        else:
            loss = torch.Tensor(loss).cuda(self._args.device)
            dist.scatter(loss, src=0, async_op=False)
            return loss[0].item()


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

    def loss_aggregation(self, ctx):
        return self._aggregator.loss_aggregation(ctx)
