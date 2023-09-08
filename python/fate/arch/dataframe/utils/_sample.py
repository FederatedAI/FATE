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
import functools
from typing import Union, Dict, Any

from sklearn.utils import resample

from ._id_generator import generate_sample_id, generate_sample_id_prefix
from .._dataframe import DataFrame

REGENERATED_TAG = "regenerated_index"
SAMPLE_INDEX_TAG = "sample_index"
REGENERATED_IDS = "regenerated_ids"


def local_sample(
    ctx,
    df: DataFrame,
    n: int=None,
    frac: Union[float, Dict[Any, float]] = None,
    replace: bool = True,
    random_state=None
):
    return _sample_guest(ctx, df, n, frac, replace, random_state, sync=False)


def federated_sample(
        ctx,
        df: DataFrame,
        n: int = None,
        frac: Union[float, Dict[Any, float]] = None,
        replace: bool = True,
        random_state=None,
        role: str = "guest"):
    if role == "guest":
        return _sample_guest(ctx, df, n, frac, replace, random_state, sync=True)
    else:
        return _federated_sample_host(ctx, df)


def _sample_guest(
    ctx,
    df: DataFrame,
    n: int = None,
    frac: Union[float, Dict[Any, float]] = None,
    replace: bool = True,
    random_state=None,
    sync=True,
):
    if n is not None and frac is not None:
        raise ValueError("sample's parameters n and frac should not be set in the same time.")

    if frac is not None:
        if isinstance(frac, float):
            if frac > 1:
                raise ValueError(f"sample's parameter frac={frac} should <= 1.0")
            n = max(1, int(frac * df.shape[0]))
        else:
            for k, f in frac.items():
                if f > 1 and replace is False:
                    raise ValueError(f"sample's parameter frac's label={k}, fraction={f} "
                                     f"should <= 1.0 if replace=False")

    if n is not None:
        if n > df.shape[0] and replace is False:
            raise ValueError(f"sample's parameter n={n} should <= data_size={df.shape[0]} if replace=False")

        if replace:
            choices = resample(list(range(df.shape[0])), replace=True, n_samples=n, random_state=random_state)
            indexer = list(df.get_indexer(target="sample_id").collect())
            regenerated_sample_id_prefix = generate_sample_id_prefix()
            regenerated_ids = generate_sample_id(n, regenerated_sample_id_prefix)
            choice_with_regenerated_ids = _agg_choices(ctx,
                                                       indexer,
                                                       choices,
                                                       regenerated_ids,
                                                       df.block_table.partitions)

            if sync:
                ctx.hosts.put(REGENERATED_TAG, True)
                ctx.hosts.put(REGENERATED_IDS, choice_with_regenerated_ids)

            regenerated_raw_table = _regenerated_sample_ids(df, choice_with_regenerated_ids)
            sample_df = _convert_raw_table_to_df(df._ctx, regenerated_raw_table, df.data_manager)
            if sync:
                sample_indexer = sample_df.get_indexer(target="sample_id")
                ctx.hosts.put(SAMPLE_INDEX_TAG, sample_indexer)

        else:
            sample_df = df.sample(n=n, random_state=random_state)
            if sync:
                sample_indexer = sample_df.get_indexer(target="sample_id")
                ctx.hosts.put(REGENERATED_TAG, False)
                ctx.hosts.put(SAMPLE_INDEX_TAG, sample_indexer)
    else:
        up_sample = False
        for label, f in frac.items():
            if f > 1.0:
                up_sample = True

        if up_sample:
            regenerated_sample_id_prefix = generate_sample_id_prefix()
            choice_with_regenerated_ids = None
            for label, f in frac.items():
                label_df = df.iloc(df.label == label)
                label_n = max(1, int(label_df.shape[0] * f))
                choices = resample(list(range(label_df.shape[0])), replace=True,
                                   n_samples=label_n, random_state=random_state)
                label_indexer = list(label_df.get_indexer(target="sample_id").collect())
                regenerated_ids = generate_sample_id(label_n, regenerated_sample_id_prefix)
                label_choice_with_regenerated_ids = _agg_choices(ctx, label_indexer, choices,
                                                                 regenerated_ids, df.block_table.partitions)
                if choice_with_regenerated_ids is None:
                    choice_with_regenerated_ids = label_choice_with_regenerated_ids
                else:
                    choice_with_regenerated_ids = choice_with_regenerated_ids.union(label_choice_with_regenerated_ids)

            if sync:
                ctx.hosts.put(REGENERATED_TAG, True)
                ctx.hosts.put(REGENERATED_IDS, choice_with_regenerated_ids)
            regenerated_raw_table = _regenerated_sample_ids(df, choice_with_regenerated_ids)
            sample_df = _convert_raw_table_to_df(df._ctx, regenerated_raw_table, df.data_manager)
            if sync:
                sample_indexer = sample_df.get_indexer(target="sample_id")
                ctx.hosts.put(SAMPLE_INDEX_TAG, sample_indexer)
        else:
            sample_df = None
            for label, f in frac.items():
                label_df = df.iloc(df.label == label)
                label_n = max(1, int(label_df.shape[0] * f))
                sample_label_df = label_df.sample(n=label_n, random_state=random_state)

                if sample_df is None:
                    sample_df = sample_label_df
                else:
                    sample_df = DataFrame.vstack([sample_df, sample_label_df])

            if sync:
                sample_indexer = sample_df.get_indexer(target="sample_id")
                ctx.hosts.put(REGENERATED_TAG, False)
                ctx.hosts.put(SAMPLE_INDEX_TAG, sample_indexer)

    return sample_df


def _federated_sample_host(
    ctx,
    df: DataFrame
):
    regenerated_tag = ctx.guest.get(REGENERATED_TAG)
    if regenerated_tag is False:
        sample_indexer = ctx.guest.get(SAMPLE_INDEX_TAG)
        sample_df = df.loc(sample_indexer, preserve_order=True)
    else:
        regenerated_ids = ctx.guest.get(REGENERATED_IDS)
        regenerated_raw_table = _regenerated_sample_ids(df, regenerated_ids)
        sample_df = _convert_raw_table_to_df(df._ctx, regenerated_raw_table, df.data_manager)

        sample_indexer = ctx.guest.get(SAMPLE_INDEX_TAG)
        sample_df = sample_df.loc(sample_indexer, preserve_order=True)

    return sample_df


def _regenerated_sample_ids(df, regenerated_ids):
    from ..ops._indexer import regenerated_sample_id
    regenerated_raw_table = regenerated_sample_id(df.block_table, regenerated_ids, df.data_manager)

    return regenerated_raw_table


def _convert_raw_table_to_df(
    ctx,
    table,
    data_manager
):
    from ..ops._indexer import get_partition_order_by_raw_table
    from ..ops._dimension_scaling import to_blocks
    partition_order_mapping = get_partition_order_by_raw_table(table, data_manager.block_row_size)
    to_block_func = functools.partial(to_blocks, dm=data_manager, partition_mappings=partition_order_mapping)
    block_table = table.mapPartitions(to_block_func,
                                      use_previous_behavior=False)

    return DataFrame(
        ctx,
        block_table,
        partition_order_mapping,
        data_manager
    )


def _agg_choices(ctx,
                 indexer,
                 choices,
                 regenerated_ids,
                 partition):
    """
    indexer: (sample_id, (partition_id, block_offset))
    """
    choice_dict = dict()
    choice_indexer = []
    for idx, choice in enumerate(choices):
        if choice not in choice_dict:
            current_l = len(choice_dict)
            choice_dict[choice] = current_l
            choice_indexer.append([])

        choice_indexer[choice_dict[choice]].append(regenerated_ids[idx])

    for choice, idx in choice_dict.items():
        choice_regenerated_sample_ids = choice_indexer[idx]
        choice_indexer[idx] = (indexer[choice][0], choice_regenerated_sample_ids)

    return ctx.computing.parallelize(choice_indexer,
                                     include_key=True,
                                     partition=partition)
