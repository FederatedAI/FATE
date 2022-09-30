import functools
import numpy as np
import torch
import pandas as pd
import PIL

from fate.arch.session import computing_session
from fate.arch.tensor import FPTensor
from fate.arch.tensor.impl.tensor.distributed import FPTensorDistributed


class TableReader(object):
    def __init__(self,
                 input_format="dense",
                 with_match_id=False,
                 match_id_name=None,
                 with_label=False,
                 label_name="y",
                 label_type="int",
                 with_weight=False,
                 weight_name="weight",
                 data_type="float64"):
        self._input_format = input_format
        self._with_match_id = with_match_id
        self._match_id_name = match_id_name
        self._with_label = with_label
        self._label_name = label_name
        self._label_type = label_type
        self._with_weight = with_weight
        self._weight_name = weight_name
        self._data_type = data_type

        self._block_partition_mapping = None

    def load(self, namespace, name):
        """
        :param namespace:
        :param name:
        :return:
        """
        pass

    def to_frame(self, ctx, table):
        if self._input_format == "dense":
            return self._dense_format_to_frame(ctx, table)
        elif self._input_format in ["libsvm", "svmlight", "sparse"]:
            ...
        elif self._input_format == "tag_value":
            ...

    def _dense_format_to_frame(self, ctx, table):

        self._block_partition_mapping = _convert_to_order_indexes(table)

        schema = self._process_schema(table.schema,
                                      self._input_format,
                                      self._with_match_id,
                                      self._match_id_name,
                                      self._with_label,
                                      self._label_name,
                                      self._with_weight,
                                      self._with_weight)

        data_trans = table.mapValues(lambda value: value.split(schema["delimiter"], -1))
        data_dict = {}

        # TODO: String tensor does not support in torch, match_id is not considered yet,
        #       maybe wrapper of data frame is much better
        if self._with_match_id:
            match_id = data_trans.mapValues(lambda value: value[schema["match_id_index"]])

        if self._with_label:
            label = data_trans.mapValues(lambda value: value[schema["label_index"]])
            data_dict["label"] = _convert_to_tensor(ctx,
                                                    label,
                                                    block_partition_mapping=self._block_partition_mapping,
                                                    dtype=getattr(torch, self._label_type),
                                                    block_type="vector")

        if self._with_weight:
            weight = data_trans.mapValues(lambda value: value[schema["weight_index"]])
            data_dict["weight"] = _convert_to_tensor(ctx,
                                                     weight,
                                                     block_partition_mapping=self._block_partition_mapping,
                                                     dtype="float64",
                                                     block_type="vector")

        if schema["feature_indexes"]:
            values = data_trans.mapValues(lambda value: np.array(value)[schema["feature_indexes"]].tolist())
            data_dict["values"] = _convert_to_tensor(ctx,
                                                     values,
                                                     block_partition_mapping=self._block_partition_mapping,
                                                     dtype=getattr(torch, "float64"),
                                                     block_type="matrix")

    @staticmethod
    def _process_schema(schema, input_format, with_match_id, match_id_name,
                        with_label, label_name, with_weight, weight_name):
        if input_format == "dense":
            post_schema = dict()
            post_schema["sid"] = schema["sid"]
            post_schema["delimiter"] = schema.get("delimiter", ",")
            header = schema.get("header", {}).split(post_schema["delimiter"], -1)

            filter_indexes = []
            if with_match_id:
                post_schema["match_id_index"] = header.index(match_id_name)
                filter_indexes.append(post_schema["match_id_index"])

            if with_label:
                post_schema["label_index"] = header.index(label_name)
                filter_indexes.append(post_schema["label_index"])

            if with_weight:
                post_schema["weight_index"] = header.index(weight_name)
                filter_indexes.append(post_schema["weight_index"])

            if header:
                post_schema["feature_indexes"] = list(filter(lambda _id: _id not in filter_indexes, range(len(header))))

            return post_schema
        else:
            raise NotImplementedError


class ImageReader(object):
    """
    Image Reader now support convert image to a 3D tensor, dtype=torch.float64
    """
    def __init__(self,
                 mode="L",

                 ):
        ...


class CSVReader(object):
    def __init__(self,
                 id_name=None,
                 delimiter=",",
                 with_label=False,
                 label_name="y",
                 label_type="int",
                 with_weight=False,
                 weight_name="weight",
                 data_type="float64",
                 partition=4):
        self._id_name = id_name
        self._delimiter = delimiter
        self._with_label = with_label
        self._label_name = label_name
        self._label_type = label_type
        self._with_weight = with_weight
        self._weight_name = weight_name
        self._data_type = data_type
        self._partition = partition

    def to_frame(self, ctx, path):
        # TODO: use table put data instead of read all data
        df = pd.read_csv(path, delimiter=self._delimiter)

        return PandasReader(id_name=self._id_name,
                            with_label=self._with_label,
                            label_name=self._label_name,
                            label_type=self._label_type,
                            with_weight=self._with_weight,
                            weight_name=self._weight_name,
                            data_type=self._data_type,
                            partition=self._partition).to_frame(ctx, df)


class HiveReader(object):
    ...


class MysqlReader(object):
    ...


class TextReader(object):
    ...


class TorchDataSetReader(object):
    # TODO: this is for Torch DataSet Reader, the passing object has attributes __len__ and __get_item__
    def __init__(self, ):
        ...

    def to_frame(self, ctx, dataset):
        ...


class PandasReader(object):
    def __init__(self,
                 id_name=None,
                 with_label=False,
                 label_name="y",
                 label_type="int",
                 with_weight=False,
                 weight_name="weight",
                 data_type="float64",
                 partition=4):
        self._id_name = id_name
        self._with_label = with_label
        self._label_name = label_name
        self._label_type = label_type
        self._with_weight = with_weight
        self._weight_name = weight_name
        self._data_type = data_type
        self._partition = partition

        self._block_partition_mapping = None

    def to_frame(self, ctx, df):
        if self._id_name:
            df = df.set_index(self._id_name)

        # TODO: need to ensure id's type is str?
        df.index = df.index.astype("str")

        id_list = df.index.tolist()

        index_table = computing_session.parallelize(
            zip(id_list, range(df.shape[0])),
            include_key=True,
            partition=self._partition
        )

        self._block_partition_mapping = _convert_to_order_indexes(index_table)

        data_dict = {}
        if self._with_label:
            label_list = df[self._label_name].tolist()
            label_table = computing_session.parallelize(
                zip(id_list, label_list),
                include_key=True,
                partition=self._partition
            )
            data_dict["label"] = _convert_to_tensor(ctx,
                                                    label_table,
                                                    block_partition_mapping=self._block_partition_mapping,
                                                    dtype=getattr(torch, self._label_type),
                                                    block_type="vector")
            df = df.drop(columns=self._label_name)

        if self._with_weight:
            weight_list = df[self._weight_name].tolist()
            weight_table = computing_session.parallelize(
                zip(id_list, weight_list),
                include_key=True,
                partition=self._partition
            )
            data_dict["weight"] = _convert_to_tensor(ctx,
                                                     weight_table,
                                                     block_partition_mapping=self._block_partition_mapping,
                                                     dtype=getattr(torch, "float64"),
                                                     block_type="vector")

            df = df.drop(columns=self._weight_name)

        if df.shape[1]:
            value_table = computing_session.parallelize(
                zip(id_list, df.values),
                include_key=True,
                partition=self._partition
            )
            data_dict["values"] = _convert_to_tensor(ctx,
                                                     value_table,
                                                     block_partition_mapping=self._block_partition_mapping,
                                                     dtype=getattr(torch, self._data_type),
                                                     block_type="matrix")


def _convert_to_order_indexes(table):
    def _get_block_summary(kvs):
        key = next(kvs)
        block_size = 1 + len(kvs)
        return {key: block_size}

    block_summary = table.mapPartitions(_get_block_summary).reduce(lambda blk1, blk2: {**blk1, **blk2})

    start_index, block_id = 0, 0
    block_partition_mapping = dict()
    for blk_key, blk_size in block_summary.items():
        block_partition_mapping[blk_key] = dict(start_index=start_index,
                                                end_index=start_index + blk_size - 1,
                                                block_id=block_id)
        start_index += blk_size
        block_id += 1

    return block_partition_mapping


def _convert_to_tensor(ctx, table, block_partition_mapping, dtype, block_type):
    def _convert_block(kvs):
        ret = []
        block_id = None
        for key, value in kvs:
            if block_id is None:
                block_id = block_partition_mapping[key]["block_id"]

            if block_type == "vector":
                ret.extend(value)
            else:
                ret.append(value)

        return block_id, torch.tensor(ret, dtype=dtype)

    return FPTensor(ctx, FPTensorDistributed(
        table.mapPartitions(_convert_block, use_previous_behavior=False)))
