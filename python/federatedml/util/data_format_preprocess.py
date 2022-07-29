#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import numpy as np


DEFAULT_LABEL_NAME = "label"
DEFAULT_MATCH_ID_PREFIX = "match_id"
SVMLIGHT_COLUMN_PREFIX = "x"


class DataFormatPreProcess(object):
    @staticmethod
    def get_feature_offset(meta):
        """
        works for sparse/svmlight/tag value data
        """
        with_label = meta.get("with_label", False)
        with_match_id = meta.get("with_match_id", False)
        id_column_num = meta.get("id_column_num", 0)

        if with_match_id:
            if not id_column_num:
                id_column_num = 1

        offset = id_column_num
        if with_label:
            offset += 1

        return offset

    @staticmethod
    def agg_partition_tags(kvs, delimiter=",", offset=0, tag_with_value=True, tag_value_delimiter=":"):
        tag_set = set()

        for _, value in kvs:
            cols = value.split(delimiter, -1)[offset:]
            if tag_with_value:
                tag_set |= set([col.split(tag_value_delimiter, -1)[0] for col in cols])
            else:
                tag_set |= set(cols)

        return tag_set

    @staticmethod
    def get_tag_list(data, schema):
        if "meta" not in schema:
            raise ValueError("Meta not in schema")

        meta = schema["meta"]
        if meta["input_format"] != "tag":
            raise ValueError("Input DataFormat Should Be Tag Or Tag Value")

        delimiter = meta["delimiter"]
        tag_with_value = meta["tag_with_value"]
        tag_value_delimiter = meta["tag_value_delimiter"]

        offset = DataFormatPreProcess.get_feature_offset(meta)

        agg_func = functools.partial(DataFormatPreProcess.agg_partition_tags,
                                     delimiter=delimiter,
                                     offset=offset,
                                     tag_with_value=tag_with_value,
                                     tag_value_delimiter=tag_value_delimiter)

        agg_tags = data.applyPartitions(agg_func).reduce(lambda tag_set1, tag_set2: tag_set1 | tag_set2)

        return sorted(agg_tags)

    @staticmethod
    def get_lib_svm_dim(data, schema):
        if "meta" not in schema:
            raise ValueError("Meta not in schema")

        meta = schema["meta"]
        if "input_format" == ["sparse", "svmlight"]:
            raise ValueError("Input DataFormat Should Be SVMLight")

        delimiter = meta.get("delimiter", " ")

        offset = DataFormatPreProcess.get_feature_offset(meta)

        max_dim = data.\
            mapValues(
                lambda value: max([int(fid_value.split(":", -1)[0]) for fid_value in value.split(delimiter, -1)[offset:]])).\
            reduce(lambda x, y: max(x, y))

        return max_dim

    @staticmethod
    def generate_header(data, schema):
        if "meta" not in schema:
            raise ValueError("Meta not in schema")

        meta = schema["meta"]
        input_format = meta.get("input_format")
        delimiter = meta.get("delimiter", ",")
        if not input_format:
            raise ValueError("InputFormat should be configured.")

        generated_header = dict(original_index_info=dict())
        if input_format == "dense":
            if "header" not in schema:
                raise ValueError("Dense input data must have schema")
            if isinstance(schema["header"], str):
                header = schema["header"].split(delimiter, -1)
            else:
                header = schema["header"]
            header_index_mapping = dict(zip(header, range(len(header))))
            with_label = meta.get("with_label", False)
            id_list = meta.get("id_list", [])
            with_match_id = meta.get("with_match_id", False)

            filter_ids = set()

            if with_match_id:
                if not id_list:
                    match_id_name = header[0]
                    match_id_index = [0]
                    filter_ids.add(0)
                else:
                    match_id_name = []
                    match_id_index = []
                    for _id in id_list:
                        if _id in header_index_mapping:
                            match_id_name.append(_id)
                            match_id_index.append(header_index_mapping[_id])
                            filter_ids.add(match_id_index[-1])

                generated_header["match_id_name"] = match_id_name
                generated_header["original_index_info"]["match_id_index"] = match_id_index

            if with_label:
                label_name = meta["label_name"]
                label_index = header_index_mapping[label_name]
                generated_header["label_name"] = label_name
                generated_header["original_index_info"]["label_index"] = label_index
                filter_ids.add(label_index)

            header_ids = list(filter(lambda ids: ids not in filter_ids, range(len(header))))
            generated_header["original_index_info"]["header_index"] = header_ids
            generated_header["header"] = np.array(header)[header_ids].tolist()
        else:
            if input_format == "tag":
                sorted_tag_list = DataFormatPreProcess.get_tag_list(data, schema)
                generated_header["header"] = sorted_tag_list
            elif input_format in ["sparse", "svmlight"]:
                max_dim = DataFormatPreProcess.get_lib_svm_dim(data, schema)
                generated_header["header"] = [SVMLIGHT_COLUMN_PREFIX + str(i) for i in range(max_dim + 1)]
            else:
                raise NotImplementedError(f"InputFormat {input_format} is not implemented")

            with_label = meta.get("with_label", False)
            with_match_id = meta.get("with_match_id", False)
            id_column_num = meta.get("id_column_num", 0)

            if with_match_id:
                if not id_column_num:
                    id_column_num = 1

                if id_column_num == 1:
                    generated_header["match_id_name"] = DEFAULT_MATCH_ID_PREFIX
                else:
                    generated_header["match_id_name"] = [DEFAULT_MATCH_ID_PREFIX + str(i) for i in range(id_column_num)]

            if with_label:
                generated_header["label_name"] = DEFAULT_LABEL_NAME

        return generated_header

    @staticmethod
    def reconstruct_header(schema):
        original_index_info = schema.get("original_index_info")
        if not original_index_info:
            return schema["header"]

        header_index_mapping = dict()
        if original_index_info["header_index"]:
            for idx, col_name in zip(original_index_info["header_index"], schema["header"]):
                header_index_mapping[col_name] = idx

        if original_index_info["match_id_index"]:
            match_id_name = schema["match_id_name"]
            match_id_index = original_index_info["match_id_index"]
            if isinstance(match_id_name, str):
                header_index_mapping[match_id_index[0]] = match_id_name
            else:
                for idx, col_name in zip(match_id_index, match_id_name):
                    header_index_mapping[idx] = col_name

        if original_index_info["label_index"]:
            header_index_mapping[original_index_info["label_index"]] = schema["label_name"]

        original_header = []
        for idx, col_name in header_index_mapping.items():
            original_header[idx] = col_name

        return original_header
