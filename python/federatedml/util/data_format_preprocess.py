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
import copy
import functools
import numpy as np


DEFAULT_LABEL_NAME = "label"
DEFAULT_MATCH_ID_PREFIX = "match_id"
SVMLIGHT_COLUMN_PREFIX = "x"
DEFAULT_SID_NAME = "sid"
DELIMITER = ","


class DataFormatPreProcess(object):
    @staticmethod
    def get_feature_offset(meta):
        """
        works for sparse/svmlight/tag value data
        """
        with_label = meta.get("with_label", False)
        with_match_id = meta.get("with_match_id", False)
        id_range = meta.get("id_range", 0)

        if with_match_id:
            if not id_range:
                id_range = 1

        offset = id_range
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
        if not isinstance(tag_with_value, bool):
            raise ValueError(f"tag with value should be bool, bug {tag_with_value} find")
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
                lambda value:
                max([int(fid_value.split(":", -1)[0]) for fid_value in value.split(delimiter, -1)[offset:]])).\
            reduce(lambda x, y: max(x, y))

        return max_dim

    @staticmethod
    def generate_header(data, schema):
        if not schema.get('meta'):
            raise ValueError("Meta not in schema")

        meta = schema["meta"]
        generated_header = dict(original_index_info=dict(), meta=meta)
        input_format = meta.get("input_format")
        delimiter = meta.get("delimiter", ",")
        if not input_format:
            raise ValueError("InputFormat should be configured.")

        if input_format == "dense":
            if "header" not in schema:
                raise ValueError("Dense input data must have schema")

            header = schema["header"].strip().split(delimiter, -1)
            header = list(map(lambda col: col.strip(), header))
            header_index_mapping = dict(zip(header, range(len(header))))
            with_label = meta.get("with_label", False)
            if not isinstance(with_label, bool):
                raise ValueError("with_label should be True or False")
            id_list = meta.get("id_list", [])
            if not isinstance(id_list, (type(None), list)):
                raise ValueError("id_list should be list type or None")

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
                        else:
                            raise ValueError(f"Can not find {_id} in id_list in data's header")

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
            id_range = meta.get("id_range", 0)

            if id_range and not with_match_id:
                raise ValueError(f"id_range {id_range} != 0, with_match_id should be true")

            if with_match_id:
                if not id_range:
                    id_range = 1

                if id_range == 1:
                    generated_header["match_id_name"] = DEFAULT_MATCH_ID_PREFIX
                else:
                    generated_header["match_id_name"] = [DEFAULT_MATCH_ID_PREFIX + str(i) for i in range(id_range)]

            if with_label:
                generated_header["label_name"] = DEFAULT_LABEL_NAME

            if id_range:
                generated_header["meta"]["id_range"] = id_range

            generated_header["is_display"] = False

        generated_header["sid"] = schema.get("sid", DEFAULT_SID_NAME).strip()

        return generated_header

    @staticmethod
    def reconstruct_header(schema):
        original_index_info = schema.get("original_index_info")
        if not original_index_info:
            return schema["header"]

        header_index_mapping = dict()
        if "header_index" in original_index_info and original_index_info["header_index"]:
            for idx, col_name in zip(original_index_info["header_index"], schema["header"]):
                header_index_mapping[idx] = col_name

        if original_index_info.get("match_id_index") is not None:
            match_id_name = schema["match_id_name"]
            match_id_index = original_index_info["match_id_index"]
            if isinstance(match_id_name, str):
                header_index_mapping[match_id_index[0]] = match_id_name
            else:
                for idx, col_name in zip(match_id_index, match_id_name):
                    header_index_mapping[idx] = col_name

        if original_index_info.get("label_index") is not None:
            header_index_mapping[original_index_info["label_index"]] = schema["label_name"]

        original_header = [None] * len(header_index_mapping)
        for idx, col_name in header_index_mapping.items():
            original_header[idx] = col_name

        return original_header

    @staticmethod
    def extend_header(schema, columns):
        schema = copy.deepcopy(schema)
        original_index_info = schema.get("original_index_info")

        columns = list(map(lambda column: column.strip(), columns))
        header = schema["header"]
        if isinstance(header, list):
            header.extend(columns)
            schema["header"] = header

            if original_index_info and "header_index" in original_index_info:
                header_index = original_index_info["header_index"]
                if header_index:
                    pre_max_col_idx = max(header_index)
                else:
                    pre_max_col_idx = -1

                if original_index_info.get("label_index") is not None:
                    pre_max_col_idx = max(original_index_info["label_index"], pre_max_col_idx)
                if original_index_info.get("match_id_index") is not None:
                    pre_max_col_idx = max(max(original_index_info["match_id_index"]), pre_max_col_idx)

                append_header_index = [i + pre_max_col_idx + 1 for i in range(len(columns))]

                schema["original_index_info"]["header_index"] = header_index + append_header_index
        else:
            if len(header) == 0:
                new_header = DELIMITER.join(columns)
            else:
                new_header = DELIMITER.join(header.split(DELIMITER, -1) + columns)

            schema["header"] = new_header
            if schema.get("sid") is not None:
                schema["sid"] = schema["sid"].strip()

        return schema

    @staticmethod
    def clean_header(schema):
        schema = copy.deepcopy(schema)
        header = schema["header"]

        if "label_name" in schema:
            del schema["label_name"]

        if "anonymous_header" in schema:
            del schema["anonymous_header"]

        if "anonymous_label" in schema:
            del schema["anonymous_label"]

        if isinstance(header, list):
            schema["header"] = []
            original_index_info = schema.get("original_index_info")
            if original_index_info:
                del schema["original_index_info"]

                if "match_id_name" in schema:
                    del schema["match_id_name"]

                if "match_id_index" in schema:
                    del schema["match_id_index"]
        else:
            schema["header"] = ""

        return schema
