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

import os
import sys
import json
import argparse
import traceback
from federatedml.ftl.data_util.common_data_util import save_data_to_eggroll_table, generate_table_namespace_n_name
from federatedml.ftl.data_util.uci_credit_card_util import load_guest_host_generators_for_UCI_Credit_Card
from arch.api.session import init


def convert_and_save_source_data_to_eggroll_table(config_data):

    file_path = config_data.get("file_path")
    overlap_ratio = config_data.get("overlap_ratio")
    guest_split_ratio = config_data.get("guest_split_ratio")
    n_feature_guest = config_data.get("n_feature_guest")
    num_samples = config_data.get("num_samples")
    balanced = config_data.get("balanced")

    if not os.path.exists(file_path):
        print(file_path, "is not exist, please check the configure")
        sys.exit()

    guest_data_generator, host_data_generator, overlap_indexes = load_guest_host_generators_for_UCI_Credit_Card(
        file_path=file_path,
        num_samples=num_samples,
        overlap_ratio=overlap_ratio,
        guest_split_ratio=guest_split_ratio,
        guest_feature_num=n_feature_guest,
        balanced=balanced)

    namespace, table_name = generate_table_namespace_n_name(file_path)
    guest_table_name = table_name + "_guest"
    host_table_name = table_name + "_host"
    guest_table = save_data_to_eggroll_table(data=guest_data_generator, namespace=namespace,
                                             table_name=guest_table_name)
    host_table = save_data_to_eggroll_table(data=host_data_generator, namespace=namespace,
                                            table_name=host_table_name)

    guest_table_count = guest_table.count()
    host_table_count = host_table.count()

    # save data meta to a json file
    print("overlap_indexes[0]", overlap_indexes[0], type(overlap_indexes[0]))
    print("overlap_indexes[0]", overlap_indexes[-1], type(overlap_indexes[-1]))
    output = dict()
    output["guest_table_namespace"] = namespace
    output["guest_table_name"] = guest_table_name
    output["guest_table_count"] = guest_table_count
    output["n_feature_guest"] = n_feature_guest
    output["host_table_namespace"] = namespace
    output["host_table_name"] = host_table_name
    output["host_table_count"] = host_table_count
    output["overlap_index_range"] = {"start": int(overlap_indexes[0]), "end": int(overlap_indexes[-1])}

    with open('./guest_host_table_metadata.json', 'w') as outfile:
        json.dump(output, outfile)

    print("------------save data finish!-----------------")
    print("namespace:%s, guest_table_name:%s, host_table_name:%s" % (namespace, guest_table_name, host_table_name))
    print(output)

    return output


if __name__ == '__main__':
    init()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, help="指定一个配置文件路径(json格式)")
    try:
        args = parser.parse_args()
        if not args.config:
            print("找不到配置文件")
            sys.exit()

        data = {}
        try:
            args.config = os.path.abspath(args.config)
            input_file_path = None
            head = None
            with open(args.config, 'r') as f:
                config__data = json.load(f)

            convert_and_save_source_data_to_eggroll_table(config__data)

        except ValueError:
            print('json解析错误')
            exit(-102)
        except IOError:
            print('文件读取错误')
            exit(-103)
    except:
        traceback.print_exc()




