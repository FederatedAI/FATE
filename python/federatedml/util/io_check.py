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
from fate_arch.computing import is_table
from federatedml.util import LOGGER


def assert_io_num_rows_equal(func):
    def _func(*args, **kwargs):
        input_count = None
        all_args = []
        all_args.extend(args)
        all_args.extend(kwargs.values())
        for arg in all_args:
            if is_table(arg):
                input_count = arg.count()
                break

        result = func(*args, **kwargs)

        if input_count is not None and is_table(result):
            output_count = result.count()
            LOGGER.debug(f"num row of input: {input_count} -> num row of output: {output_count}")
            if input_count != output_count:
                raise EnvironmentError(
                    f"num row of input({input_count}) not equals to num row of output({output_count})")
        return result

    return _func


def check_with_inst_id(data_instances):
    instance = data_instances.first()[1]
    if type(instance).__name__ == "Instance" and instance.with_inst_id:
        return True
    return False


def check_is_instance(data_instances):
    instance = data_instances.first()[1]
    if type(instance).__name__ == "Instance":
        return True
    return False


def assert_match_id_consistent(func):
    def _func(*args, **kwargs):
        input_with_inst_id = None
        all_args = []
        all_args.extend(args)
        all_args.extend(kwargs.values())
        for arg in all_args:
            if is_table(arg):
                input_with_inst_id = check_with_inst_id(arg)
                break

        result = func(*args, **kwargs)

        if input_with_inst_id is not None and is_table(result):
            if check_is_instance(result):
                result_with_inst_id = check_with_inst_id(result)
                LOGGER.debug(
                    f"Input with match id: {input_with_inst_id} -> output with match id: {result_with_inst_id}")
                if input_with_inst_id and not result_with_inst_id:
                    raise EnvironmentError(
                        f"Input with match id: {input_with_inst_id} -> output with match id: {result_with_inst_id}，"
                        f"func: {func}")
        return result

    return _func
