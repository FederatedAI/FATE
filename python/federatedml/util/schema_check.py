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


def check_schema(input_schema, output_schema):
    LOGGER.debug(f"input schema: {input_schema} -> output schema: {output_schema}")
    if output_schema is None:
        raise EnvironmentError(
            f"output_schema is None while input data has schema.")

    input_header = input_schema.get("header", None)
    output_header = output_schema.get("header", None)
    if input_header is not None and output_header is None:
        raise EnvironmentError(
            f"output header is None while input data has header.")


def assert_schema_consistent(func):
    def _func(*args, **kwargs):
        input_schema = None
        all_args = []
        all_args.extend(args)
        all_args.extend(kwargs.values())
        for arg in all_args:
            if is_table(arg):
                input_schema = arg.schema
                break
        result = func(*args, **kwargs)
        if input_schema is not None:
            # single data set
            if is_table(result) and result.count() > 0:
                output_schema = result.schema
                check_schema(input_schema, output_schema)

            # multiple data sets
            elif type(result).__name__ in ["list", "tuple"]:
                for output_data in result:
                    if is_table(output_data) and output_data.count() > 0:
                        output_schema = output_data.schema
                        check_schema(input_schema, output_schema)
        return result

    return _func
