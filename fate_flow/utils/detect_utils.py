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
import typing


def check_config(config: typing.Dict, required_arguments: typing.List):
    no_arguments = []
    error_arguments = []
    for argument in required_arguments:
        if isinstance(argument, tuple):
            if config.get(argument[0], None) != argument[1]:
                error_arguments.append(argument)
        elif argument not in config:
            no_arguments.append(argument)
    if no_arguments or error_arguments:
        raise Exception('the following arguments are required: {} {}'.format(','.join(no_arguments), ','.join(['{}={}'.format(a[0], a[1]) for a in error_arguments])))
