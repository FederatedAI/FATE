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

"""
execute with python -m fate.components.runner --session_id xxx --type xxx --address xxx
"""
if __name__ == "__main__":
    import argparse

    arguments = argparse.ArgumentParser()
    arguments.add_argument("session_id")
    arguments.add_argument("type")
    arguments.add_argument("address")
    args = arguments.parse_args()

    if args.type == "execute_component":
        from fate.components.runner.entrypoint.exec_component import task_execute

        task_execute(args.address)
    elif args.type == "clean_task":
        from fate.components.runner.entrypoint.clean_task import task_clean

        task_clean()
    elif args.type == "validate_params":
        from fate.components.runner.entrypoint.validate_params import params_validate

        params_validate()
    else:
        raise RuntimeError(f"task type `{args.type}` unknown")
