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

import click


@click.command()
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def task_schema(save):
    "generate component_desc task config json schema"
    from fate.components.core.spec.task import TaskConfigSpec

    if save:
        save.write(TaskConfigSpec.schema_json())
    else:
        print(TaskConfigSpec.schema_json())


if __name__ == "__main__":
    task_schema()
