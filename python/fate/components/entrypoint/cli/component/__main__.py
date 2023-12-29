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
from fate.components.entrypoint.cli.component import (
    artifact_type_cli,
    cleanup_cli,
    desc_cli,
    execute_cli,
    list_cli,
    task_schema_cli,
)

component = click.Group(name="component")
component.add_command(execute_cli.execute)
component.add_command(cleanup_cli.cleanup)
component.add_command(desc_cli.desc)
component.add_command(list_cli.list)
component.add_command(artifact_type_cli.artifact_type)
component.add_command(task_schema_cli.task_schema)

if __name__ == "__main__":
    component(prog_name="python -m fate.components.entrypoint.cli.component")
