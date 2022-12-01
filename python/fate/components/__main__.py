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
execute with python -m fate.components --execution_id xxx --config xxx
"""


if __name__ == "__main__":
    import click
    from fate.components.entrypoint.clean_cli import clean
    from fate.components.entrypoint.component_cli import component

    cli = click.Group()
    cli.add_command(component)
    cli.add_command(clean)
    cli(prog_name="python -m fate.component")
