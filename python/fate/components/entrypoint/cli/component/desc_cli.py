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
@click.option("--name", required=True, help="name of component_desc")
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def desc(name, save):
    "generate component_desc describe config"
    from fate.components.core import load_component

    cpn = load_component(name)
    if save:
        cpn.dump_yaml(save)
    else:
        print(cpn.dump_yaml())


if __name__ == "__main__":
    desc()
