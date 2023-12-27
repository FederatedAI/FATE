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
@click.option("--save", type=click.File(mode="w", lazy=True), help="save list output to specified file in json format")
def list(save):
    "list all components"
    from fate.components.core import list_components

    if save:
        import json

        json.dump(list_components(), save)
    else:
        print(list_components())


if __name__ == "__main__":
    list()
