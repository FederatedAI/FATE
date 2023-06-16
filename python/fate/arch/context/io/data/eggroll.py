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


class EggrollRawTableReader:
    def __init__(self, ctx, namespace, name, metadata: dict) -> None:
        self.ctx = ctx
        self.name = name
        self.namespace = namespace
        self.metadata = metadata

    def read_dataframe(self):
        import inspect

        from fate.arch import dataframe

        table = load_table(self.ctx, self.namespace, self.name, self.metadata)

        kwargs = {}
        p = inspect.signature(dataframe.RawTableReader.__init__).parameters
        parameter_keys = p.keys()
        for k, v in table.schema.items():
            if k in parameter_keys:
                kwargs[k] = v

        return dataframe.RawTableReader(**kwargs).to_frame(self.ctx, table)


def load_table(ctx, namespace, name, metadata: dict):
    from fate.arch.computing._address import EggRollAddress

    meta_name = f"{name}.meta"
    meta_key, meta = list(
        ctx.computing.load(
            address=EggRollAddress(name=meta_name, namespace=namespace),
            partitions=1,
            schema={},
            **metadata,
        ).collect()
    )[0]
    assert meta_key == "schema"
    num_partitions = metadata.get("num_partitions")

    return table
