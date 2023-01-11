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

from ....unify import EggrollURI


class EggrollDataFrameWriter:
    def __init__(self, ctx, uri: EggrollURI, metadata: dict) -> None:
        self.ctx = ctx
        self.uri = EggrollMetaURI(uri)
        self.metadata = metadata

    def write_dataframe(self, df):
        from fate.arch import dataframe
        from fate.arch.computing._address import EggRollAddress

        table = dataframe.serialize(self.ctx, df)
        schema = {}
        table.save(
            address=EggRollAddress(name=self.uri.get_data_name(), namespace=self.uri.get_data_namespace()),
            partitions=int(self.metadata.get("num_partitions", table.partitions)),
            schema=schema,
            **self.metadata,
        )
        # save meta
        meta_table = self.ctx.computing.parallelize([("schema", schema)], partition=1, include_key=True)
        meta_table.save(
            address=EggRollAddress(name=self.uri.get_meta_name(), namespace=self.uri.get_meta_namespace()),
            partitions=1,
            schema={},
            **self.metadata,
        )


class EggrollDataFrameReader:
    def __init__(self, ctx, uri: EggrollURI, metadata: dict) -> None:
        self.ctx = ctx
        self.uri = EggrollMetaURI(uri)
        self.metadata = metadata

    def read_dataframe(self):
        from fate.arch import dataframe

        from .df import Dataframe

        table = load_table(self.ctx, self.uri, self.metadata)
        df = dataframe.deserialize(self.ctx, table)
        return Dataframe(df, df.shape[1], df.shape[0])


class EggrollRawTableReader:
    def __init__(self, ctx, name: str, uri: EggrollURI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = EggrollMetaURI(uri)
        self.metadata = metadata

    def read_dataframe(self):
        import inspect

        from fate.arch import dataframe

        from .df import Dataframe

        table = load_table(self.ctx, self.uri, self.metadata)

        kwargs = {}
        p = inspect.signature(dataframe.RawTableReader.__init__).parameters
        parameter_keys = p.keys()
        for k, v in table.schema.items():
            if k in parameter_keys:
                kwargs[k] = v

        dataframe_reader = dataframe.RawTableReader(**kwargs).to_frame(self.ctx, table)
        return Dataframe(dataframe_reader, dataframe_reader.shape[1], dataframe_reader.shape[0])


class EggrollMetaURI:
    def __init__(self, uri: EggrollURI) -> None:
        self.uri = uri

    def get_data_namespace(self):
        return self.uri.namespace

    def get_data_name(self):
        return self.uri.name

    def get_meta_namespace(self):
        return self.uri.namespace

    def get_meta_name(self):
        return f"{self.uri.name}.meta"


def load_table(ctx, uri: EggrollMetaURI, metadata: dict):
    from fate.arch.computing._address import EggRollAddress

    meta_key, meta = list(
        ctx.computing.load(
            address=EggRollAddress(name=uri.get_meta_name(), namespace=uri.get_meta_namespace()),
            partitions=1,
            schema={},
            **metadata,
        ).collect()
    )[0]
    assert meta_key == "schema"
    num_partitions = metadata.get("num_partitions")
    table = ctx.computing.load(
        address=EggRollAddress(name=uri.get_data_name(), namespace=uri.get_data_namespace()),
        partitions=num_partitions,
        schema=meta,
        **metadata,
    )

    return table
