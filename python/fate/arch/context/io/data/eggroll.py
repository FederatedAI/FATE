from fate.arch.abc import CTableABC

from ....unify import EggrollURI


class EggrollDataFrameWriter:
    def __init__(self, ctx, uri: EggrollURI, metadata: dict) -> None:
        self.ctx = ctx
        self.uri = EggrollMetaURI(uri)
        self.metadata = metadata

    def write_dataframe(self, df):
        from fate.arch import dataframe
        from fate.arch.common.address import EggRollAddress

        table: CTableABC = dataframe.serialize(self.ctx, df)
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
        from fate.arch.common.address import EggRollAddress

        meta_key, meta = list(
            self.ctx.computing.load(
                address=EggRollAddress(name=self.uri.get_meta_name(), namespace=self.uri.get_meta_namespace()),
                partitions=1,
                schema={},
                **self.metadata,
            ).collect()
        )[0]
        assert meta_key == "schema"
        num_partitions = self.metadata.get("num_partitions")
        table = self.ctx.computing.load(
            address=EggRollAddress(name=self.uri.get_data_name(), namespace=self.uri.get_data_namespace()),
            partitions=num_partitions,
            schema=meta,
            **self.metadata,
        )
        df = dataframe.deserialize(self.ctx, table)
        return df


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
