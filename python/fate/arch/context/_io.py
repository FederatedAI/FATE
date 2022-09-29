from ..unify import URI


class Reader:
    def read(self):
        ...

    def read_dataframe(self):
        ...


class CSVReader:
    def read(self):
        ...

    def read_dataframe(self):
        ...


class LibSVMReader:
    def read(self):
        ...

    def read_dataframe(self):
        ...


class ReadKit:
    def reader(self, uri: URI):
        """auto detect from uri head"""
        uri.to_schema()
        ...

    def csv(self, csv_path) -> CSVReader:
        ...

    def libsvm(
        self,
    ) -> LibSVMReader:
        ...


class Writer:
    def write(self, bytes: bytes):
        ...

    def write_dataframe(self, df):
        ...


class CSVWriter:
    ...


class LibSVMWriter:
    ...


class WriteKit:
    def writer(self, uri: URI) -> Writer:
        uri.to_schema()
        ...

    def csv(self, csv_path) -> CSVWriter:
        ...

    def libsvm(self, libsvm_path) -> LibSVMWriter:
        ...
