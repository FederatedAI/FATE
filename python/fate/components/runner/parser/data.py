from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Datasets:
    train_data: Optional[Any]
    has_train_data: bool
    validate_data: Optional[Any]
    has_validate_data: bool
    test_data: Optional[Any]
    has_test_data: bool
    data: dict
    has_data: bool

    def get_datas(self):
        return self.train_data, self.validate_data, self.test_data, self.data

    @classmethod
    def parse(cls, datasets, cpn) -> "Datasets":
        class _T:
            def __init__(self, exists=False, data=None) -> None:
                self.exists = exists
                self.data = data

            def update(self, exists, data):
                self.exists = exists
                self.data = data

        @dataclass
        class _Datasets:
            train_data: _T
            eval_data: _T
            validate_data: _T
            test_data: _T
            data: _T

        datasets_ns = _Datasets(_T(), _T(), _T(), _T(), _T(data={}))
        for cpn_name, data_dict in datasets.items():
            for k, v in data_dict.items():
                if k in ["train_data", "eval_data", "validate_data", "test_data"]:
                    # NOTE: obtain_data seems stupied here
                    # cpn can control whether to unbox list of data by override obtain_data method
                    getattr(datasets_ns, k).update(True, cpn.obtain_data(v))
                # merge others in `data`
                else:
                    data = cpn.obtain_data(v)
                    if isinstance(data, list):
                        datasets_ns.data.data.update(
                            {f"{cpn_name}.{k}.{i}": d for i, d in enumerate(data)}
                        )
                    else:
                        datasets_ns.data.data[f"{cpn_name}.{k}"] = data
                    datasets_ns.data.exists = True

        # NOTE: it seems that `eval_data` exists here for back compatibility, could be remove in 2.0?
        if datasets_ns.eval_data.exists:
            if datasets_ns.validate_data.exists or datasets_ns.test_data.exists:
                raise DataConfigError(
                    "eval_data input should not be configured simultaneously"
                    " with validate_data or test_data"
                )
        if datasets_ns.train_data.exists:
            # "eval_data" overwrite "validate_data" ?
            if datasets_ns.eval_data.exists:
                datasets_ns.validate_data.data = datasets_ns.eval_data.data
                datasets_ns.validate_data.exists = True
        # without training, nothing to validate, should we raise error?
        elif datasets_ns.validate_data.exists:
            datasets_ns.validate_data.data = None
            datasets_ns.validate_data.exists = False

        # `eval_data` as `test_data` if train_data` not provided
        if datasets_ns.eval_data.exists and not datasets_ns.train_data.exists:
            datasets_ns.test_data.data = datasets_ns.eval_data.data

        return Datasets(
            train_data=datasets_ns.train_data.data,
            has_train_data=datasets_ns.train_data.exists,
            validate_data=datasets_ns.validate_data.data,
            has_validate_data=datasets_ns.validate_data.exists,
            test_data=datasets_ns.test_data.data,
            has_test_data=datasets_ns.test_data.exists,
            data=datasets_ns.data.data,
            has_data=datasets_ns.data.exists,
        )

    def schema(self):
        for d in [self.train_data, self.validate_data, self.test_data]:
            if d is not None:
                return d.schema
        return None


class DataConfigError(ValueError):
    pass
