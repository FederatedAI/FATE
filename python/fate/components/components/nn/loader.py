import os
import sys
import importlib.util
from abc import ABC, abstractmethod
import json
import yaml
import difflib


class _Source(object):
    MODEL_ZOO = 'fate.ml.nn.model_zoo'
    DATASET = 'fate.ml.nn.dataset'
    CUST_FUNC = 'fate.ml.nn.cust_func'


SOURCE_FILE = 'source.yaml'


def is_path(s):
    return os.path.exists(s)


def load_source():
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    with open(script_dir + '/' + SOURCE_FILE, 'r') as f:
        source = yaml.safe_load(f)
    return source


class AbstractLoader(ABC):
    @abstractmethod
    def __init__(self, module_name, item_name, source=None):
        pass

    @abstractmethod
    def call_item(self):
        pass

    @abstractmethod
    def load_item(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class Loader(AbstractLoader):

    def __init__(self, module_name, item_name, source=None, **kwargs):

        self.item_name = item_name
        self.module_name = module_name
        self.source = source
        self.source_path = None

        if isinstance(source, str):
            self.module_name = module_name
            source_dict = load_source()
            if self.source in source_dict:
                self.source_path = source_dict[self.source]
            else:
                raise ValueError(
                    'source name {} is not found in the source.yaml file. Please check the source name.'.format(
                        self.source))
        elif source is None:
            self.module_name = module_name

        self.kwargs = kwargs

    def __call__(self):
        return self.call_item()

    def call_item(self):
        item = self._load_item()

        if item is not None and callable(item):
            item = item(**self.kwargs)

        return item

    def load_item(self):
        return self._load_item()

    def _load_item(self):
        if self.source_path is not None:
            sys.path.append(self.source_path)

        spec = importlib.util.find_spec(self.module_name)
        if spec is None:
            # Search for similar module names
            suggestion = self._find_similar_module_names()
            if suggestion:
                raise ValueError(
                    "Module: {} not found in the import path. Do you mean {}?".format(
                        self.module_name, suggestion))
            else:
                raise ValueError(
                    "Module: {} not found in the import path.".format(
                        self.module_name))

        module = importlib.import_module(self.module_name)

        item = getattr(module, self.item_name, None)
        if item is None:
            raise ValueError(
                "Item: {} not found in module: {}.".format(
                    self.item_name, self.module_name))

        if self.source_path is not None:
            sys.path.remove(self.source_path)

        return item

    def _find_similar_module_names(self):

        if self.source_path is None:
            return None
        files = os.listdir(self.source_path)
        print('source matches are', files)
        similar_names = difflib.get_close_matches(self.module_name, files)
        return similar_names[0] if similar_names else None

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            'module_name': self.module_name,
            'item_name': self.item_name,
            'kwargs': self.kwargs,
            'source': self.source
        }

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return Loader.from_dict(data)

    @staticmethod
    def from_dict(data_dict):
        return Loader(module_name=data_dict['module_name'],
                      item_name=data_dict['item_name'],
                      source=data_dict.get('source', None),
                      **data_dict.get('kwargs', {})
                      )


class ModelLoader(Loader):
    def __init__(self, module_name, item_name, source=None, **kwargs):
        if source is None:
            # add prefix for moduele loader
            module_name = f'{_Source.MODEL_ZOO}.{module_name}'
        super(
            ModelLoader,
            self).__init__(
            module_name,
            item_name,
            source,
            **kwargs)


class DatasetLoader(Loader):
    def __init__(self, module_name, item_name, source=None, **kwargs):
        if source is None:
            # add prefix for moduele loader
            module_name = f'{_Source.DATASET}.{module_name}'
        super(
            DatasetLoader,
            self).__init__(
            module_name,
            item_name,
            source,
            **kwargs)


class CustFuncLoader(Loader):
    def __init__(self, module_name, item_name, source=None, **kwargs):
        if source is None:
            # add prefix for moduele loader
            module_name = f'{_Source.CUST_FUNC}.{module_name}'
        super(
            CustFuncLoader,
            self).__init__(
            module_name,
            item_name,
            source,
            **kwargs)
