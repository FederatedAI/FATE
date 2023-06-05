import sys
import importlib.util
from abc import ABC, abstractmethod
from enum import Enum
import json


class Source(object):
    MODEL_ZOO = 'fate.ml.nn.model_zoo'
    DATASET = 'fate.ml.nn.dataset'
    CUST_FUNC = 'fate.ml.nn.cust_func'


MODULE_PATH = set(
    ['fate.ml.nn.model_zoo',
    'fate.ml.nn.dataset',
    'fate.ml.nn.cust_func']
)


class AbstractLoader(ABC):
    @abstractmethod
    def __init__(self, module_name, item_name, source=None):
        pass

    @abstractmethod
    def load_inst(self):
        pass

    @abstractmethod
    def load_class(self):
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

        if isinstance(source, str) and source in MODULE_PATH:
            self.module_name = f'{source}.{module_name}'
            self.source = None
        elif isinstance(source, str):
            self.module_name = module_name
            self.source = source
        elif source is None:
            self.module_name = module_name
            self.source = None
        else:
            raise TypeError("The 'source' parameter must be either a string or an instance of the 'Source' enum.")

        self.kwargs = kwargs

    def load_inst(self):
        item = self._load_item()

        if item is not None and callable(item):
            item = item(**self.kwargs)

        return item

    def load_class(self):
        return self._load_item()

    def _load_item(self):
        if self.source is not None:
            sys.path.append(self.source)

        spec = importlib.util.find_spec(self.module_name)
        if spec is None:
            print("Module: {} not found.".format(self.module_name))
            return None

        module = importlib.import_module(self.module_name)

        item = getattr(module, self.item_name, None)
        if item is None:
            print("Item: {} not found in module: {}.".format(self.item_name, self.module_name))

        if self.source is not None:
            sys.path.remove(self.source)

        return item

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        return {
            'module_name': self.module_name,
            'item_name': self.item_name,
            'source': self.source if self.source else self.module_name.split('.')[0],
            'kwargs': self.kwargs
        }

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return Loader.from_dict(data)

    @staticmethod
    def from_dict(data_dict):
        return Loader(module_name=data_dict['module_name'], 
                      item_name=data_dict['item_name'], 
                      source=data_dict.get('source'),
                      **data_dict.get('kwargs', {})
                      )
