import importlib
import sys
from importlib.abc import MetaPathFinder, Loader

from importlib.machinery import ModuleSpec

flag = True


class ClientImportLoader(Loader):

    def module_repr(self, module):
        pass

    def create_module(self, spec):
        module = importlib.import_module("arch.api")
        return module

    @staticmethod
    def exec_module(module):
        session = importlib.import_module("compatible_hook._session")
        module.session = session


class ClientImportFinder(MetaPathFinder):

    @staticmethod
    def find_spec(full_name, paths=None, target=None):
        global flag
        if full_name == "arch.api" and flag:
            flag = False
            loader = ClientImportLoader()
            return ModuleSpec(full_name, loader, origin=paths)
        return None


finder = ClientImportFinder()
sys.meta_path.insert(0, finder)
