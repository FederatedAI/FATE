import sys
import importlib.util
import json


class Loader:
    def __init__(self, module_name, item_name, path=None, **kwargs):
        """Create a new Loader to import an item (class or function) from a module.

        Args:
            module_name (str): The name of the module.
            item_name (str): The name of the item (class or function).
            path (str, optional): The path where the module can be found. Defaults to None.
            **kwargs: The arguments to be used for initializing the imported item.
        """
        self.module_name = module_name
        self.item_name = item_name
        self.path = path
        self.kwargs = kwargs

        # Check if kwargs are JSON serializable
        try:
            json.dumps(self.kwargs)
        except TypeError as e:
            raise TypeError("The kwargs are not JSON serializable, please make sure that parameters of the item can be jsonized: " + str(e))

    def load_inst(self):
        """Load the item (class or function) from the module and initialize it.

        Returns:
            The initialized item (class or function), or None if it could not be found.
        """
        # Load the item (class or function) from the module
        item = self._load_item()
        
        # Initialize the item
        if item is not None and callable(item):
            item = item(**self.kwargs)

        return item

    def load_class(self):
        """Load the item (class or function) from the module without initializing it.

        Returns:
            The loaded item (class or function), or None if it could not be found.
        """
        return self._load_item()

    def _load_item(self):
        """Load the item (class or function) from the module.

        Returns:
            The loaded item (class or function), or None if it could not be found.
        """
        # Add the path to sys.path if it was provided
        if self.path is not None:
            sys.path.append(self.path)

        # Use importlib to load the module
        spec = importlib.util.find_spec(self.module_name)
        if spec is None:
            print("Module: {} not found.".format(self.module_name))
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the item (class or function) from the module
        item = getattr(module, self.item_name, None)
        if item is None:
            print("Item: {} not found in module: {}.".format(self.item_name, self.module_name))

        # Remove the path from sys.path if it was added
        if self.path is not None:
            sys.path.remove(self.path)

        return item

    def to_json(self):
        """Convert the Loader's parameters to a JSON string.

        Returns:
            A JSON string representation of the Loader's parameters.
        """
        return json.dumps(self.to_dict())
    
    def to_dict(self):
        """Convert the Loader's parameters to a dict.

        Returns:
            A dict representation of the Loader's parameters.
        """
        return {
            'module_name': self.module_name,
            'item_name': self.item_name,
            'path': self.path,
            'params': self.kwargs
        }