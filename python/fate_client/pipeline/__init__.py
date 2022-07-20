try:
    from pipeline.component.nn.backend.fate_torch.import_hook import fate_torch_hook
    from pipeline.component.nn.backend import fate_torch
except ImportError:
    fate_torch_hook, fate_torch = None, None
except ValueError:
    fate_torch_hook, fate_torch = None, None

__all__ = ['fate_torch_hook', 'fate_torch']
