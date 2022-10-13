from .._base import device, _StorageOpsHandler


class _device_register:
    _device_handlers = {}

    @classmethod
    def register(cls, device: device, storage_ops_handler):
        cls._device_handlers[device] = storage_ops_handler

    @classmethod
    def get_device_ops_handler(cls, device: device) -> "_StorageOpsHandler":
        if device not in cls._device_handlers:
            raise NotImplementedError(f"lack of implemention for device {device}")
        return cls._device_handlers[device]
