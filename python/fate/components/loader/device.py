def load_device(device_spec):
    from fate.arch.unify import device
    from fate.components.spec.device import CPUSpec, GPUSpec

    if isinstance(device_spec, CPUSpec):
        return device.CPU

    if isinstance(device_spec, GPUSpec):
        return device.CUDA
    raise ValueError(f"device `{device_spec}` not implemeted yet")
