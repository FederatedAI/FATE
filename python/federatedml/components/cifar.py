from .components import ComponentMeta

cifar = ComponentMeta("Cifar")


@cifar.bind_param
def cifar_param():
    from federatedml.cifar.param import CifarParam

    return CifarParam


@cifar.bind_runner.on_guest.on_host
def cifar_runner():
    from federatedml.cifar.train import CifarTrainer

    return CifarTrainer
