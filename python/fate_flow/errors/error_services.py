from fate_flow.errors import FateFlowError

__all__ = ['ServicesError', 'ServiceNotSupported', 'ZooKeeperNotConfigured',
           'MissingZooKeeperUsernameOrPassword', 'ZooKeeperBackendError']


class ServicesError(FateFlowError):
    message = 'Unknown services error'


class ServiceNotSupported(ServicesError):
    message = 'The service {} is not supported'


class ZooKeeperNotConfigured(ServicesError):
    message = 'ZooKeeper has not been configured'


class MissingZooKeeperUsernameOrPassword(FateFlowError):
    message = 'Using ACL for ZooKeeper is enabled but username or password is not configured'


class ZooKeeperBackendError(ServicesError):
    message = 'ZooKeeper backend error: {}'
