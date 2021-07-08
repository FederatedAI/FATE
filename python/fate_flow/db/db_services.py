import abc
import atexit
from urllib import parse

from kazoo.client import KazooClient
from kazoo.security import make_digest_acl
from kazoo.exceptions import ZookeeperError, NodeExistsError, NoNodeError

from fate_arch.common.conf_utils import get_base_config
from fate_flow.settings import IP, HTTP_PORT, FATE_FLOW_MODEL_TRANSFER_ENDPOINT, \
    FATE_SERVICES_REGISTRY, stat_logger
from fate_flow.db.db_models import MachineLearningModelInfo as MLModel
from fate_flow.errors.error_services import *


def check_service_supported(method):
    """Decorator to check if `service_name` is supported.
    The attribute `supported_services` MUST be defined in class.
    The first and second arguments of `method` MUST be `self` and `service_name`.

    :param Callable method: The class method.
    :return: The inner wrapper function.
    :rtype: Callable
    """
    def magic(self, service_name, *args, **kwargs):
        if service_name not in self.supported_services:
            raise ServiceNotSupported(None, service_name)
        return method(self, service_name, *args, **kwargs)
    return magic


def get_model_download_endpoint():
    """Get the url endpoint of model download.
    `protocol`, `ip`, `port` and `endpoint` are defined on `conf/service_conf.yaml`.

    :return: The url endpoint.
    :rtype: str
    """

    return f'http://{IP}:{HTTP_PORT}{FATE_FLOW_MODEL_TRANSFER_ENDPOINT}'


def get_model_download_url(model_id, model_version):
    """Get the full url of model download.

    :param str model_id: The model id, `#` will be replaced with `_`.
    :param str model_version: The model version.
    :return: The download url.
    :rtype: str
    """
    return '{endpoint}/{model_id}/{model_version}'.format(
        endpoint=get_model_download_endpoint(),
        model_id=model_id.replace('#', '_'),
        model_version=model_version,
    )


class ServicesDB(abc.ABC):
    """Database for storage service urls.
    Abstract base class for the real backends.

    """
    @property
    @abc.abstractmethod
    def supported_services(self):
        """The names of supported services.
        The returned list SHOULD contain `fateflow` (model download) and `servings` (FATE-Serving).

        :return: The service names.
        :rtype: list
        """
        pass

    @abc.abstractmethod
    def _insert(self, service_name, service_url):
        pass

    @check_service_supported
    def insert(self, service_name, service_url):
        """Insert a service url to database.

        :param str service_name: The service name.
        :param str service_url: The service url.
        :return: None
        """
        try:
            self._insert(service_name, service_url)
        except ServicesError as e:
            stat_logger.exception(e)

    @abc.abstractmethod
    def _delete(self, service_name, service_url):
        pass

    @check_service_supported
    def delete(self, service_name, service_url):
        """Delete a service url from database.

        :param str service_name: The service name.
        :param str service_url: The service url.
        :return: None
        """
        try:
            self._delete(service_name, service_url)
        except ServicesError as e:
            stat_logger.exception(e)

    def register_model(self, model_id, model_version):
        """Call `self.insert` for insert a service url to database.
        Currently, only `fateflow` (model download) urls are supported.

        :param str model_id: The model id, `#` will be replaced with `_`.
        :param str model_version: The model version.
        :return: None
        """
        self.insert('fateflow', get_model_download_url(model_id, model_version))

    def unregister_model(self, model_id, model_version):
        """Call `self.delete` for delete a service url from database.
        Currently, only `fateflow` (model download) urls are supported.

        :param str model_id: The model id, `#` will be replaced with `_`.
        :param str model_version: The model version.
        :return: None
        """
        self.delete('fateflow', get_model_download_url(model_id, model_version))

    @abc.abstractmethod
    def _get_urls(self, service_name):
        pass

    @check_service_supported
    def get_urls(self, service_name):
        """Query service urls from database. The urls may belong to other nodes.
        Currently, only `fateflow` (model download) urls and `servings` (FATE-Serving) urls are supported.

        :param str service_name: The service name.
        :return: The service urls.
        :rtype: list
        """
        try:
            return self._get_urls(service_name)
        except ServicesError as e:
            stat_logger.exception(e)
            return []

    @property
    def models(self):
        return (MLModel.select(MLModel.f_model_id, MLModel.f_model_version).
                group_by(MLModel.f_model_id, MLModel.f_model_version))

    def register_models(self):
        """Register all service urls of each model to database on this node.

        :return: None
        """
        for model in self.models:
            self.register_model(model.f_model_id, model.f_model_version)

    def unregister_models(self):
        """Unregister all service urls of each model to database on this node.

        :return: None
        """
        for model in self.models:
            self.unregister_model(model.f_model_id, model.f_model_version)


class ZooKeeperDB(ServicesDB):
    """ZooKeeper Database

    """
    znodes = FATE_SERVICES_REGISTRY['zookeeper']
    supported_services = znodes.keys()

    def __init__(self):
        config = get_base_config('zookeeper')
        if not isinstance(config, dict) or not config:
            raise ZooKeeperNotConfigured()

        hosts = config.get('hosts')
        if not isinstance(hosts, list) or not hosts:
            raise ZooKeeperNotConfigured()

        client_kwargs = {'hosts': hosts}

        use_acl = config.get('use_acl', False)
        if use_acl:
            username = config.get('user')
            password = config.get('password')
            if not username or not password:
                raise MissingZooKeeperUsernameOrPassword()

            client_kwargs['default_acl'] = [make_digest_acl(username, password, all=True)]
            client_kwargs['auth_data'] = [('digest', ':'.join([username, password]))]

        try:
            # `KazooClient` is thread-safe, it contains `_thread.RLock` and can not be pickle.
            # So be careful when using `self.client` outside the class.
            self.client = KazooClient(**client_kwargs)
            self.client.start()
        except ZookeeperError as e:
            raise ZooKeeperBackendError(None, repr(e))

        atexit.register(self.client.stop)

    def _insert(self, service_name, service_url):
        try:
            self.client.create(self._get_znode_path(service_name, service_url), ephemeral=True, makepath=True)
        except NodeExistsError:
            pass
        except ZookeeperError as e:
            raise ZooKeeperBackendError(None, repr(e))

    def _delete(self, service_name, service_url):
        try:
            self.client.delete(self._get_znode_path(service_name, service_url))
        except NoNodeError:
            pass
        except ZookeeperError as e:
            raise ZooKeeperBackendError(None, repr(e))

    def _get_znode_path(self, service_name, service_url):
        """Get the znode path by service_name.

        :param str service_name: The service name.
        :param str service_url: The service url.
        :return: The znode path composed of `self.znodes[service_name]` and escaped `service_url`.
        :rtype: str

        :example:

        >>> self._get_znode_path('fateflow','http://127.0.0.1:9380/v1/model/transfer/arbiter-10000_guest-9999_host-10000_model/202105060929263278441')
        '/FATE-SERVICES/flow/online/transfer/providers/http%3A%2F%2F127.0.0.1%3A9380%2Fv1%2Fmodel%2Ftransfer%2Farbiter-10000_guest-9999_host-10000_model%2F202105060929263278441'
        """
        return '/'.join([self.znodes[service_name], parse.quote(service_url, safe='')])

    def _get_urls(self, service_name):
        try:
            urls = self.client.get_children(self.znodes[service_name])
        except ZookeeperError as e:
            raise ZooKeeperBackendError(None, repr(e))

        # remove prefix and unescape the url
        return [parse.unquote(url.rsplit('/', 1)[-1]) for url in urls]


class FallbackDB(ServicesDB):
    """Fallback Database.
       This class get the service url from `conf/service_conf.yaml`
       It cannot insert or delete the service url.

    """
    supported_services = ['fateflow', 'servings']

    def _insert(self, *args, **kwargs):
        pass

    def _delete(self, *args, **kwargs):
        pass

    def _get_urls(self, service_name):
        if service_name == 'fateflow':
            return [get_model_download_endpoint()]

        urls = get_base_config(service_name, [])
        if isinstance(urls, dict):
            urls = urls.get('hosts', [])
        if not isinstance(urls, list):
            return urls
        return [f'http://{url}' for url in urls]


def service_db():
    """Initialize services database.
    Currently only ZooKeeper is supported.

    :return ZooKeeperDB if `use_registry` is `True`, else FallbackDB.
            FallbackDB is a compatible class and it actually does nothing.
    """
    use_registry = get_base_config('use_registry', False)
    if not use_registry:
        return FallbackDB()
    if isinstance(use_registry, str):
        use_registry = use_registry.lower()
        if use_registry == 'zookeeper':
            return ZooKeeperDB()
    # backward compatibility
    return ZooKeeperDB()
