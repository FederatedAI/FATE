
from typing import Union

from arch.api.base.federation import Federation
from arch.api.utils import file_utils
from arch.api.utils import log_utils
from arch.api.impl.based_spark.based_hdfs.rabbit_manager import RabbitManager

LOGGER = log_utils.getLogger()



class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)
        self._role = runtime_conf.get("local").get("role")
        self._party_id = str(runtime_conf.get("local").get("party_id"))
        self._session_id = session_id
        #init mq here
        init_mq_context(runtime_conf)


    def init_mq_context(self, runtime_conf):
        _path = file_utils.get_project_base_directory() + "/arch/conf/server_conf.json"
        server_conf = file_utils.load_json_conf(_path)
        self._mq_conf = server_conf.get('servers').get('rabbitmq')
        host = self._mq_conf.get("self").get("host")
        port = self._mq_conf.get("self").get("port")
        base_user = self._mq_conf.get("self").get("user")
        base_password = self._mq_conf.get("self").get("password")

        self._rabbit_manager = RabbitManager(base_user, base_password, "{}:{}".format(host, port))

        mq_info = runtime_conf.get("mq_info", {})
        self._user = mq_info.get("user")
        self._password = mq_info.get("pswd")

        # initial user
        self._rabbit_manager.CreateUser(self._user, self._password)

        self._queque_map = {}


    def get_mq_names(self, party_id):
        names = self._queque_map.get(party_id)
        if names is None:
            names = {}

            # initial vhost
            vhost = "{}-{}".format(self._session_id, party_id)
            self._rabbit_manager.CreateVhost(vhost)
            self._rabbit_manager.AddUserToVhost(self._user, vhost)
            names["vhost"] = vhost

            # initial send queue, the name is send-${vhost}
            send_queue_name = "{}-{}".format("send", vhost)
            self._rabbit_manager.CreateQueue(vhost, send_queue_name)
            names["send"] = send_queue_name

            # initial receive queue, the name is receive-${vhost}
            receive_queue_name = "{}-{}".format("receive", vhost)
            self._rabbit_manager.CreateQueue(vhost, receive_queue_name)
            names["receive"] = receive_queue_name

            host = self._mq_conf.get(party_id).get("host")
            port = self._mq_conf.get(party_id).get("port")
            
            upstream_uri = "amqp://{}:{}@{}:{}".format(self._user, self._password, host, port)
            union_name = self._rabbit_manager.FederateQueue(upstream_uri , vhost, receive_queue_name)
            names["union"] = union_name

            self._queque_map[party_id] = names

        return names


    def get(self, name, tag, parties: Union[Party, list]):
        pass


    def remote(self, obj, name, tag, parties):
        pass
