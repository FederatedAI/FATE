
from typing import Union

from arch.api.base.federation import Federation, Party
from arch.api.utils import file_utils
from arch.api.utils import log_utils
from arch.api.impl.based_spark.based_hdfs.rabbit_manager import RabbitManager
from arch.api.impl.based_spark.based_hdfs.table import RDDTable
from functools import partial
import pika
from pickle import dumps as p_dumps, loads as p_loads
import json

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
        self._host = self._mq_conf.get("self").get("host")
        self._port = self._mq_conf.get("self").get("port")
        base_user = self._mq_conf.get("self").get("user")
        base_password = self._mq_conf.get("self").get("password")

        self._rabbit_manager = RabbitManager(base_user, base_password, "{}:{}".format(self._host, self._port))

        mq_info = runtime_conf.get("mq_info", {})
        self._user = mq_info.get("user")
        self._password = mq_info.get("pswd")

        # initial user
        self._rabbit_manager.CreateUser(self._user, self._password)

        self._queque_map = {}


    def get_mq_names(self, parties: Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]
        party_ids = [party.party_id for party in parties]
        mq_names = {}
        for role, party_id in party_ids:
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
            mq_names[party_id] = names

        return mq_names

    
    @staticmethod
    def _get_channels(mq_names, host, port, user, password):
        channel_infos = []
        for mq in mq_names:
            credentials = pika.PlainCredentials(user, password)
            connection = pika.BlockingConnection(pika.ConnectionParameters(host, port, mq["vhost"], credentials))
            info= {}
            info["channel"] = connection.channel()
            info["send"] = mq["send"]
            channel_infos.append(info)
        return channel_infos


    @staticmethod
    def _send_kv(name, tag, data, channel_infos):
        properties=pika.BasicProperties(
            content_type='application/json',
            name=name,
            tag=tag,
            dtype='kv'
        )
        for info in channel_infos:
            info["channel"].basic_publish(exchange='', routing_key=info["send"], body=json.dumps(data), properties=properties)

    
    @staticmethod
    def _send_obj(name, tag, data, channel_infos):
        properties=pika.BasicProperties(
            content_type='text/plain',
            name=name,
            tag=tag,
            dtype='obj'
        )
        for info in channel_infos:
            info["channel"].basic_publish(exchange='', routing_key=info["send"], body=data, properties=properties)



    @staticmethod
    def _partition_send(kvs, mq_names, host, port, user, password, name, tag):
        channel_infos = _get_channels(mq_names=mq_names, host=host, port=port, user=user, password=password)
        data = []
        count = 0
        MESSAGE_MAX_SIZE = 200000
        for k, v in kvs:
            el = {}
            el['k'] = p_dumps(k)
            el['v'] = p_dumps(v)
            data.append(el)
            count = count + 1
            if count > MESSAGE_MAX_SIZE:
                _send_kv(name=name, tag=tag, data=data, channel_infos=channel_infos)
                count = 0
                data.clear()
        _send_kv(name=name, tag=tag, data=data, channel_infos=channel_infos)
        return data


    def get(self, name, tag, parties: Union[Party, list]):
        pass


    def remote(self, obj, name, tag, parties: Union[Party, list]):
        mq_names = self.get_mq_names(parties)

        if isinstance(obj, RDDTable):
            send_func = partial(self._partition_send, mq_names=mq_names, 
                        host=self._host, port=self._port, 
                        user=self._user, password=self._password,
                        name=name, tag=tag)
            obj.mapPartitions(send_func)
        else:
            channel_infos = _get_channels(mq_names=mq_names, host=self._host, port=self._port, 
                                            user=self._user, password=self._password)
            _send_obj(name=name, tag=tag, data=p_dumps(obj), channel_infos=channel_infos)
            



