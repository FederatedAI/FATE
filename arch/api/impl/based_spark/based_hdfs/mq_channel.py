import pika
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()
 
class MQChannel(object):
 
    def __init__(self, host, port, user, password, party_id, vhost, send_queue_name, receive_queue_name):
        self._host = host
        self._port = port
        self._credentials = pika.PlainCredentials(user, password)
        self._party_id = party_id
        self._vhost = vhost
        self._send_queue_name = send_queue_name
        self._receive_queue_name = receive_queue_name
        self._conn = None
        self._channel = None

    
    def basic_publish(self, body, properties):
        try:
            self._get_channel()
            return self._channel.basic_publish(exchange='', routing_key=self._send_queue_name, body=body, properties=properties)
        except (pika.exceptions.ConnectionClosed, pika.exceptions.StreamLostError) as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.info("Trying to reconnect...")
            self._get_channel()
            return self._channel.basic_publish(exchange='', routing_key=self._send_queue_name, body=body, properties=properties)

 
    def consume(self):
        try:
            self._get_channel()
            return self._channel.consume(queue=self._receive_queue_name)
        except (pika.exceptions.ConnectionClosed, pika.exceptions.StreamLostError) as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.info("Trying to reconnect...")
            self._get_channel()
            return self._channel.consume(queue=self._receive_queue_name)


    def basic_ack(self, delivery_tag):
        try:
            self._get_channel()
            return self._channel.basic_ack(delivery_tag=delivery_tag)
        except (pika.exceptions.ConnectionClosed, pika.exceptions.StreamLostError) as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.info("Trying to reconnect...")
            self._get_channel()
            return self._channel.basic_ack(delivery_tag=delivery_tag)


    def cancel(self):
        try:
            self._get_channel()
            return self._channel.cancel()
        except (pika.exceptions.ConnectionClosed, pika.exceptions.StreamLostError) as e:
            LOGGER.error("Lost connection to rabbitmq service on manager, exception:{}.".format(e))
            self._clear()
            LOGGER.info("Trying to reconnect...")
            self._get_channel()
            return self._channel.cancel()


    def _get_channel(self):
        if self._check_alive():
            return
        else:
            self._clear()
 
        if not self._conn:
            self._conn = pika.BlockingConnection(pika.ConnectionParameters(host=self._host, port=self._port, 
                                            virtual_host=self._vhost, credentials=self._credentials))
        if not self._channel:
            self._channel = self._conn.channel()
    

    def _clear(self):
        def clear_conn():
            if self._conn and self._conn.is_open:
                self._conn.close()
            self._conn = None
 
        def clear_channel():
            if self._channel and self._channel.is_open:
                self._channel.close()
            self._channel = None
 
        if not (self._conn and self._conn.is_open):
            clear_conn()
        clear_channel()


    def _check_alive(self):
        return self._channel and self._channel.is_open and self._conn and self._conn.is_open


    
 
 
