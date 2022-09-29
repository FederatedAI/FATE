## FATE通信接口

```python
class MQChannel(object):
    def __init__(self,
                 host,
                 port,
                 namespace,
                 send_topic,
                 receive_topic,
                 src_party_id,
                 src_role,
                 dst_party_id,
                 dst_role
                 ):
        self._host = host
        self._port = port
        self._namespace = namespace
        self._send_topic = send_topic
        self._receive_topic = receive_topic
        self._src_party_id = src_party_id
        self._src_role = src_role
        self._dst_party_id = dst_party_id
        self._dst_role = dst_role
        self._channel = None
        self._stub = None

    @nretry
    def consume(self, start_offset=-1):
        self._get_or_create_channel()
        response = self._stub.consumeUnary(
            firework_transfer_pb2.ConsumeRequest(transferId=self._receive_topic, startOffset=start_offset,
                                                 sessionId=self._namespace))
        return response

    @nretry
    def cleanup(self):
        self._get_or_create_channel()
        response = self._stub.cancelTransfer(
            firework_transfer_pb2.CancelTransferRequest(transferId=self._receive_topic, sessionId=self._namespace))
        return response

    @nretry
    def query(self):
        self._get_or_create_channel()
        LOGGER.debug(f"try to query {self._receive_topic} session {self._namespace}")
        response = self._stub.queryTransferQueueInfo(
            firework_transfer_pb2.QueryTransferQueueInfoRequest(transferId=self._receive_topic,
                                                                sessionId=self._namespace))
        return response

    @nretry
    def produce(self, body, properties, is_over=True):
        self._get_or_create_channel()
        packet = firework_transfer_pb2.ProduceRequest(transferId=self._send_topic, sessionId=self._namespace,
                                                      routeInfo=firework_transfer_pb2.RouteInfo(
                                                            srcPartyId=self._src_party_id,
                                                          srcRole=self._src_role,
                                                          desPartyId=self._dst_party_id,
                                                          desRole=self._dst_role),
                                                      message=firework_transfer_pb2.Message(
                                                          head=bytes(json.dumps(properties), encoding="utf-8"),
                                                          body=body), isOver=is_over)
        result = self._stub.produceUnary(packet)
        return result

    @nretry
    def ack(self, start_offset=1):
        self._get_or_create_channel()
        result = self._stub.ack(
            firework_transfer_pb2.AckRequest(transferId=self._receive_topic, startOffset=start_offset,
                                             sessionId=self._namespace))
        return result

    def close(self):
        try:
            if self._channel:
                self._channel.close()
            self._channel = None
            self._stub = None
        except Exception as e:
            LOGGER.exception(e)
            self._stub = None
            self._channel = None

    def cancel(self):
        self.close()

    def _get_or_create_channel(self):
        target = '{}:{}'.format(self._host, self._port)
        if self._check_alive():
            return

        self._channel = grpc.insecure_channel(
            target=target,
            options=[('grpc.max_send_message_length',
                      int((2 << 30) - 1)),
                     ('grpc.max_receive_message_length',
                      int((2 << 30) - 1)),
                     ('grpc.max_metadata_size',
                      128 << 20),
                     ('grpc.keepalive_time_ms',
                      7200 * 1000),
                     ('grpc.keepalive_timeout_ms',
                      3600 * 1000),
                     ('grpc.keepalive_permit_without_calls',
                      int(False)),
                     ('grpc.per_rpc_retry_buffer_size',
                      int(16 << 20)),
                     ('grpc.enable_retries', 1),
                     ('grpc.service_config',
                      '{ "retryPolicy":{ '
                      '"maxAttempts": 4, "initialBackoff": "0.1s", '
                      '"maxBackoff": "1s", "backoffMutiplier": 2, '
                      '"retryableStatusCodes": [ "UNAVAILABLE" ] } }')])

        self._stub = FireworkQueueServiceStub(self._channel)

    def _check_alive(self):
        status = grpc._common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY[
            self._channel._channel.check_connectivity_state(True)] if self._channel is not None else None

        if status == grpc.ChannelConnectivity.SHUTDOWN:
            return True
        else:
            return False

```



##### 1. 发送消息

- ```produce(self, body, properties, is_over=True)```

##### 2. 接收消息

- ```consume(self, start_offset=-1)```

##### 3. 应答

- ```ack(self, start_offset=1)```

##### 4. 查询

- ``` query(self)```

##### 5. 停止

- ```cancel(self```)
- ```stop(self)```

