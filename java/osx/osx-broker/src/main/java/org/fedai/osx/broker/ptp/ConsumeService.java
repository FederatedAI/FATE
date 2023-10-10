/*
 * Copyright 2019 The FATE Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.fedai.osx.broker.ptp;

import com.google.inject.Inject;
import com.google.inject.Singleton;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.fedai.osx.broker.consumer.ConsumerManager;
import org.fedai.osx.broker.consumer.UnaryConsumer;
import org.fedai.osx.broker.pojo.ConsumeRequest;
import org.fedai.osx.broker.pojo.ConsumerResponse;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.TransferQueueNotExistException;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.router.RouterInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
 @Singleton
 @Register(uris= {UriConstants.POP,UriConstants.PEEK},allowInterUse = false)
 public class ConsumeService extends AbstractServiceAdaptorNew< ConsumeRequest, ConsumerResponse> {

    Logger logger = LoggerFactory.getLogger(ConsumeService.class);
    @Inject
    public ConsumeService() {
        this.setServiceName("pop");
    }
    @Inject
    TransferQueueManager  transferQueueManager;
    @Inject
    ConsumerManager  consumerManager;

    @Override
    protected ConsumerResponse doService(OsxContext context, ConsumeRequest inbound) {

        context.setActionType(ActionType.DEFUALT_CONSUME.name());
        String topic = inbound.getTopic();
        int timeout =  inbound.getTimeout()>0?inbound.getTimeout():MetaInfo.CONSUME_MSG_WAITING_TIMEOUT;
        context.setTopic(topic);
        AbstractQueue transferQueue = transferQueueManager.getQueue(topic);
            if (transferQueue == null) {
                if (MetaInfo.isCluster()) {
                    TransferQueueApplyInfo transferQueueApplyInfo = transferQueueManager.queryGlobleQueue(topic);
                    if (transferQueueApplyInfo == null) {
                        throw new TransferQueueNotExistException("topic  "+topic+" not found" );
//                        CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(topic, context.getSessionId(), false);
//                        if (createQueueResult.getTransferQueue() == null) {
//                            //重定向
//                            Osx.TopicInfo topicInfo = Osx.TopicInfo.newBuilder()
//                                    .setTopic(topic)
//                                    .setCreateTimestamp(System.currentTimeMillis())
//                                    .setIp(createQueueResult.getRedirectIp())
//                                    .setPort(createQueueResult.getPort())
//                                    .build();
//                            return TransferUtil.buildResponseInner(StatusCode.TRANSFER_QUEUE_REDIRECT,"NEED REDIRECT",topicInfo.toByteArray()).build();
//                        }
                    } else {
                        String[] args = transferQueueApplyInfo.getInstanceId().split(":");
                        String ip = args[0];
                        int port = Integer.parseInt(args[1]);
                        RouterInfo routerInfo = new RouterInfo();
                        routerInfo.setHost(ip);
                        routerInfo.setPort(port);
                        ConsumerResponse consumerResponse = new ConsumerResponse();
                        consumerResponse.setNeedRedirect(true);
                        consumerResponse.setRedirectRouterInfo(routerInfo);
                        return consumerResponse;

                    }
                } else {
                    /**
                     * 单机版直接创建队列
                     */
                    logger.warn("create topic {} by consume request ", topic);
                    CreateQueueResult createQueueResult = transferQueueManager.createNewQueue(topic, context.getSessionId(), true, QueueType.NORMAL);
                    if (createQueueResult.getQueue() == null) {
                        throw new TransferQueueNotExistException();
                    }
                }
            }

            UnaryConsumer consumer = consumerManager.getOrCreateUnaryConsumer(topic);
            TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(context, -1);
        transferQueueConsumeResult.getLogicIndexTotal();
            context.setReturnCode(transferQueueConsumeResult.getCode());
            if (transferQueueConsumeResult.getCode().equals(StatusCode.CONSUME_NO_MESSAGE)) {
                // 由其他扫描线程应答
                if (inbound.isNeedBlock()) {
                    StreamObserver streamObserver = (StreamObserver) context.getData(Dict.RESPONSE_STREAM_OBSERVER);
                    UnaryConsumer.LongPullingHold longPullingHold = new UnaryConsumer.LongPullingHold();
                    longPullingHold.setGrpcContext(io.grpc.Context.current());
                    longPullingHold.setNeedOffset(-1);
                    longPullingHold.setStreamObserver(streamObserver);
                    longPullingHold.setContext(context.subContext());
                    long current = System.currentTimeMillis();
                    longPullingHold.setExpireTimestamp(current + Long.valueOf(timeout));
                    consumer.addLongPullingQueue(longPullingHold);
                    logger.info("add long pulling {} {}",longPullingHold,timeout);
                    return null;
                }
            }
            ConsumerResponse consumeResponse  = new ConsumerResponse();
            consumeResponse.setCode(StatusCode.PTP_SUCCESS);
            consumeResponse.setPayload(transferQueueConsumeResult.getMessage().getBody());
            return   consumeResponse;


    }

     @Override
     protected ConsumerResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        ConsumerResponse consumerResponse =  new ConsumerResponse();
        consumerResponse.setCode(exceptionInfo.getCode());
        consumerResponse.setMsg(exceptionInfo.getMessage());
        return consumerResponse;
     }

     private Osx.TransportOutbound redirect(OsxContext context, RouterInfo routerInfo, Osx.PopInbound inbound) {
        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        context.setActionType(ActionType.REDIRECT_CONSUME.name());
        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        return stub.pop(inbound);
    }


     @Override
     public ConsumeRequest decode(Object object) {
         return null;
     }

     @Override
     public Osx.Outbound toOutbound(ConsumerResponse response) {
         return null;
     }
 }
