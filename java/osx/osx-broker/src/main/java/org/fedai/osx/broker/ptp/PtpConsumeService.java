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

import com.google.common.base.Preconditions;
import io.grpc.ManagedChannel;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.api.context.Context;
import org.fedai.osx.api.router.RouterInfo;
import org.fedai.osx.broker.ServiceContainer;
import org.fedai.osx.broker.consumer.UnaryConsumer;
import org.fedai.osx.broker.queue.CreateQueueResult;
import org.fedai.osx.broker.queue.TransferQueue;
import org.fedai.osx.broker.queue.TransferQueueApplyInfo;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.context.FateContext;
import org.fedai.osx.core.exceptions.ParameterException;
import org.fedai.osx.core.exceptions.TransferQueueNotExistException;
import org.fedai.osx.core.frame.GrpcConnectionFactory;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;
import org.ppc.ptp.PrivateTransferProtocolGrpc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PtpConsumeService extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpConsumeService.class);
    public PtpConsumeService() {
        this.setServiceName("consume-unary");
    }
    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {

        context.setActionType(ActionType.DEFUALT_CONSUME.getAlias());
        Osx.Inbound inbound = data.getBody();
        String topic = context.getTopic();
            TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(topic);
            if (transferQueue == null) {
                if (MetaInfo.isCluster()) {
                    TransferQueueApplyInfo transferQueueApplyInfo = ServiceContainer.transferQueueManager.queryGlobleQueue(topic);
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
                        return redirect(context, routerInfo, inbound);
                    }
                } else {
                    /**
                     * 单机版直接创建队列
                     */
                    logger.warn("create topic {} by consume request ", topic);
                    CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(topic, context.getSessionId(), true);
                    if (createQueueResult.getTransferQueue() == null) {
                        throw new TransferQueueNotExistException();
                    }
                }
            }
            StreamObserver streamObserver = (StreamObserver) context.getData(Dict.RESPONSE_STREAM_OBSERVER);
            Long offset = (Long) context.getData(Dict.REQUEST_INDEX);
            Preconditions.checkArgument(offset != null);
            if (offset == null) {
                throw new ParameterException("offset is null");
            }
            if (offset > 0) {
                context.setActionType(ActionType.CUSTOMER_CONSUME.getAlias());
            }
            UnaryConsumer consumer = ServiceContainer.consumerManager.getOrCreateUnaryConsumer(topic);
            TransferQueue.TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(context, offset);
            context.setReturnCode(transferQueueConsumeResult.getCode());
            if (transferQueueConsumeResult.getCode().equals(StatusCode.CONSUME_NO_MESSAGE)) {
                // 由其他扫描线程应答
                if (offset < 0) {

                    UnaryConsumer.LongPullingHold longPullingHold = new UnaryConsumer.LongPullingHold();

                    longPullingHold.setGrpcContext(io.grpc.Context.current());
                    longPullingHold.setNeedOffset(offset);
                    longPullingHold.setStreamObserver(streamObserver);
                    longPullingHold.setContext(context.subContext());
                    String timeOutString = inbound.getMetadataMap().get(Osx.Metadata.Timeout.name());
                    if (StringUtils.isNotEmpty(timeOutString)) {
                        long current = System.currentTimeMillis();
                        longPullingHold.setExpireTimestamp(current + Long.valueOf(timeOutString));
                    }
                    consumer.addLongPullingQueue(longPullingHold);
                    return null;
                }
            }
            Osx.Outbound consumeResponse = TransferUtil.buildResponse(transferQueueConsumeResult.getCode(), "", transferQueueConsumeResult);
            return consumeResponse;

    }
    private Osx.Outbound redirect(Context context, RouterInfo routerInfo, Osx.Inbound inbound) {
        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
        context.setActionType(ActionType.REDIRECT_CONSUME.getAlias());
        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
        return stub.invoke(inbound);
    }
}
