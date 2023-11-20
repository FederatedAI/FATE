///*
// * Copyright 2019 The FATE Authors. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//package org.fedai.osx.broker.ptp;
//
//import com.google.common.base.Preconditions;
//import com.google.inject.Inject;
//import com.google.inject.Singleton;
//import io.grpc.ManagedChannel;
//import io.grpc.stub.StreamObserver;
//import org.apache.commons.lang3.StringUtils;
//
//import org.fedai.osx.broker.consumer.ConsumerManager;
//import org.fedai.osx.broker.consumer.UnaryConsumer;
//import org.fedai.osx.broker.queue.*;
//import org.fedai.osx.broker.service.Register;
//import org.fedai.osx.broker.util.TransferUtil;
//import org.fedai.osx.core.config.MetaInfo;
//import org.fedai.osx.core.constant.ActionType;
//import org.fedai.osx.core.constant.Dict;
//import org.fedai.osx.core.constant.StatusCode;
//import org.fedai.osx.core.constant.UriConstants;
//import org.fedai.osx.core.context.OsxContext;
//import org.fedai.osx.core.exceptions.ExceptionInfo;
//import org.fedai.osx.core.exceptions.ParameterException;
//import org.fedai.osx.core.exceptions.TransferQueueNotExistException;
//import org.fedai.osx.core.frame.GrpcConnectionFactory;
//import org.fedai.osx.core.router.RouterInfo;
//import org.fedai.osx.core.service.InboundPackage;
//import org.ppc.ptp.Osx;
//import org.ppc.ptp.PrivateTransferProtocolGrpc;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
// @Singleton
//// @Register(uri= UriConstants.PEEK,allowInterUse=false)
// public class PeekConsumeService extends AbstractPtpServiceAdaptor<Osx.PeekInbound,Osx.TransportOutbound> {
//
//    Logger logger = LoggerFactory.getLogger(PeekConsumeService.class);
//    @Inject
//    public PeekConsumeService() {
//        this.setServiceName("peek");
//    }
//
//     @Override
//     protected Osx.TransportOutbound transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
//        Osx.TransportOutbound.Builder builder = Osx.TransportOutbound.newBuilder();
//        builder.setCode(exceptionInfo.getCode());
//        builder.setMessage(exceptionInfo.getMessage());
//        return builder.build();
//     }
//     @Inject
//    TransferQueueManager  transferQueueManager;
//    @Inject
//    ConsumerManager  consumerManager;
//
//    @Override
//    protected Osx.TransportOutbound doService(OsxContext context, InboundPackage<Osx.PeekInbound> data) {
//
//        context.setActionType(ActionType.DEFUALT_CONSUME.getAlias());
//        Osx.PeekInbound inbound = data.getBody();
//        String topic = inbound.getTopic();
//        context.setTopic(topic);
//        AbstractQueue transferQueue = transferQueueManager.getQueue(topic);
//        if (transferQueue == null) {
//                if (MetaInfo.isCluster()) {
//                    TransferQueueApplyInfo transferQueueApplyInfo = transferQueueManager.queryGlobleQueue(topic);
//                    if (transferQueueApplyInfo == null) {
//                        throw new TransferQueueNotExistException("topic  "+topic+" not found" );
////                        CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(context.getSessionId(),topic, false);
////                        if (createQueueResult.getTransferQueue() == null) {
////                            //重定向
////                            Osx.TopicInfo topicInfo = Osx.TopicInfo.newBuilder()
////                                    .setTopic(topic)
////                                    .setCreateTimestamp(System.currentTimeMillis())
////                                    .setIp(createQueueResult.getRedirectIp())
////                                    .setPort(createQueueResult.getPort())
////                                    .build();
////                            return TransferUtil.buildResponseInner(StatusCode.TRANSFER_QUEUE_REDIRECT,"NEED REDIRECT",topicInfo.toByteArray()).build();
////                        }
//                    } else {
//                        String[] args = transferQueueApplyInfo.getInstanceId().split(":");
//                        String ip = args[0];
//                        int port = Integer.parseInt(args[1]);
//                        RouterInfo routerInfo = new RouterInfo();
//                        routerInfo.setHost(ip);
//                        routerInfo.setPort(port);
//                        return redirect(context, routerInfo, inbound);
//                    }
//                } else {
//
//                }
//            }
//
//            UnaryConsumer consumer = consumerManager.getOrCreateUnaryConsumer(topic);
//            TransferQueueConsumeResult transferQueueConsumeResult = consumer.consume(context, -1);
//            context.setReturnCode(transferQueueConsumeResult.getCode());
//
//            Osx.TransportOutbound consumeResponse = TransferUtil.buildTransportOutbound(transferQueueConsumeResult.getCode(), "", transferQueueConsumeResult);
//            return consumeResponse;
//
//    }
//    private Osx.TransportOutbound redirect(OsxContext context, RouterInfo routerInfo, Osx.PeekInbound inbound) {
//        ManagedChannel managedChannel = GrpcConnectionFactory.createManagedChannel(routerInfo,true);
//        context.setActionType(ActionType.REDIRECT_CONSUME.getAlias());
//        PrivateTransferProtocolGrpc.PrivateTransferProtocolBlockingStub stub = PrivateTransferProtocolGrpc.newBlockingStub(managedChannel);
//        return stub.peek(inbound);
//    }
//}
