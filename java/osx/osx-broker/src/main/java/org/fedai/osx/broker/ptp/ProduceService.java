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
import com.google.protobuf.InvalidProtocolBufferException;
import com.webank.ai.eggroll.api.networking.proxy.Proxy;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import org.apache.commons.lang3.StringUtils;

import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.grpc.QueuePushReqStreamObserver;
import org.fedai.osx.broker.interceptor.RouterInterceptor;
import org.fedai.osx.broker.message.MessageDecoder;
import org.fedai.osx.broker.message.MessageExtBrokerInner;
import org.fedai.osx.broker.pojo.ProduceRequest;
import org.fedai.osx.broker.pojo.ProduceResponse;
import org.fedai.osx.broker.pojo.PushRequest;
import org.fedai.osx.broker.pojo.PushResponse;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.broker.router.DefaultFateRouterServiceImpl;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.broker.service.TokenApplyService;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.constant.*;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.*;
import org.fedai.osx.core.flow.FlowCounterManager;
import org.fedai.osx.core.ptp.TargetMethod;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.service.Interceptor;
import org.fedai.osx.core.service.OutboundPackage;
import org.fedai.osx.core.utils.FlowLogUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;

import static org.fedai.osx.core.constant.UriConstants.PUSH;

@Singleton
@Register(uris={PUSH})
@Data
public class ProduceService extends AbstractServiceAdaptorNew< ProduceRequest, ProduceResponse> {


    Logger logger = LoggerFactory.getLogger(ProduceService.class);
    @Inject
    TransferQueueManager transferQueueManager;
    @Inject
    TokenApplyService tokenApplyService;
    @Inject
    FlowCounterManager flowCounterManager;
    @Inject
    DefaultFateRouterServiceImpl  defaultFateRouterService;

//    @Inject
//    public PushService(RouterInterceptor routerInterceptor) {
//        this.addPreProcessor(routerInterceptor);
//        this.addPreProcessor(new Interceptor<OsxContext, Object, Object>() {
//            @Override
//            public void doProcess(OsxContext context, PushRequest inboundPackage, OutboundPackage outboundPackage) {
//                System.err.println("xxxxxxxxxxxxxxxx");
//                if(inboundPackage.getBody() instanceof  Osx.Inbound){
//                    System.err.println("yyyyyyyyyyyyyyy");
//                    Osx.Inbound  inbound = (Osx.Inbound)inboundPackage.getBody();
//                    try {
//                        Osx.PushInbound pushInbound =   Osx.PushInbound.parseFrom(inbound.getPayload());
//                        inboundPackage.setBody(pushInbound);
//                    } catch (InvalidProtocolBufferException e) {
//                        e.printStackTrace();
//                    }
//                }
//            }
//        });
////        this.addPostProcessor(new Interceptor<OsxContext, Osx.Inbound, Osx.Outbound>() {
////            @Override
////            public void doProcess(OsxContext context, InboundPackage<Osx.Inbound> inboundPackage, OutboundPackage<Osx.Outbound> outboundPackage) {
////                TransferQueue transferQueue = (TransferQueue) context.getData(Dict.TRANSFER_QUEUE);
////                if (transferQueue != null) {
////                    transferQueue.cacheReceivedMsg(inboundPackage.getBody().getMetadataMap().get(Osx.Metadata.MessageCode.name()), outboundPackage);
////                }
////            }
////        });
//    }

    @Override
    protected ProduceResponse doService(OsxContext context, ProduceRequest produceRequest) {
        logger.info("push service begin");
        AbstractQueue   queue ;
        String topic = produceRequest.getTopic();
        context.setTopic(topic);
//        RouterInfo routerInfo = context.getRouterInfo();
//        String srcPartyId = context.getSrcPartyId();
        String sessionId = context.getSessionId();
            /*
             * 本地处理
             */
            if (StringUtils.isEmpty(topic)) {
                throw new ParameterException(StatusCode.PARAM_ERROR, "topic is null");
            }
            if (StringUtils.isEmpty(sessionId)) {
                throw new ParameterException(StatusCode.PARAM_ERROR, "sessionId is null");
            }
            int dataSize = produceRequest.getPayload().length;
            context.setActionType(ActionType.MSG_DOWNLOAD.getAlias());
            context.setRouterInfo(null);
            context.setDataSize(dataSize);

            QueueType  queueType = QueueType.NORMAL;
            if(StringUtils.isNotEmpty(context.getQueueType())){
                queueType=  QueueType.valueOf(context.getQueueType());
            }
            queue = transferQueueManager.getQueue(topic);
            CreateQueueResult createQueueResult = null;
            if (queue == null) {
                createQueueResult = transferQueueManager.createNewQueue(topic, sessionId, false,queueType);
                if (createQueueResult == null) {
                    throw new CreateTopicErrorException("create topic " + topic + " error");
                }
                queue = createQueueResult.getQueue();
                if(QueueType.DIRECT.equals(queueType)){
                    DirectBackStreamObserver  directBackStreamObserver = new  DirectBackStreamObserver(defaultFateRouterService,context.getTopic(),
                            context.getSessionId(),context.getDesNodeId(),context.getSrcNodeId());
                    QueuePushReqStreamObserver queuePushReqStreamObserver = new QueuePushReqStreamObserver(context,
                            defaultFateRouterService,  transferQueueManager,directBackStreamObserver);
                    ((DirectQueue )queue).setStreamObserver(queuePushReqStreamObserver);
                    ((DirectQueue )queue).setInputParser(new DataParser() {
                        @Override
                        public Object parse(Object src) {
                            try {
                                Osx.PushInbound  inbound =   Osx.PushInbound.parseFrom((byte[]) src);
                                logger.info("receive push inbound ======= {}",inbound);

                                Proxy.Packet result =  Proxy.Packet.parseFrom((byte[])src);
                                logger.info("receive package ======= {}",result);

                                return  result;
                            } catch (Exception e) {
                                logger.error("parse Proxy.Packet error ",e);
                            }
                            return null;
                        }
                    });
                }
            }
            if (queue != null) {
                //限流
                // String resource = TransferUtil.buildResource(produceRequest);
             //  tokenApplyService.applyToken(context, resource, dataSize);
             //  flowCounterManager.pass(resource, dataSize);
                context.putData(Dict.TRANSFER_QUEUE, queue);
//                String msgCode = produceRequest.getMetadataMap().get(Osx.Metadata.MessageCode.name());
//                String retryCountString = produceRequest.getMetadataMap().get(Osx.Metadata.RetryCount.name());
                // TODO: 2023/9/19   重试逻辑需要再修改
                //此处为处理重复请求  
//                if (StringUtils.isNotEmpty(msgCode)) {
//                    if (transferQueue.checkMsgIdDuplicate(msgCode)) {//检查消息是不是已经存在于队列里面
//                        if (StringUtils.isBlank(retryCountString)) {//重复请求，非重试请求
//                            Osx.TransportOutbound.Builder outBoundBuilder = Osx.TransportOutbound.newBuilder();
//                            outBoundBuilder.setCode(StatusCode.SUCCESS);
//                            outBoundBuilder.setMessage(Dict.DUP_MSG);
//                            return outBoundBuilder.build();
//                        } else {
//                            logger.info("receive retry request , topic {} msgcode {} try count {}", topic, msgCode, retryCountString);
//                        }
//                        OutboundPackage<Osx.TransportOutbound> cacheReceivedMsg = transferQueue.getReceivedMsgCache(msgCode);
//                        if (cacheReceivedMsg != null) {//返回上次缓存的结果
//                            return cacheReceivedMsg.getData();
//                        } else {//重试请求，但是缓存的结果已经过期
//                            logger.warn("The cached message has expired , msgCode = {}", msgCode);
//                            Osx.TransportOutbound.Builder outBoundBuilder = Osx.TransportOutbound.newBuilder();
//                            outBoundBuilder.setCode(StatusCode.SUCCESS);
//                            outBoundBuilder.setMessage(Dict.PROCESSED_MSG);
//                            return outBoundBuilder.build();
//                        }
//                    }
//                }

                byte[] msgBytes = produceRequest.getPayload();
                MessageFlag messageFlag = MessageFlag.SENDMSG;
                if (StringUtils.isNotEmpty(context.getMessageFlag())) {
                    messageFlag = MessageFlag.valueOf(context.getMessageFlag());
                }



//                MessageExtBrokerInner messageExtBrokerInner = MessageDecoder.buildMessageExtBrokerInner(topic, msgBytes, produceRequest.getMsgCode(), messageFlag, context.getSrcPartyId(),
//                        context.getDesPartyId());
//                messageExtBrokerInner.getProperties().put(Dict.SESSION_ID, sessionId);
//                messageExtBrokerInner.getProperties().put(Dict.SOURCE_COMPONENT, context.getSrcComponent() != null ? context.getSrcComponent() : "");
//                messageExtBrokerInner.getProperties().put(Dict.DES_COMPONENT, context.getDesComponent() != null ? context.getDesComponent() : "");
//                PutMessageResult putMessageResult = transferQueue.putMessage(messageExtBrokerInner);
//                if (putMessageResult.getPutMessageStatus() != PutMessageStatus.PUT_OK) {
//                    throw new PutMessageException("put status " + putMessageResult.getPutMessageStatus());
//                }
//                long logicOffset = putMessageResult.getMsgLogicOffset();
//                context.putData(Dict.CURRENT_INDEX, transferQueue.getIndexQueue().getLogicOffset().get());


                queue.putMessage(context,msgBytes,messageFlag,produceRequest.getMsgCode());
                context.setReturnCode(StatusCode.PTP_SUCCESS);
                ProduceResponse    produceResponse=  new ProduceResponse(StatusCode.PTP_SUCCESS,Dict.SUCCESS);
                return produceResponse;
            }
//            else {
//                /*
//                 * 集群内转发
//                 */
//                if (MetaInfo.PROPERTY_DEPLOY_MODE.equals(DeployMode.cluster.name())) {
//                    RouterInfo redirectRouterInfo = new RouterInfo();
//                    String redirectIp = createQueueResult.getRedirectIp();
//                    int redirectPort = createQueueResult.getPort();
//                    if (StringUtils.isEmpty(redirectIp) || redirectPort == 0) {
//                        logger.error("invalid redirect info {}:{}", redirectIp, redirectPort);
//                        throw new InvalidRedirectInfoException();
//                    }
//                    redirectRouterInfo.setHost(redirectIp);
//                    redirectRouterInfo.setPort(redirectPort);
//                    context.putData(Dict.ROUTER_INFO, redirectRouterInfo);
//                    context.setActionType(ActionType.INNER_REDIRECT.getAlias());
//                    return TransferUtil.redirectPush(context, produceRequest, redirectRouterInfo,true);
//                } else {
//                    logger.error("create topic {} error", topic);
//                    throw new ProduceMsgExcption();
//                }
//            }

        return null;

    }

    @Override
    protected ProduceResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return new ProduceResponse(exceptionInfo.getCode(),exceptionInfo.getMessage());
    }

    @Override
    public ProduceRequest decode(Object object) {
        logger.info("decode {}",object.getClass());

        ProduceRequest  produceRequest = null;
        if(object instanceof Osx.PushInbound){
            Osx.PushInbound inbound =  (Osx.PushInbound)object;
            produceRequest = buildProduceRequest(inbound);
        }
        if(object  instanceof  Osx.Inbound){
            Osx.Inbound  inbound = (Osx.Inbound)object;
            try {
               Osx.PushInbound  pushInbound=  Osx.PushInbound.parseFrom(inbound.getPayload());
                produceRequest = buildProduceRequest(pushInbound);
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        return produceRequest;
    }

    @Override
    public Osx.Outbound toOutbound(ProduceResponse response) {
        Osx.Outbound.Builder  builder = Osx.Outbound.newBuilder();
        builder.setCode(response.getCode());
        builder.setMessage(response.getMsg());
        return  builder.build();
    }

    private   ProduceRequest buildProduceRequest(Osx.PushInbound  inbound){
        ProduceRequest  produceRequest = new ProduceRequest();
        produceRequest.setPayload(inbound.getPayload().toByteArray());
        produceRequest.setTopic(inbound.getTopic());
        return   produceRequest;
    }
}
