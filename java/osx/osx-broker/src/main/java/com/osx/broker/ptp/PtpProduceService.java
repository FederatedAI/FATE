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
package com.osx.broker.ptp;

import com.osx.api.router.RouterInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.constants.MessageFlag;
import com.osx.broker.message.MessageDecoder;
import com.osx.broker.message.MessageExtBrokerInner;
import com.osx.broker.queue.CreateQueueResult;
import com.osx.broker.queue.PutMessageResult;
import com.osx.broker.queue.PutMessageStatus;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.util.TransferUtil;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.DeployMode;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.*;
import com.osx.core.service.InboundPackage;
import com.osx.core.service.Interceptor;
import com.osx.core.service.OutboundPackage;
import com.osx.core.utils.FlowLogUtil;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.osx.broker.util.TransferUtil.redirect;

public class PtpProduceService extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpProduceService.class);


    public PtpProduceService() {
        this.addPostProcessor(new Interceptor<FateContext, Osx.Inbound, Osx.Outbound>() {
            @Override
            public void doProcess(FateContext context, InboundPackage<Osx.Inbound> inboundPackage, OutboundPackage<Osx.Outbound> outboundPackage) {
                TransferQueue transferQueue = (TransferQueue) context.getData(Dict.TRANSFER_QUEUE);
                if (transferQueue != null) {
                    transferQueue.cacheReceivedMsg(inboundPackage.getBody().getMetadataMap().get(Osx.Metadata.MessageCode.name()), outboundPackage);
                }
            }
        });
    }

    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {
        TransferQueue   transferQueue ;
        String topic = context.getTopic();
        RouterInfo routerInfo = context.getRouterInfo();
        String srcPartyId = context.getSrcPartyId();
        String sessionId = context.getSessionId();
        Osx.Inbound produceRequest = data.getBody();
        if (!MetaInfo.PROPERTY_SELF_PARTY.contains(context.getDesPartyId())) {
            //向外转发
            Osx.Outbound response = null;
            int tryTime = 0;
            context.setActionType(ActionType.MSG_REDIRECT.getAlias());
            boolean usePooled = true;
            while (tryTime < MetaInfo.PROPERTY_PRODUCE_MSG_MAX_TRY_TIME) {
                tryTime++;

                try {
                    if (tryTime > 1) {
                        context.setRetryTime(tryTime);
                        produceRequest = produceRequest.toBuilder().putMetadata(Osx.Metadata.RetryCount.name(), Integer.toString(tryTime)).build();
                        usePooled = false;
                    }
                    response = redirect(context, produceRequest, routerInfo,usePooled);
                    if (response == null) {
                        continue;
                    }
                    break;
                } catch (RemoteRpcException e) {
                    logger.error("redirect retry count {}", tryTime);
                    if (tryTime == MetaInfo.PROPERTY_PRODUCE_MSG_MAX_TRY_TIME) {
                        throw e;
                    }else{
                        FlowLogUtil.printFlowLog(context);
                    }
                    try {
                        Thread.sleep(MetaInfo.PROPERTY_PRODUCE_MSG_RETRY_INTERVAL);
                    } catch (InterruptedException ignore) {

                    }
                }
            }
            return response;
        } else {
            /*
             * 本地处理
             */
            if (StringUtils.isEmpty(topic)) {
                throw new ParameterException(StatusCode.PARAM_ERROR, "topic is null");
            }
            if (StringUtils.isEmpty(sessionId)) {
                throw new ParameterException(StatusCode.PARAM_ERROR, "sessionId is null");
            }
            int dataSize = produceRequest.getSerializedSize();
            context.setActionType(ActionType.MSG_DOWNLOAD.getAlias());
            context.setRouterInfo(null);
            context.setDataSize(dataSize);
            transferQueue = ServiceContainer.transferQueueManager.getQueue(topic);
            CreateQueueResult createQueueResult = null;
            if (transferQueue == null) {
                createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(topic, sessionId, false);
                if (createQueueResult == null) {
                    throw new CreateTopicErrorException("create topic " + topic + " error");
                }
                transferQueue = createQueueResult.getTransferQueue();
            }
            String resource = TransferUtil.buildResource(produceRequest);


            if (transferQueue != null) {
                ServiceContainer.tokenApplyService.applyToken(context, resource, dataSize);
                ServiceContainer.flowCounterManager.pass(resource, dataSize);
                context.putData(Dict.TRANSFER_QUEUE, transferQueue);
                String msgCode = produceRequest.getMetadataMap().get(Osx.Metadata.MessageCode.name());
                String retryCountString = produceRequest.getMetadataMap().get(Osx.Metadata.RetryCount.name());
                //此处为处理重复请求
                if (StringUtils.isNotEmpty(msgCode)) {
                    if (transferQueue.checkMsgIdDuplicate(msgCode)) {//检查消息是不是已经存在于队列里面
                        if (StringUtils.isBlank(retryCountString)) {//重复请求，非重试请求
                            Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
                            outBoundBuilder.setCode(StatusCode.SUCCESS);
                            outBoundBuilder.setMessage(Dict.DUP_MSG);
                            return outBoundBuilder.build();
                        } else {
                            logger.info("receive retry request , topic {} msgcode {} try count {}", topic, msgCode, retryCountString);
                        }
                        OutboundPackage<Osx.Outbound> cacheReceivedMsg = transferQueue.getReceivedMsgCache(msgCode);
                        if (cacheReceivedMsg != null) {//返回上次缓存的结果
                            return cacheReceivedMsg.getData();
                        } else {//重试请求，但是缓存的结果已经过期
                            logger.warn("The cached message has expired , msgCode = {}", msgCode);
                            Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
                            outBoundBuilder.setCode(StatusCode.SUCCESS);
                            outBoundBuilder.setMessage(Dict.PROCESSED_MSG);
                            return outBoundBuilder.build();
                        }
                    }
                }

                byte[] msgBytes = produceRequest.getPayload().toByteArray();
                String flag = produceRequest.getMetadataMap().get(Osx.Metadata.MessageFlag.name());
                MessageFlag messageFlag = MessageFlag.SENDMSG;
                if (StringUtils.isNotEmpty(flag)) {
                    messageFlag = MessageFlag.valueOf(flag);
                }
                context.putData(Dict.MESSAGE_FLAG, messageFlag.name());
                MessageExtBrokerInner messageExtBrokerInner = MessageDecoder.buildMessageExtBrokerInner(topic, msgBytes, msgCode, messageFlag, context.getSrcPartyId(),
                        context.getDesPartyId());
                messageExtBrokerInner.getProperties().put(Dict.SESSION_ID, sessionId);
                messageExtBrokerInner.getProperties().put(Dict.SOURCE_COMPONENT, context.getSrcComponent() != null ? context.getSrcComponent() : "");
                messageExtBrokerInner.getProperties().put(Dict.DES_COMPONENT, context.getDesComponent() != null ? context.getDesComponent() : "");
                PutMessageResult putMessageResult = transferQueue.putMessage(messageExtBrokerInner);
                if (putMessageResult.getPutMessageStatus() != PutMessageStatus.PUT_OK) {
                    throw new PutMessageException("put status " + putMessageResult.getPutMessageStatus());
                }
                long logicOffset = putMessageResult.getMsgLogicOffset();
                context.putData(Dict.CURRENT_INDEX, transferQueue.getIndexQueue().getLogicOffset().get());
                Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
                outBoundBuilder.setCode(StatusCode.SUCCESS);
                outBoundBuilder.setMessage(Dict.SUCCESS);
                return outBoundBuilder.build();
            } else {
                /*
                 * 集群内转发
                 */
                if (MetaInfo.PROPERTY_DEPLOY_MODE.equals(DeployMode.cluster.name())) {
                    RouterInfo redirectRouterInfo = new RouterInfo();
                    String redirectIp = createQueueResult.getRedirectIp();
                    int redirectPort = createQueueResult.getPort();
                    if (StringUtils.isEmpty(redirectIp) || redirectPort == 0) {
                        logger.error("invalid redirect info {}:{}", redirectIp, redirectPort);
                        throw new InvalidRedirectInfoException();
                    }
                    redirectRouterInfo.setHost(redirectIp);
                    redirectRouterInfo.setPort(redirectPort);
                    context.putData(Dict.ROUTER_INFO, redirectRouterInfo);
                    context.setActionType(ActionType.INNER_REDIRECT.getAlias());
                    return redirect(context, produceRequest, redirectRouterInfo,true);
                } else {
                    logger.error("create topic {} error", topic);
                    throw new ProduceMsgExcption();
                }
            }
        }
    }
}
