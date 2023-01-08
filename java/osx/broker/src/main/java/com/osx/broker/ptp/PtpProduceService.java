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
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.exceptions.*;
import com.osx.core.router.RouterInfo;
import com.osx.core.service.InboundPackage;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.osx.broker.util.TransferUtil.redirect;

public class PtpProduceService extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpProduceService.class);

    @Override
    protected Osx.Outbound doService(Context context, InboundPackage<Osx.Inbound> data) {

        String topic = context.getTopic();
        boolean isDst = false;
        RouterInfo routerInfo = context.getRouterInfo();
        String srcPartyId = context.getSrcPartyId();
        String sessionId = context.getSessionId();
        Osx.Inbound produceRequest = data.getBody();
        if (MetaInfo.PROPERTY_SELF_PARTY.contains(context.getDesPartyId())) {
            isDst = true;
        }
        if (!isDst) {
            /**
             * 向外转发
             */
            return redirect(context, produceRequest, routerInfo, false);
        } else {
            /**
             * 本地处理
             */
            if (StringUtils.isEmpty(topic)) {
                throw new ParameterException(StatusCode.PARAM_ERROR, "topic is null");
            }
            if (StringUtils.isEmpty(sessionId)) {
                throw new ParameterException(StatusCode.PARAM_ERROR, "sessionId is null");
            }
            context.setActionType(ActionType.MSG_DOWNLOAD.getAlias());
            context.setRouterInfo(null);
            CreateQueueResult createQueueResult = ServiceContainer.transferQueueManager.createNewQueue(topic, sessionId, false);
            if (createQueueResult == null) {
                throw new CreateTopicErrorException("create topic "+topic+" error");
            }
            String resource = TransferUtil.buildResource(produceRequest);
            int  dataSize = produceRequest.getSerializedSize();
             ServiceContainer.tokenApplyService.applyToken(context,resource,dataSize);
            ServiceContainer.flowCounterManager.pass(resource,dataSize);
            TransferQueue transferQueue = createQueueResult.getTransferQueue();
            if (transferQueue != null) {
                //MessageExtBrokerInner messageExtBrokerInner = new MessageExtBrokerInner();
                byte[] msgBytes = produceRequest.getPayload().toByteArray();
                //context.(msgBytes.length);
                //messageExtBrokerInner.setBody(msgBytes);

                MessageExtBrokerInner messageExtBrokerInner = MessageDecoder.buildMessageExtBrokerInner(topic, msgBytes, 0, MessageFlag.MSG, context.getSrcPartyId(),
                        context.getDesPartyId());
                PutMessageResult putMessageResult = transferQueue.putMessage(messageExtBrokerInner);
                if (putMessageResult.getPutMessageStatus() != PutMessageStatus.PUT_OK) {
                    throw new PutMessageException("put status " + putMessageResult.getPutMessageStatus());
                }
                long logicOffset = putMessageResult.getMsgLogicOffset();
                Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
                outBoundBuilder.setCode(StatusCode.SUCCESS);
                outBoundBuilder.setMessage("SUCCESS");
                return outBoundBuilder.build();
            } else {
                /**
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
                    context.setRouterInfo(redirectRouterInfo);
                    context.setActionType(ActionType.INNER_REDIRECT.getAlias());
                    return redirect(context, produceRequest, redirectRouterInfo, true);
                } else {
                    logger.error("create topic {} error", topic);
                    throw new ProduceMsgExcption();
                }
            }
        }
    }


}
