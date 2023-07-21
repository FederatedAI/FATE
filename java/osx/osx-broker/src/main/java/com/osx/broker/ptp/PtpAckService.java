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
import com.osx.broker.consumer.UnaryConsumer;
import com.osx.broker.queue.TransferQueue;
import com.osx.broker.queue.TransferQueueApplyInfo;
import com.osx.core.config.MetaInfo;
import com.osx.core.constant.ActionType;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ConsumerNotExistException;
import com.osx.core.exceptions.InvalidRedirectInfoException;
import com.osx.core.exceptions.TransferQueueNotExistException;
import com.osx.core.service.InboundPackage;
import org.apache.commons.lang3.StringUtils;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.osx.broker.util.TransferUtil.redirect;

public class PtpAckService extends AbstractPtpServiceAdaptor {
    Logger logger = LoggerFactory.getLogger(PtpAckService.class);

    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {
        context.setActionType(ActionType.LOCAL_ACK.getAlias());
        Osx.Inbound inbound = data.getBody();
        Osx.Outbound.Builder outboundBuilder = Osx.Outbound.newBuilder();
        String topic = context.getTopic();
//        Long offset = context.getRequestMsgIndex();
        Long  offset  = (Long)context.getData(Dict.REQUEST_INDEX);
        TransferQueue transferQueue = ServiceContainer.transferQueueManager.getQueue(topic);
        /*
         * 若本地queue不存在，则检查是否在集群中其他节点
         */
        if (transferQueue == null) {
            if (MetaInfo.isCluster()) {
                TransferQueueApplyInfo transferQueueApplyInfo = ServiceContainer.transferQueueManager.queryGlobleQueue(topic);
                if (transferQueueApplyInfo == null) {
                    throw new TransferQueueNotExistException();
                } else {

                    context.setActionType(ActionType.REDIRECT_ACK.getAlias());
                    String[] ipport = transferQueueApplyInfo.getInstanceId().split(":");

                    RouterInfo redirectRouterInfo = new RouterInfo();
                    String redirectIp = ipport[0];
                    int redirectPort = Integer.parseInt(ipport[1]);
                    if (StringUtils.isEmpty(redirectIp) || redirectPort == 0) {
                        logger.error("invalid redirect info {}:{}", redirectIp, redirectPort);
                        throw new InvalidRedirectInfoException();
                    }
                    redirectRouterInfo.setHost(redirectIp);
                    redirectRouterInfo.setPort(redirectPort);
                    //context.setRouterInfo(redirectRouterInfo);
                    return redirect(context, inbound, redirectRouterInfo,true);
                }
            } else {
                throw new TransferQueueNotExistException();
            }
        }
        UnaryConsumer unaryConsumer = ServiceContainer.consumerManager.getUnaryConsumer(topic);
        if (unaryConsumer != null) {
            unaryConsumer.ack(offset);
            //context.setCurrentMsgIndex(currentMsgIndex);
            outboundBuilder.setCode(StatusCode.SUCCESS);
            outboundBuilder.setMessage(Dict.SUCCESS);
            return outboundBuilder.build();
        } else {
            throw new ConsumerNotExistException("consumer is not exist");
        }
    }
}
