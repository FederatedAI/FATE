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
import lombok.Data;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.broker.constants.MessageFlag;
import org.fedai.osx.broker.grpc.QueuePushReqStreamObserver;
import org.fedai.osx.broker.pojo.HttpInvoke;
import org.fedai.osx.broker.pojo.ProduceRequest;
import org.fedai.osx.broker.pojo.ProduceResponse;
import org.fedai.osx.broker.queue.*;
import org.fedai.osx.broker.router.RouterServiceRegister;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.QueueType;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.CreateTopicErrorException;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.exceptions.InvalidRequestException;
import org.fedai.osx.core.exceptions.ParameterException;
import org.fedai.osx.core.flow.FlowCounterManager;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.utils.JsonUtil;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Base64;
import static org.fedai.osx.core.constant.UriConstants.HTTP_PUSH;
import static org.fedai.osx.core.constant.UriConstants.PUSH;

@Singleton
@Register(uris = {PUSH, HTTP_PUSH})
@Data
public class ProduceService extends AbstractServiceAdaptorNew<ProduceRequest, ProduceResponse> {
    Logger logger = LoggerFactory.getLogger(ProduceService.class);
    @Inject
    TransferQueueManager transferQueueManager;
    @Inject
    FlowCounterManager flowCounterManager;
    @Inject
    RouterServiceRegister routerServiceRegister;

    Base64.Decoder base64Decoder = Base64.getDecoder();

    @Override
    protected ProduceResponse doService(OsxContext context, ProduceRequest produceRequest) {

        AbstractQueue queue;
        String topic = produceRequest.getTopic();
        context.setTopic(topic);
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
        context.setActionType(ActionType.MSG_DOWNLOAD.name());
        context.setRouterInfo(null);
        context.setDataSize(dataSize);

        QueueType queueType = QueueType.NORMAL;
        if (StringUtils.isNotEmpty(context.getQueueType())) {
            queueType = QueueType.valueOf(context.getQueueType());
        }
        queue = transferQueueManager.getQueue(sessionId, topic);
        CreateQueueResult createQueueResult = null;
        if (queue == null) {
            createQueueResult = transferQueueManager.createNewQueue(sessionId, topic, false, queueType);
            if (createQueueResult == null) {
                throw new CreateTopicErrorException("create topic " + topic + " error");
            }
            queue = createQueueResult.getQueue();
        }
        if (queue != null) {
            context.putData(Dict.TRANSFER_QUEUE, queue);
            byte[] msgBytes = produceRequest.getPayload();
            MessageFlag messageFlag = MessageFlag.SENDMSG;
            if (StringUtils.isNotEmpty(context.getMessageFlag())) {
                messageFlag = MessageFlag.valueOf(context.getMessageFlag());
            }
            queue.putMessage(context, msgBytes, messageFlag, produceRequest.getMsgCode());
            context.setReturnCode(StatusCode.PTP_SUCCESS);
            ProduceResponse produceResponse = new ProduceResponse(StatusCode.PTP_SUCCESS, Dict.SUCCESS);
            return produceResponse;
        }else{

        }
        return null;
    }

    @Override
    protected ProduceResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return new ProduceResponse(exceptionInfo.getCode(), exceptionInfo.getMessage());
    }

    @Override
    public ProduceRequest decode(Object object) {
        ProduceRequest produceRequest = null;
        if (object instanceof Osx.PushInbound) {
            Osx.PushInbound inbound = (Osx.PushInbound) object;
            produceRequest = buildProduceRequest(inbound);
        }
        if (object instanceof HttpInvoke) {
            HttpInvoke inbound = (HttpInvoke) object;
            produceRequest = JsonUtil.json2Object(inbound.getPayload(), ProduceRequest.class);
        }
        if (object instanceof Osx.Inbound) {
            Osx.Inbound inbound = (Osx.Inbound) object;
            try {
                Osx.PushInbound pushInbound = Osx.PushInbound.parseFrom(inbound.getPayload());
                produceRequest = buildProduceRequest(pushInbound);
            } catch (InvalidProtocolBufferException e) {
                e.printStackTrace();
            }
        }
        if(produceRequest==null){
            logger.error("invalid produce request {}",object.getClass());
            throw new InvalidRequestException("invalid request for produce msg");
        }
        return produceRequest;
    }

    @Override
    public Osx.Outbound toOutbound(ProduceResponse response) {
        Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();
        builder.setCode(response.getCode());
        builder.setMessage(response.getMsg());
        return builder.build();
    }

    private ProduceRequest buildProduceRequest(Osx.PushInbound inbound) {
        ProduceRequest produceRequest = new ProduceRequest();
        produceRequest.setPayload(inbound.getPayload().toByteArray());
        produceRequest.setTopic(inbound.getTopic());
        return produceRequest;
    }
}
