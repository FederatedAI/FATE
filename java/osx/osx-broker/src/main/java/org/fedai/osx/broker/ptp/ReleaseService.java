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
import org.fedai.osx.broker.consumer.ConsumerManager;
import org.fedai.osx.broker.pojo.ConsumeRequest;
import org.fedai.osx.broker.pojo.ConsumerResponse;
import org.fedai.osx.broker.pojo.ReleaseRequest;
import org.fedai.osx.broker.pojo.ReleaseResponse;
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.AbstractServiceAdaptorNew;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;

import java.util.List;
@Singleton
@Register(uris= {UriConstants.RELEASE},allowInterUse = false)
public class ReleaseService extends AbstractServiceAdaptorNew<ReleaseRequest, ReleaseResponse> {
    public ReleaseService() {
    }
    @Inject
    TransferQueueManager   transferQueueManager;
    @Inject
    ConsumerManager  consumerManager;

    @Override
    protected ReleaseResponse doService(OsxContext context, ReleaseRequest data) {

        ReleaseResponse  releaseResponse = new  ReleaseResponse();
        context.setActionType(ActionType.CANCEL_TOPIC.name());
        String sessionId = context.getSessionId();
        String topic = data.getTopic();
        context.setTopic(topic);
        List<String> cleanedTransferId = transferQueueManager.cleanByParam(sessionId, topic);
        if (cleanedTransferId != null) {
            for (String transferIdClean : cleanedTransferId) {
                    consumerManager.onComplete(transferIdClean);
            }
        }
        releaseResponse.setCode(StatusCode.PTP_SUCCESS);
        releaseResponse.setMessage(Dict.SUCCESS);
        return releaseResponse;
    }
    @Override
    protected ReleaseResponse transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        ReleaseResponse  releaseResponse =  new  ReleaseResponse();
        releaseResponse.setCode(exceptionInfo.getCode());
        releaseResponse.setMessage(exceptionInfo.getMessage());
        return releaseResponse;
    }

    @Override
    public ReleaseRequest decode(Object object) {
        return null;
    }

    @Override
    public Osx.Outbound toOutbound(ReleaseResponse response) {
        return null;
    }
}
