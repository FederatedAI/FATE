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
import org.fedai.osx.broker.queue.TransferQueueManager;
import org.fedai.osx.broker.service.Register;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.constant.UriConstants;
import org.fedai.osx.core.context.OsxContext;
import org.fedai.osx.core.exceptions.ExceptionInfo;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;

import java.util.List;
@Singleton
//@Register(uri= UriConstants.RELEASE,allowInterUse = false)
public class ReleaseService extends AbstractPtpServiceAdaptor< Osx.ReleaseInbound, Osx.TransportOutbound> {

    public ReleaseService() {
        this.setServiceName("cancel-unary");
    }
    @Inject
    TransferQueueManager   transferQueueManager;

    @Inject
    ConsumerManager  consumerManager;

    @Override
    protected Osx.TransportOutbound doService(OsxContext context, InboundPackage<Osx.ReleaseInbound> data) {

        String sessionId = context.getSessionId();
        String topic = context.getTopic();
        List<String> cleanedTransferId = transferQueueManager.cleanByParam(sessionId, topic);
        if (cleanedTransferId != null) {
            for (String transferIdClean : cleanedTransferId) {
                    consumerManager.onComplete(transferIdClean);
            }
        }
        Osx.TransportOutbound.Builder outBoundBuilder = Osx.TransportOutbound.newBuilder();
        outBoundBuilder.setCode(StatusCode.SUCCESS).setMessage(Dict.SUCCESS);
        return outBoundBuilder.build();
    }

    @Override
    protected Osx.TransportOutbound transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
        return null;
    }


}
