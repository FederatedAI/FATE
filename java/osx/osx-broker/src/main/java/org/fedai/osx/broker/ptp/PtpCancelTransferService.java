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

import org.fedai.osx.broker.ServiceContainer;
import org.fedai.osx.core.constant.Dict;
import org.fedai.osx.core.constant.StatusCode;
import org.fedai.osx.core.context.FateContext;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;

import java.util.List;

public class PtpCancelTransferService extends AbstractPtpServiceAdaptor {

    public PtpCancelTransferService() {
        this.setServiceName("cancel-unary");
    }



    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {

        String sessionId = context.getSessionId();
        String topic = context.getTopic();
        List<String> cleanedTransferId = ServiceContainer.transferQueueManager.cleanByParam(sessionId, topic);
        if (cleanedTransferId != null) {
            for (String transferIdClean : cleanedTransferId) {
                ServiceContainer.consumerManager.onComplete(transferIdClean);
            }
        }
        Osx.Outbound.Builder outBoundBuilder = Osx.Outbound.newBuilder();
        outBoundBuilder.setCode(StatusCode.SUCCESS).setMessage(Dict.SUCCESS);
        return outBoundBuilder.build();
    }


}
