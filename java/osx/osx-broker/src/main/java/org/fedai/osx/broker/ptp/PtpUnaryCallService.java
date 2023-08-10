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

import org.fedai.osx.api.router.RouterInfo;
import org.fedai.osx.broker.util.TransferUtil;
import org.fedai.osx.core.constant.ActionType;
import org.fedai.osx.core.context.FateContext;
import org.fedai.osx.core.service.InboundPackage;
import org.ppc.ptp.Osx;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PtpUnaryCallService extends AbstractPtpServiceAdaptor {

    Logger logger = LoggerFactory.getLogger(PtpUnaryCallService.class);
    @Override
    protected Osx.Outbound doService(FateContext context, InboundPackage<Osx.Inbound> data) {

        context.setActionType(ActionType.UNARY_CALL_NEW.getAlias());
        RouterInfo routerInfo = context.getRouterInfo();
        Osx.Inbound inbound = data.getBody();
       // logger.info("PtpUnaryCallService receive : {}",inbound);
        Osx.Outbound outbound = TransferUtil.redirect(context,inbound,routerInfo,true);
        return outbound;
    }

}
