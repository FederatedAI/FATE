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
package org.fedai.osx.broker.interceptor;

import org.fedai.osx.api.context.Context;
import org.fedai.osx.api.router.RouterInfo;
import org.fedai.osx.broker.ServiceContainer;
import org.fedai.osx.broker.router.FateRouterService;
import org.fedai.osx.broker.router.RouterService;
import org.fedai.osx.core.service.InboundPackage;
import org.fedai.osx.core.service.Interceptor;
import org.fedai.osx.core.service.OutboundPackage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RouterInterceptor implements Interceptor {

    Logger logger = LoggerFactory.getLogger(RouterInterceptor.class);
    public RouterInterceptor(){
        this.fateRouterService = fateRouterService;
    }
    FateRouterService  fateRouterService;
    @Override
    public void doProcess(Context context, InboundPackage inboundPackage, OutboundPackage outboundPackage) throws Exception {
        String routerKey = buildRouterKey(context);
        RouterService routerService = ServiceContainer.routerRegister.getRouterService(routerKey);
        String sourcePartyId = context.getSrcPartyId();
        String desPartyId = context.getDesPartyId();
        String sourceComponentName  = context.getSrcComponent();
        String desComponentName = context.getDesComponent();
        RouterInfo routerInfo = routerService.route(sourcePartyId,sourceComponentName,desPartyId,desComponentName);
//        logger.info("router===================={}  =============={}",routerService,routerInfo);
        context.setRouterInfo(routerInfo);
    }
    private String buildRouterKey (Context context){
        return  context.getTechProviderCode();
    }
}
