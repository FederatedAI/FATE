///*
// * Copyright 2019 The FATE Authors. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//package org.fedai.osx.broker.interceptor;
//
//import com.google.inject.Singleton;
//import org.fedai.osx.api.context.Context;
//import org.fedai.osx.broker.util.TransferUtil;
//import org.fedai.osx.core.service.InboundPackage;
//import org.fedai.osx.core.service.Interceptor;
//import org.fedai.osx.core.service.OutboundPackage;
//import org.ppc.ptp.Osx;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//@Singleton
//public class PcpHandleInterceptor implements Interceptor<Context,Osx.Inbound,Osx.Outbound> {
//    Logger logger = LoggerFactory.getLogger(PcpHandleInterceptor.class);
//
//    @Override
//    public void doProcess(Context context, InboundPackage<Osx.Inbound> inboundPackage, OutboundPackage<Osx.Outbound> outboundPackage) {
//        Osx.Inbound inbound = inboundPackage.getBody();
//        TransferUtil.assableContextFromInbound(context,inbound);
//    }
//}
