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
//package org.fedai.osx.broker.ptp;
//
//import com.google.inject.Inject;
//import com.google.inject.Singleton;
//import lombok.extern.slf4j.Slf4j;
//import org.fedai.osx.broker.interceptor.RouterInterceptor;
//import org.fedai.osx.broker.interceptor.TokenValidatorInterceptor;
//import org.fedai.osx.broker.service.Register;
//import org.fedai.osx.broker.util.TransferUtil;
//import org.fedai.osx.core.constant.ActionType;
//import org.fedai.osx.core.constant.UriConstants;
//import org.fedai.osx.core.context.OsxContext;
//import org.fedai.osx.core.exceptions.ExceptionInfo;
//import org.fedai.osx.core.router.RouterInfo;
//import org.fedai.osx.core.service.InboundPackage;
//import org.ppc.ptp.Osx;
//@Singleton
////@Register(uri= UriConstants.UNARYCALL)
//@Slf4j
//public class PtpUnaryCallService extends AbstractPtpServiceAdaptor< Osx.Inbound,Osx.Outbound> {
//
//    @Inject
//    public  PtpUnaryCallService(TokenValidatorInterceptor tokenValidatorInterceptor ,
//                                RouterInterceptor  routerInterceptor){
//        this.addPreProcessor(tokenValidatorInterceptor);
//        this.addPreProcessor(routerInterceptor);
//    }
//
//    @Override
//    protected Osx.Outbound doService(OsxContext context, InboundPackage<Osx.Inbound> data) {
//
//        context.setActionType(ActionType.UNARY_CALL_NEW.getAlias());
//        RouterInfo routerInfo = context.getRouterInfo();
//        Osx.Inbound inbound = data.getBody();
//       // logger.info("PtpUnaryCallService receive : {}",inbound);
//        Osx.Outbound outbound = TransferUtil.redirect(context,inbound,routerInfo,true);
//        return outbound;
//    }
//
//
//    @Override
//    protected Osx.Outbound transformExceptionInfo(OsxContext context, ExceptionInfo exceptionInfo) {
//        Osx.Outbound.Builder builder = Osx.Outbound.newBuilder();
//        builder.setCode(exceptionInfo.getCode());
//        builder.setMessage(exceptionInfo.getMessage());
//        return builder.build();
//    }
//
//}
