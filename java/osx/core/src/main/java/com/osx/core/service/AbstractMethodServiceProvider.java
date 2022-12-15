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
//
//package com.osx.core.service;
//
//
//import com.firework.core.annotation.FateService;
//import com.firework.core.annotation.FateServiceMethod;
//import com.osx.core.context.Context;
//import com.osx.core.exceptions.BaseException;
//import com.osx.core.exceptions.SysException;
//import com.osx.core.exceptions.UnSupportMethodException;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.lang.reflect.InvocationTargetException;
//import java.lang.reflect.Method;
//import java.util.Map;
//
//public abstract class AbstractMethodServiceProvider<req, resp> extends AbstractServiceAdaptor<req, resp> {
//
//    Logger logger  = LoggerFactory.getLogger(AbstractMethodServiceProvider.class);
//    @Override
//    protected resp transformExceptionInfo(Context context, ExceptionInfo exceptionInfo) {
//        return null;
//    }
//
//    public  AbstractMethodServiceProvider(){
//        Method[] methods = this.getClass().getMethods();
//        for (Method method : methods) {
//            FateServiceMethod fateServiceMethod = method.getAnnotation(FateServiceMethod.class);
//            if (fateServiceMethod != null) {
//                String[] names = fateServiceMethod.name();
//                for (String name : names) {
//                    this.getMethodMap().put(name, method);
//                }
//            }
//        }
//
//    }
//
//
//
//
//    @Override
//    protected resp doService(Context context, InboundPackage<req> data, OutboundPackage<resp> outboundPackage) {
//        Map<String, Method> methodMap = this.getMethodMap();
//        String actionType = context.getActionType();
//        resp result = null;
//        try {
//            Method method = methodMap.get(actionType);
//            if (method == null) {
//                throw new UnSupportMethodException();
//            }
//            result = (resp) method.invoke(this, context, data);
//        } catch (Throwable e) {
//            logger.error("xxxxxxx",e);
//            if (e.getCause() != null && e.getCause() instanceof BaseException) {
//                BaseException baseException = (BaseException) e.getCause();
//                throw baseException;
//            } else if (e instanceof InvocationTargetException) {
//                InvocationTargetException ex = (InvocationTargetException) e;
//                throw new SysException(ex.getTargetException().getMessage());
//            } else {
//                throw new SysException(e.getMessage());
//            }
//        }
//        return result;
//    }
//
//    @Override
//    protected void printFlowLog(Context context) {
//
//        flowLogger.info("{}|{}|" +
//                        "{}|{}|{}|{}|" +
//                        "{}|{}",
//                context.getCaseId(), context.getReturnCode(), context.getCostTime(),
//                context.getDownstreamCost(), serviceName, context.getRouterInfo() != null ? context.getRouterInfo() : "");
//    }
//}
