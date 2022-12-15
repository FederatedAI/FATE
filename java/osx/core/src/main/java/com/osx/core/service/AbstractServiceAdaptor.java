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

package com.osx.core.service;

import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.Context;
import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.utils.JsonUtil;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;


import io.grpc.stub.AbstractStub;
import lombok.Data;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @Description 默认的服务适配器
 * @Author
 **/

    public abstract class AbstractServiceAdaptor<req, resp> implements ServiceAdaptor<req, resp> {


    static public AtomicInteger requestInHandle = new AtomicInteger(0);
    public static boolean isOpen = true;
    protected Logger flowLogger = LoggerFactory.getLogger("flow");
    protected String serviceName;
    Logger logger = LoggerFactory.getLogger(this.getClass().getName());
    ServiceAdaptor serviceAdaptor;
    InterceptorChain preChain = new DefaultInterceptorChain();
    InterceptorChain postChain = new DefaultInterceptorChain();
    private Map<String, Method> methodMap = Maps.newHashMap();
    private AbstractStub serviceStub;
    public AbstractServiceAdaptor() {

    }

    public void registerMethod(String actionType,Method  method){
        this.methodMap.put(actionType,method);
    }


    public Map<String, Method> getMethodMap() {
        return methodMap;
    }

    public void setMethodMap(Map<String, Method> methodMap) {
        this.methodMap = methodMap;
    }

    public AbstractServiceAdaptor  addPreProcessor(Interceptor interceptor) {
        preChain.addInterceptor(interceptor);
        return this;
    }

    public void addPostProcessor(Interceptor interceptor) {
        postChain.addInterceptor(interceptor);
    }

    public ServiceAdaptor getServiceAdaptor() {
        return serviceAdaptor;
    }

    public void setServiceAdaptor(ServiceAdaptor serviceAdaptor) {
        this.serviceAdaptor = serviceAdaptor;
    }

    public AbstractStub getServiceStub() {
        return serviceStub;
    }

    public void setServiceStub(AbstractStub serviceStub) {
        this.serviceStub = serviceStub;
    }

    public String getServiceName() {
        return serviceName;
    }

    public void setServiceName(String serviceName) {
        this.serviceName = serviceName;
    }

    protected abstract resp doService(Context context, InboundPackage<req> data);

    /**
     * @param context
     * @param data
     * @return
     * @throws Exception
     */
    @Override
    public OutboundPackage<resp> service(Context context, InboundPackage<req> data) throws RuntimeException {

        OutboundPackage<resp> outboundPackage = new OutboundPackage<resp>();
       // context.preProcess();
        List<Throwable> exceptions = Lists.newArrayList();
        context.setReturnCode(StatusCode.SUCCESS);
//        if (!isOpen) {
//            return this.serviceFailInner(context, data, new ShowDownRejectException());
//        }
        if(data.getBody()!=null) {
            context.putData(Dict.INPUT_DATA, data.getBody());
        }

        try {
            requestInHandle.addAndGet(1);
            resp result = null;
            context.setServiceName(this.serviceName);
            try {
                preChain.doPreProcess(context, data);
                result = doService(context, data);
                if (logger.isDebugEnabled()) {
                    logger.debug("do service, router info: {}, service name: {}, result: {}", JsonUtil.object2Json(context.getRouterInfo()), serviceName, result);
                }
            } catch (Throwable e) {
                exceptions.add(e);
                e.printStackTrace();
                logger.error("do service fail, cause by: {}", e.getMessage());
            }
            outboundPackage.setData(result);
            //postChain.doPostProcess(context, data, outboundPackage);

        } catch (Throwable e) {
            exceptions.add(e);
            logger.error("service error",e);
        } finally {
            requestInHandle.decrementAndGet();
            try {
                if (exceptions.size() != 0) {
                    outboundPackage = this.serviceFail(context, data, exceptions);
                }
            }finally {
                printFlowLog(context);
            }
          //  int returnCode = context.getReturnCode();

//            if(outboundPackage.getData()!=null) {
//                context.putData(Dict.OUTPUT_DATA, outboundPackage.getData());
//            }
           // context.postProcess(data, outboundPackage);

        }
        return outboundPackage;
    }

    protected void printFlowLog(Context context) {
        
        context.printFlowLog();

//        flowLogger.info("{}|{}|{}|{}|" +
//                        "{}|{}|{}|{}|" +
//                        "{}|{}",
//                context.getSourceIp(), context.getSrcPartyId(),
//                context.getDesPartyId(), context.getReturnCode(), context.getCostTime(),
//                context.getDownstreamCost(), serviceName, context.getRouterInfo() != null ? context.getRouterInfo() : "",
//                MetaInfo.PROPERTY_PRINT_INPUT_DATA?context.getData(Dict.INPUT_DATA):"",
//                MetaInfo.PROPERTY_PRINT_OUTPUT_DATA?context.getData(Dict.OUTPUT_DATA):"");
    }

    protected OutboundPackage<resp> serviceFailInner(Context context, InboundPackage<req> data, Throwable e) {
        OutboundPackage<resp> outboundPackage = new OutboundPackage<resp>();
        ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context ,e);
        context.setReturnCode(exceptionInfo.getCode());
        resp rsp = transformExceptionInfo(context, exceptionInfo);
        outboundPackage.setData(rsp);
        outboundPackage.setThrowable(exceptionInfo.getThrowable());
        return outboundPackage;
    }

    @Override
    public OutboundPackage<resp> serviceFail(Context context, InboundPackage<req> data, List<Throwable> errors) throws RuntimeException {
        Throwable e = errors.get(0);
        logger.error("service fail ", e);
        return serviceFailInner(context, data, e);
    }

      protected abstract resp transformExceptionInfo(Context context, ExceptionInfo exceptionInfo);


}