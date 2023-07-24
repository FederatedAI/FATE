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

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.osx.api.context.Context;
import com.osx.core.constant.Dict;
import com.osx.core.constant.StatusCode;
import com.osx.core.context.FateContext;
import com.osx.core.exceptions.ErrorMessageUtil;
import com.osx.core.exceptions.ExceptionInfo;
import com.osx.core.utils.FlowLogUtil;
import io.grpc.stub.AbstractStub;
import org.apache.commons.lang3.exception.ExceptionUtils;
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

public abstract class AbstractServiceAdaptor<ctx extends Context, req, resp> implements ServiceAdaptor<ctx, req, resp> {


    static public AtomicInteger requestInHandle = new AtomicInteger(0);
    public static boolean isOpen = true;
//    protected Logger flowLogger = LoggerFactory.getLogger("flow");
    protected String serviceName;
    Logger logger = LoggerFactory.getLogger(this.getClass().getName());
    ServiceAdaptor<ctx, req, resp> serviceAdaptor;
    InterceptorChain<ctx, req, resp> preChain = new DefaultInterceptorChain<>();
    InterceptorChain<ctx, req, resp> postChain = new DefaultInterceptorChain<>();
    private Map<String, Method> methodMap = Maps.newHashMap();
    private AbstractStub serviceStub;

    public AbstractServiceAdaptor() {

    }

    public void registerMethod(String actionType, Method method) {
        this.methodMap.put(actionType, method);
    }


    public Map<String, Method> getMethodMap() {
        return methodMap;
    }

    public void setMethodMap(Map<String, Method> methodMap) {
        this.methodMap = methodMap;
    }

    public AbstractServiceAdaptor<ctx, req, resp> addPreProcessor(Interceptor interceptor) {
        preChain.addInterceptor(interceptor);
        return this;
    }

    public void addPostProcessor(Interceptor interceptor) {
        postChain.addInterceptor(interceptor);
    }

    public ServiceAdaptor<ctx, req, resp> getServiceAdaptor() {
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

    protected abstract resp doService(ctx context, InboundPackage<req> data);

    /**
     * @param context
     * @param data
     * @return
     * @throws Exception
     */
    @Override
    public OutboundPackage<resp> service(ctx context, InboundPackage<req> data) throws RuntimeException {

        OutboundPackage<resp> outboundPackage = new OutboundPackage<resp>();
        // context.preProcess();
        List<Throwable> exceptions = Lists.newArrayList();
        context.setReturnCode(StatusCode.SUCCESS);
//        if (!isOpen) {
//            return this.serviceFailInner(context, data, new ShowDownRejectException());
//        }
        if (data.getBody() != null) {
            context.putData(Dict.INPUT_DATA, data.getBody());
        }

        try {
            requestInHandle.addAndGet(1);
            resp result = null;
            context.setServiceName(this.serviceName);
            try {
                preChain.doProcess(context, data, outboundPackage);
                result = doService(context, data);
            } catch (Throwable e) {
                exceptions.add(e);
                e.printStackTrace();
                logger.error("do service fail, {} ", ExceptionUtils.getStackTrace(e));
            }
            outboundPackage.setData(result);
        } catch (Throwable e) {
            exceptions.add(e);
            logger.error("service error", e);
        } finally {
            requestInHandle.decrementAndGet();
            try {
                if (exceptions.size() != 0) {
                    outboundPackage = this.serviceFail(context, data, exceptions);
                }
            } finally {
                if(context instanceof FateContext )
                {
                    FateContext fateContext =(FateContext )context;
                    if(fateContext.needPrintFlowLog()){
                        FlowLogUtil.printFlowLog(context);
                    }
                }else {

                    FlowLogUtil.printFlowLog(context);
                }
            }
            //  int returnCode = context.getReturnCode();

//            if(outboundPackage.getData()!=null) {
//                context.putData(Dict.OUTPUT_DATA, outboundPackage.getData());
//            }
            // context.postProcess(data, outboundPackage);

        }
        try {
            postChain.doProcess(context, data, outboundPackage);
        } catch (Exception e) {
            logger.error("service PostDoProcess error", e);
        }
        return outboundPackage;
    }

//    protected void printFlowLog(ctx context) {
////        context.printFlowLog();
//        FlowLogUtil.printFlowLog(context);
//    }

    protected OutboundPackage<resp> serviceFailInner(ctx context, InboundPackage<req> data, Throwable e) {
        OutboundPackage<resp> outboundPackage = new OutboundPackage<resp>();
        ExceptionInfo exceptionInfo = ErrorMessageUtil.handleExceptionExceptionInfo(context, e);
        context.setReturnCode(exceptionInfo.getCode());
        context.setReturnMsg(exceptionInfo.getMessage());
        resp rsp = transformExceptionInfo(context, exceptionInfo);
        outboundPackage.setData(rsp);
        outboundPackage.setThrowable(exceptionInfo.getThrowable());
        return outboundPackage;
    }

    @Override
    public OutboundPackage<resp> serviceFail(ctx context, InboundPackage<req> data, List<Throwable> errors) throws RuntimeException {
        Throwable e = errors.get(0);
        logger.error("service fail ", e);
        return serviceFailInner(context, data, e);
    }

    protected abstract resp transformExceptionInfo(ctx context, ExceptionInfo exceptionInfo);


}