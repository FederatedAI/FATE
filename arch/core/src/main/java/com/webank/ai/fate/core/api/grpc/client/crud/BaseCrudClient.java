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

package com.webank.ai.fate.core.api.grpc.client.crud;


import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.core.factory.CallMetaModelFactory;
import com.webank.ai.fate.core.factory.GrpcStreamObserverFactory;
import com.webank.ai.fate.core.factory.GrpcStubFactory;
import com.webank.ai.fate.core.model.DelayedResult;
import com.webank.ai.fate.core.model.impl.SingleDelayedResult;
import com.webank.ai.fate.core.utils.ErrorUtils;
import com.webank.ai.fate.core.utils.ReflectionUtils;
import io.grpc.ManagedChannel;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.lang.reflect.ParameterizedType;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
/**
 * S: Stub type
 *
 */
public abstract class BaseCrudClient<S> {
    protected static final Logger LOGGER = LogManager.getLogger();
    protected S stub;
    @Autowired
    private GrpcStubFactory grpcStubFactory;
    @Autowired
    private GrpcStreamObserverFactory grpcStreamObserverFactory;
    @Autowired
    private CallMetaModelFactory callMetaModelFactory;
    @Autowired
    private ReflectionUtils reflectionUtils;
    @Autowired
    private ErrorUtils errorUtils;
    private BasicMeta.Endpoint endpoint;
    private Class<S> stubClass;
    private Class grpcClass;

    public void init(BasicMeta.Endpoint endpoint) {
        this.endpoint = endpoint;
        this.stubClass = (Class<S>) ((ParameterizedType) getClass().getGenericSuperclass()).getActualTypeArguments()[0];

        try {
            grpcClass = Class.forName(StringUtils.substringBeforeLast(stubClass.getCanonicalName(), "."));
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException(e);
        }
        this.stub = (S) grpcStubFactory.createGrpcStub(true, grpcClass, endpoint, false);
    }

    public void init(String ip, int port) {
        BasicMeta.Endpoint.Builder builder = BasicMeta.Endpoint.newBuilder();
        BasicMeta.Endpoint endpoint = builder.setIp(ip).setPort(port).build();

        init(endpoint);
    }

    protected Object doCrudRequest(Object model, CrudRequestProcessor crudRequestProcessor) {
        BasicMeta.CallRequest request = callMetaModelFactory.createCallRequestFromObject(model);

        DelayedResult<BasicMeta.CallResponse> delayedResult = new SingleDelayedResult<>();

        StreamObserver<BasicMeta.CallResponse> responseObserver
                = grpcStreamObserverFactory.createDelayedResultResponseStreamObserver(delayedResult);

        // check if stub's channel is shutdown or terminated
        AbstractStub abstractStub = (AbstractStub) stub;
        ManagedChannel managedChannel = (ManagedChannel) abstractStub.getChannel();
        if (managedChannel.isShutdown() || managedChannel.isTerminated()) {
            LOGGER.info("[COMMON] invalid channel. status: {}", managedChannel.getState(true).name());
            this.stub = (S) grpcStubFactory.createGrpcStub(true, grpcClass, endpoint, false);
        }

        crudRequestProcessor.process(stub, request, responseObserver);

        BasicMeta.CallResponse response = null;
        try {
            // todo: if server has error response this should return immediately
            response = delayedResult.getResult(10, TimeUnit.MINUTES);

            // todo: pull this error handling code up if duplicate
            BasicMeta.ReturnStatus returnStatus = null;

            if (response != null) {
                returnStatus = response.getReturnStatus();
            }

            if (response == null || returnStatus == null || 0 != returnStatus.getCode()) {
                StringBuilder errMsgBuilder = new StringBuilder();
                errMsgBuilder.append("Error in crud operation: ");

                if (delayedResult.hasError()) {
                    errMsgBuilder.append(errorUtils.getStackTrace(delayedResult.getError()));
                } else if (response == null) {
                    errMsgBuilder.append("call response is null");
                } else if (returnStatus != null) {
                    errMsgBuilder.append("errcode: ")
                            .append(returnStatus.getCode())
                            .append(":")
                            .append(returnStatus.getMessage());
                }

                throw new RuntimeException(errMsgBuilder.toString());
            }
        } catch (InterruptedException e) {
            // todo: exception handling
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }

        responseObserver.onCompleted();

        return callMetaModelFactory.extractModelObject(response);
    }

    protected <T> T doCrudRequest(Object model, CrudRequestProcessor crudRequestProcessor, Class<T> targetClass) {
        Object callResult = doCrudRequest(model, crudRequestProcessor);

        return reflectionUtils.castIfNotNull(callResult, targetClass);
    }

    public interface CrudRequestProcessor<T> {
        public void process(T stub, BasicMeta.CallRequest request, StreamObserver<BasicMeta.CallResponse> responseObserver);
    }
}
