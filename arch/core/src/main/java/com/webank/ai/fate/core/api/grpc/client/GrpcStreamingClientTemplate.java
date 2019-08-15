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

package com.webank.ai.fate.core.api.grpc.client;

import com.google.protobuf.Message;
import com.webank.ai.fate.core.api.grpc.client.crud.BaseStreamProcessor;
import com.webank.ai.fate.core.factory.GrpcStreamObserverFactory;
import com.webank.ai.fate.core.model.DelayedResult;
import com.webank.ai.fate.core.utils.ErrorUtils;
import com.webank.ai.fate.core.utils.ToStringUtils;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * S: Stub type
 * R: calleR type
 * E: calleE type
 */
@Component
@Scope("prototype")
public class GrpcStreamingClientTemplate<S extends AbstractStub, R extends Message, E extends Message> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private GrpcStreamObserverFactory grpcStreamObserverFactory;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private ToStringUtils toStringUtils;
    private GrpcAsyncClientContext<S, R, E> grpcAsyncClientContext;
    private StreamProcessor<R> streamProcessor;
    private StreamObserver<R> requestObserver;
    private CountDownLatch finishLatch;

    public GrpcStreamingClientTemplate<S, R, E> setGrpcAsyncClientContext(GrpcAsyncClientContext<S, R, E> grpcAsyncClientContext) {
        this.grpcAsyncClientContext = grpcAsyncClientContext;
        return this;
    }

    public void initCallerStreamingRpc() {
        S stub = grpcAsyncClientContext.createStub();

        finishLatch = grpcAsyncClientContext.createFinishLatch();

        StreamObserver<E> responseObserver
                = grpcStreamObserverFactory.createCallerResponseStreamObserver(
                grpcAsyncClientContext.getCallerStreamObserverClass(),
                finishLatch,
                grpcAsyncClientContext.getObserverConstructorArguments());

        requestObserver = grpcAsyncClientContext.getCallerStreamingMethodInvoker().invoke(stub, responseObserver);

        streamProcessor = grpcStreamObserverFactory.createStreamProcessor(grpcAsyncClientContext.getRequestStreamProcessorClass(),
                requestObserver,
                grpcAsyncClientContext.getRequestStreamProcessorConstructorArguments());
        // streamProcessor = grpcAsyncClientContext.getRequestStreamProcessor();
    }

    public void processCallerStreamingRpc() {

        streamProcessor.process();
    }

    public void errorCallerStreamingRpc(Throwable t) {
        requestObserver.onError(errorUtils.toGrpcRuntimeException(t));
    }

    public void completeStreamingRpc() {
        streamProcessor.complete();
        if (!(streamProcessor instanceof BaseStreamProcessor)) {
            try {
                requestObserver.onCompleted();
            } catch (IllegalStateException e) {
                LOGGER.error(errorUtils.toGrpcRuntimeException(e));
            }
        }

        grpcAsyncClientContext.awaitFinish(finishLatch,
                grpcAsyncClientContext.getFinishTimeout(),
                grpcAsyncClientContext.getFinishTimeoutUnit(),
                grpcAsyncClientContext.getExceptionHandler());
    }

    public void calleeStreamingRpc(R request) {
        S stub = grpcAsyncClientContext.createStub();

        CountDownLatch finishLatch = grpcAsyncClientContext.createFinishLatch();

        StreamObserver<E> responseObserver
                = grpcStreamObserverFactory.createCallerResponseStreamObserver(
                grpcAsyncClientContext.getCallerStreamObserverClass(),
                finishLatch,
                grpcAsyncClientContext.getObserverConstructorArguments());

        grpcAsyncClientContext.getCalleeStreamingMethodInvoker().invoke(stub, request, responseObserver);

        grpcAsyncClientContext.awaitFinish(finishLatch,
                grpcAsyncClientContext.getFinishTimeout(),
                grpcAsyncClientContext.getFinishTimeoutUnit(),
                grpcAsyncClientContext.getExceptionHandler());

        try {
            responseObserver.onCompleted();
        } catch (IllegalStateException ignore) {
            ;
            // LOGGER.warn("repeated close for request: {}", toStringUtils.toOneLineString(request));
        }
        // responseObserver.onCompleted();
    }

    public void calleeStreamingRpcNoWait(R request) {
        S stub = grpcAsyncClientContext.createStub();

        CountDownLatch finishLatch = grpcAsyncClientContext.createFinishLatch();

        StreamObserver<E> responseObserver
                = grpcStreamObserverFactory.createCallerResponseStreamObserver(
                grpcAsyncClientContext.getCallerStreamObserverClass(),
                finishLatch,
                grpcAsyncClientContext.getObserverConstructorArguments());

        grpcAsyncClientContext.getCalleeStreamingMethodInvoker().invoke(stub, request, responseObserver);
    }

    public <T> T calleeStreamingRpcWithImmediateDelayedResult(R request,
                                                              DelayedResult<T> delayedResult)
            throws InvocationTargetException {
        calleeStreamingRpc(request);

        // LOGGER.info("hasResult: {}", delayedResult.hasResult());
        if (delayedResult.hasError()) {
            throw new InvocationTargetException(delayedResult.getError());
        }

        T result = delayedResult.getResultNow();
        return result;
    }

    public <T> T calleeStreamingRpcWithTimeoutDelayedResult(R request,
                                                            DelayedResult<T> delayedResult,
                                                            long timeout, TimeUnit timeUnit)
            throws InvocationTargetException, InterruptedException {
        calleeStreamingRpc(request);

        if (delayedResult.hasError()) {
            throw new InvocationTargetException(delayedResult.getError());
        }

        T result = null;

        result = delayedResult.getResult(timeout, timeUnit);

        return result;
    }
}
