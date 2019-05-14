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
import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.core.api.grpc.observer.BaseCallerResponseStreamObserver;
import com.webank.ai.fate.core.error.handler.ExceptionHandler;
import com.webank.ai.fate.core.factory.GrpcStubFactory;
import io.grpc.Metadata;
import io.grpc.stub.AbstractStub;
import io.grpc.stub.MetadataUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.Nullable;
import java.lang.reflect.ParameterizedType;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * S: Stub type
 * R: calleR type
 * E: calleE type
 */
@Component
@Scope("prototype")
public class GrpcAsyncClientContext<S extends AbstractStub, R extends Message, E extends Message> {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private GrpcStubFactory grpcStubFactory;
    @Autowired
    private ExceptionHandler exceptionHandler;
    private Class<? extends BaseCallerResponseStreamObserver> callerStreamObserverClass;
    private GrpcCallerStreamingStubMethodInvoker<S, R, E> callerStreamingMethodInvoker;
    private GrpcCalleeStreamingStubMethodInvoker<S, R, E> calleeStreamingMethodInvoker;
    private StreamProcessor<R> requestStreamProcessor;
    private long finishTimeout;
    private TimeUnit finishTimeoutUnit;
    private Object[] observerConstructorArguments;
    private Class<? extends StreamProcessor> requestStreamProcessorClass;
    private Object[] requestStreamProcessorConstructorArguments;
    private S stub;
    private Class<? extends AbstractStub> stubClass;
    private Class<?> grpcClass;
    private Metadata grpcMetadata;
    private BasicMeta.Endpoint endpoint;
    private int latchInitCount = 1;
    private boolean isSecureRequest;

    public GrpcAsyncClientContext() {
    }

    public GrpcAsyncClientContext(BasicMeta.Endpoint endpoint) {
        setEndpoint(endpoint);
    }

    public void init() {
        if (stubClass == null) {
            this.stubClass = (Class<S>) ((ParameterizedType) getClass().getGenericSuperclass()).getActualTypeArguments()[0];
        }

        try {
            this.grpcClass = Class.forName(StringUtils.substringBeforeLast(stubClass.getCanonicalName(), "."));
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException(e);
        }
    }

    public GrpcAsyncClientContext<S, R, E> setGrpcClass(Class<?> grpcClass) {
        this.grpcClass = grpcClass;
        return this;
    }

    public GrpcAsyncClientContext<S, R, E> setLatchInitCount(int latchInitCount) {
        this.latchInitCount = latchInitCount;
        return this;
    }

    public Metadata getGrpcMetadata() {
        return grpcMetadata;
    }

    public GrpcAsyncClientContext<S, R, E> setGrpcMetadata(Metadata grpcMetadata) {
        this.grpcMetadata = grpcMetadata;
        return this;
    }

    public BasicMeta.Endpoint getEndpoint() {
        return endpoint;
    }

    public GrpcAsyncClientContext<S, R, E> setEndpoint(BasicMeta.Endpoint endpoint) {
        this.endpoint = endpoint;

        return this;
    }

    public GrpcAsyncClientContext<S, R, E> setStubClass(Class<? extends AbstractStub> stubClass) {
        this.stubClass = stubClass;
        return this;
    }

    public GrpcAsyncClientContext<S, R, E> setCallerStreamObserverClassAndArguments(Class<? extends BaseCallerResponseStreamObserver> callerStreamObserverClass, Object... constructorArguments) {
        this.callerStreamObserverClass = callerStreamObserverClass;
        observerConstructorArguments = constructorArguments;
        return this;
    }

    public Class<? extends StreamProcessor> getRequestStreamProcessorClass() {
        return requestStreamProcessorClass;
    }

    public GrpcAsyncClientContext<S, R, E> setRequestStreamProcessorClassAndArguments(Class<? extends StreamProcessor<R>> streamProcessorClass, Object... constructorArguments) {
        this.requestStreamProcessorClass = streamProcessorClass;
        this.requestStreamProcessorConstructorArguments = constructorArguments;
        return this;
    }

    public Object[] getRequestStreamProcessorConstructorArguments() {
        return requestStreamProcessorConstructorArguments;
    }

    public GrpcAsyncClientContext<S, R, E> setFinishTimeout(long finishTimeout, TimeUnit finishTimeoutUnit) {
        this.finishTimeout = finishTimeout;
        this.finishTimeoutUnit = finishTimeoutUnit;
        return this;
    }

    public Class<? extends BaseCallerResponseStreamObserver> getCallerStreamObserverClass() {
        return callerStreamObserverClass;
    }

    public GrpcCallerStreamingStubMethodInvoker<S, R, E> getCallerStreamingMethodInvoker() {
        return callerStreamingMethodInvoker;
    }

    public GrpcAsyncClientContext<S, R, E> setCallerStreamingMethodInvoker(GrpcCallerStreamingStubMethodInvoker<S, R, E> callerStreamingMethodInvoker) {
        this.callerStreamingMethodInvoker = callerStreamingMethodInvoker;
        return this;
    }

    public GrpcCalleeStreamingStubMethodInvoker<S, R, E> getCalleeStreamingMethodInvoker() {
        return calleeStreamingMethodInvoker;
    }

    public GrpcAsyncClientContext<S, R, E> setCalleeStreamingMethodInvoker(GrpcCalleeStreamingStubMethodInvoker<S, R, E> calleeStreamingMethodInvoker) {
        this.calleeStreamingMethodInvoker = calleeStreamingMethodInvoker;
        return this;
    }

    public StreamProcessor<R> getRequestStreamProcessor() {
        return requestStreamProcessor;
    }

    public GrpcAsyncClientContext<S, R, E> setRequestStreamProcessor(StreamProcessor<R> requestStreamProcessor) {
        this.requestStreamProcessor = requestStreamProcessor;
        return this;
    }

    public Object[] getObserverConstructorArguments() {
        return observerConstructorArguments;
    }

    public long getFinishTimeout() {
        return finishTimeout;
    }

    public TimeUnit getFinishTimeoutUnit() {
        return finishTimeoutUnit;
    }

    public ExceptionHandler getExceptionHandler() {
        return exceptionHandler;
    }

    public GrpcAsyncClientContext<S, R, E> setExceptionHandler(ExceptionHandler exceptionHandler) {
        this.exceptionHandler = exceptionHandler;
        return this;
    }

    public boolean isSecureRequest() {
        return isSecureRequest;
    }

    public GrpcAsyncClientContext<S, R, E> setSecureRequest(boolean secureRequest) {
        isSecureRequest = secureRequest;
        return this;
    }

    public S getStub() {
        return stub;
    }

    public S createStub() {
        if (stub == null) {
            init();

            this.stub = (S) grpcStubFactory.createGrpcStub(true, grpcClass, endpoint, isSecureRequest());
            if (grpcMetadata != null) {
                this.stub = (S) MetadataUtils.attachHeaders(this.stub, grpcMetadata);
            }
        }
        return stub;
    }

    public CountDownLatch createFinishLatch() {
        return new CountDownLatch(latchInitCount);
    }

/*    public <R, E> BaseCallerResponseStreamObserver<R, E> createCallerResponseStreamObserver(
            Class<? extends BaseCallerResponseStreamObserver<R, E>> streamObserverClass, CountDownLatch finishLatch) {
        if (finishLatch == null || finishLatch.getCount() == 0L) {
            throw new IllegalStateException("finish latch is null or count == 0");
        }

        return grpcStreamObserverFactory.createCallerResponseStreamObserver(streamObserverClass, finishLatch);
    }*/

    public boolean awaitFinish(CountDownLatch finishLatch, long timeout, TimeUnit unit) throws InterruptedException {
        return finishLatch.await(timeout, unit);
    }

    public boolean awaitFinish(CountDownLatch finishLatch, long timeout, TimeUnit unit, @Nullable ExceptionHandler handler) {
        if (handler == null) {
            handler = this.exceptionHandler;
        }

        boolean result = false;
        try {
            result = awaitFinish(finishLatch, timeout, unit);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            handler.handleCheckedException(e);
        }

        return result;
    }
}
