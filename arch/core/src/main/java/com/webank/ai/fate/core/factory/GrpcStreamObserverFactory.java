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

package com.webank.ai.fate.core.factory;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.core.api.grpc.client.StreamProcessor;
import com.webank.ai.fate.core.api.grpc.observer.DefaultSyncCallResponseStreamObserver;
import com.webank.ai.fate.core.model.DelayedResult;
import io.grpc.stub.StreamObserver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.util.concurrent.CountDownLatch;

@Component
public class GrpcStreamObserverFactory {
    @Autowired
    private ApplicationContext applicationContext;

    public StreamObserver<BasicMeta.CallResponse> createDelayedResultResponseStreamObserver(DelayedResult delayedResult) {
        return applicationContext.getBean(DefaultSyncCallResponseStreamObserver.class, delayedResult);
    }

    public <T> T createCallerResponseStreamObserver(Class<T> soClass, CountDownLatch finishLatch, Object... objects) {
        T result = null;

        if (objects == null || objects.length <= 0) {
            result = (T) applicationContext.getBean(soClass, finishLatch);
        } else {
            Object[] constructorParams = new Object[objects.length + 1];
            constructorParams[0] = finishLatch;
            System.arraycopy(objects, 0, constructorParams, 1, objects.length);

            result = (T) applicationContext.getBean(soClass, constructorParams);
        }

        return result;
    }

    public <T, U> T createCallerResponseStreamObserver(Class<T> soClass, CountDownLatch finishLatch, DelayedResult<U> delayedResult, Object... objects) {
        T result = null;

        if (objects == null || objects.length <= 0) {
            result = (T) applicationContext.getBean(soClass, finishLatch, delayedResult);
        } else {
            Object[] constructorParams = new Object[objects.length + 2];
            constructorParams[0] = finishLatch;
            constructorParams[1] = finishLatch;
            System.arraycopy(objects, 0, constructorParams, 2, objects.length);

            result = (T) applicationContext.getBean(soClass, constructorParams);

            result = (T) applicationContext.getBean(soClass, finishLatch, delayedResult, objects);
        }
        return result;
    }

    public <T> StreamProcessor<T> createStreamProcessor(Class<? extends StreamProcessor> streamProcessorClass, StreamObserver streamObserver, Object... objects) {
        StreamProcessor<T> result = null;
        if (objects == null || objects.length == 0) {
            result = (StreamProcessor<T>) applicationContext.getBean(streamProcessorClass, streamObserver);
        } else {
            Object[] constructorParams = new Object[objects.length + 1];
            constructorParams[0] = streamObserver;
            System.arraycopy(objects, 0, constructorParams, 1, objects.length);

            result = (StreamProcessor<T>) applicationContext.getBean(streamProcessorClass, constructorParams);
        }

        return result;
    }

    public <T> T createCallerResponseStreamObserver(Class<T> soClass, DelayedResult<?> delayedResult) {
        return (T) applicationContext.getBean(soClass, delayedResult);
    }

    public <T> T createCalleeRequestStreamObserver(Class<T> soClass, StreamObserver<?> remoteNotifier, Object... objects) {
        return (T) applicationContext.getBean(soClass, remoteNotifier, objects);
    }
}
