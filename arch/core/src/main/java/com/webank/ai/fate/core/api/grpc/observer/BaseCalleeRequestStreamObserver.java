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

package com.webank.ai.fate.core.api.grpc.observer;

import com.google.common.collect.Lists;
import com.webank.ai.fate.core.utils.ErrorUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;

/**
 * This class is used at SERVER SIDE in source streaming for server to handleCheckedException incoming stream objects.
 *
 * @param <R> calleR parameter type (in the source streaming context, objects from type S are being streaming out)
 * @param <E> calleE parameter type (in the source streaming context, an object from type R will be returned)
 */
public abstract class BaseCalleeRequestStreamObserver<R, E> implements StreamObserver<R> {
    protected final StreamObserver<E> callerNotifier;
    private final Logger LOGGER = LogManager.getLogger(this);
    @Autowired
    protected ErrorUtils errorUtils;
    protected Throwable throwable;
    protected List<StreamObserver> errorListeners;
    private String classSimpleName;

    public BaseCalleeRequestStreamObserver(StreamObserver<E> callerNotifier) {
        this.callerNotifier = callerNotifier;
        this.errorListeners = Lists.newLinkedList();

        this.classSimpleName = this.getClass().getSimpleName();
    }

    @Override
    public void onError(Throwable throwable) {
        this.throwable = errorUtils.toGrpcRuntimeException(throwable);

        for (StreamObserver streamObserver : errorListeners) {
            if (streamObserver != null) {
                streamObserver.onError(throwable);
            }
        }

        callerNotifier.onError(this.throwable);
        LOGGER.debug("{} dest streaming error: {}", this.classSimpleName, errorUtils.getStackTrace(throwable));
    }

    @Override
    public void onCompleted() {
        callerNotifier.onCompleted();
    }

    public void addErrorListener(StreamObserver streamObserver) {
        if (streamObserver != null) {
            this.errorListeners.add(streamObserver);
        }
    }
}
