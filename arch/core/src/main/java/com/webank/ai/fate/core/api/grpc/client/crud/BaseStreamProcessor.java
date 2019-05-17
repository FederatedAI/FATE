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

import com.webank.ai.fate.core.api.grpc.client.StreamProcessor;
import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

// todo: migrate to ClientCallStreamObserver
public abstract class BaseStreamProcessor<T> implements StreamProcessor<T> {
    protected StreamObserver<T> streamObserver;
    protected ClientCallStreamObserver<T> clientCallStreamObserver;
    private static final Logger LOGGER = LogManager.getLogger();

    public BaseStreamProcessor(StreamObserver<T> streamObserver) {
        this.streamObserver = streamObserver;
        this.clientCallStreamObserver = (ClientCallStreamObserver<T>) streamObserver;
    }

    @Override
    public void process() {
        try {
            while (!clientCallStreamObserver.isReady()) {
                Thread.sleep(50);
            }
        } catch (InterruptedException e) {
            LOGGER.error(ExceptionUtils.getStackTrace(e));
        }
    }

    @Override
    public void complete() {
        try {
            this.streamObserver.onCompleted();
        } catch (Exception e) {
            LOGGER.error(e);
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
    }
}
