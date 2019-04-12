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

package com.webank.ai.fate.core.api.grpc.server;

import com.webank.ai.fate.core.utils.ErrorUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

@Component
@Scope("prototype")
public class GrpcServerWrapper {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private ErrorUtils errorUtils;

    private volatile boolean inited = false;

    @PostConstruct
    private void init() {
        if (errorUtils == null) {
            errorUtils = new ErrorUtils();
        }
        inited = true;
    }

    public void wrapGrpcServerRunnable(StreamObserver responseObserver, GrpcServerRunnable target) {
        if (!inited) {
            init();
        }
        try {
            target.run();
        } catch (Throwable t) {
            LOGGER.error(errorUtils.getStackTrace(t));
            responseObserver.onError(errorUtils.toGrpcRuntimeException(t));
        }
    }

    public <T> T wrapGrpcServerCallable(StreamObserver responseObserver, GrpcServerCallable<T> target) {
        T result = null;

        try {
            result = target.run();
        } catch (Throwable t) {
            LOGGER.error(errorUtils.getStackTrace(t));
            responseObserver.onError(errorUtils.toGrpcRuntimeException(t));
        }

        return result;
    }
}
