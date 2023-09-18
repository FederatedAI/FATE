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
package org.fedai.osx.broker.grpc;

import io.grpc.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ServiceExceptionHandler implements ServerInterceptor {
    private static final Logger logger = LoggerFactory.getLogger(ServiceExceptionHandler.class);

    @Override
    public <ReqT, RespT> ServerCall.Listener<ReqT> interceptCall(ServerCall<ReqT, RespT> call,
                                                                 Metadata requestHeaders, ServerCallHandler<ReqT, RespT> next) {
        ServerCall.Listener<ReqT> delegate = next.startCall(call, requestHeaders);
        return new ForwardingServerCallListener.SimpleForwardingServerCallListener<ReqT>(delegate) {
            @Override
            public void onHalfClose() {
                try {
                    super.onHalfClose();
                } catch (Exception e) {
                    logger.error("ServiceException:", e);
                    call.close(Status.INTERNAL
                            .withCause(e)
                            .withDescription(e.getMessage()), new Metadata());
                }
            }
        };
    }
}