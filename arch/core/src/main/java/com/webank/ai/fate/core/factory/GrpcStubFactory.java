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

import com.google.common.base.Preconditions;
import com.webank.ai.fate.api.core.BasicMeta;
import io.grpc.ManagedChannel;
import io.grpc.stub.AbstractStub;
import org.apache.commons.lang3.reflect.MethodUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationTargetException;

@Component
public class GrpcStubFactory {
    private static final String asyncStubMethodName = "newStub";
    private static final String blockingStubMethodName = "newBlockingStub";
    @Autowired
    private GrpcChannelFactory grpcChannelFactory;

    private static final Logger LOGGER = LogManager.getLogger();

    public AbstractStub createGrpcStub(boolean isAsync, Class grpcClass, ManagedChannel managedChannel) {
        /*LOGGER.info("[CORE] create stub. channel status: isShutdown: {}, isTerminated: {}",
                managedChannel.isShutdown(), managedChannel.isTerminated());*/
        String methodName = null;
        if (isAsync) {
            methodName = asyncStubMethodName;
        } else {
            methodName = blockingStubMethodName;
        }

        AbstractStub result = null;
        try {
            result = (AbstractStub) MethodUtils.invokeStaticMethod(grpcClass, methodName, managedChannel);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException("error creating stub", e);
        }

        return result;
    }

    public AbstractStub createGrpcStub(boolean isAsync, Class grpcClass, BasicMeta.Endpoint endpoint, boolean isSecure) {
        Preconditions.checkNotNull(endpoint, "Endpoint cannot be null");
        ManagedChannel managedChannel = grpcChannelFactory.getChannel(endpoint, isSecure);

        return createGrpcStub(isAsync, grpcClass, managedChannel);
    }
}
