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

package com.webank.ai.eggroll.framework.roll.api.grpc.client;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.egg.NodeServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import com.webank.ai.eggroll.core.constant.RuntimeConstants;
import com.webank.ai.eggroll.core.model.DelayedResult;
import com.webank.ai.eggroll.core.model.impl.SingleDelayedResult;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.api.grpc.observer.processor.egg.node.EggNodeServiceEndpointToEndpointResponseObserver;
import com.webank.ai.eggroll.framework.roll.factory.EggNodeServiceCallModelTemplateFactory;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.lang.reflect.InvocationTargetException;

@Component
@Scope("prototype")
public class EggNodeManagerClient {
    private static final Logger LOGGER = LogManager.getLogger();

    @Autowired
    private TypeConversionUtils typeConversionUtils;
    @Autowired
    private EggNodeServiceCallModelTemplateFactory eggNodeServiceCallModelTemplateFactory;

    public BasicMeta.Endpoint getProcessor(Node node) {
        return getProcessor(typeConversionUtils.toEndpoint(node));
    }

    public BasicMeta.Endpoint getProcessor(BasicMeta.Endpoint request) {
        GrpcAsyncClientContext<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint> context
                = eggNodeServiceCallModelTemplateFactory.createEndpointToEndpointContext();

        DelayedResult<BasicMeta.Endpoint> delayedResult = new SingleDelayedResult<>();

        context.setLatchInitCount(1)
                .setEndpoint(request)
                .setFinishTimeout(RuntimeConstants.DEFAULT_WAIT_TIME, RuntimeConstants.DEFAULT_TIMEUNIT)
                .setCalleeStreamingMethodInvoker(NodeServiceGrpc.NodeServiceStub::getProcessor)
                .setCallerStreamObserverClassAndArguments(EggNodeServiceEndpointToEndpointResponseObserver.class, delayedResult);

        GrpcStreamingClientTemplate<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint> template
                = eggNodeServiceCallModelTemplateFactory.createEndpointToEndpointTemplate();

        template.setGrpcAsyncClientContext(context);

        BasicMeta.Endpoint result;

        try {
            result = template.calleeStreamingRpcWithImmediateDelayedResult(request, delayedResult);
        } catch (InvocationTargetException e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    public BasicMeta.Endpoints getAllPossibleProcessors(Node node) {
        return null;
    }

    public BasicMeta.Endpoint killProcessor(Node node, int port) {
        return null;
    }

    public BasicMeta.Endpoints killAllProcessors(Node node) {
        return null;
    }
}
