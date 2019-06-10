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

package com.webank.ai.eggroll.framework.roll.factory;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.egg.NodeServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcAsyncClientContext;
import com.webank.ai.eggroll.core.api.grpc.client.GrpcStreamingClientTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

@Component
@SuppressWarnings("unchecked")
public class EggNodeServiceCallModelTemplateFactory {
    @Autowired
    private ApplicationContext applicationContext;

    public GrpcAsyncClientContext<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint>
    createEndpointToEndpointContext() {
        GrpcAsyncClientContext<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint> result =
                (GrpcAsyncClientContext<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint>)
                        applicationContext.getBean(GrpcAsyncClientContext.class);
        result.setStubClass(NodeServiceGrpc.NodeServiceStub.class);

        return result;
    }

    public GrpcStreamingClientTemplate<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint>
    createEndpointToEndpointTemplate() {
        return (GrpcStreamingClientTemplate<NodeServiceGrpc.NodeServiceStub, BasicMeta.Endpoint, BasicMeta.Endpoint>)
                applicationContext.getBean(GrpcStreamingClientTemplate.class);
    }
}


