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

package com.webank.ai.fate.eggroll.egg.api.grpc.server;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.eggroll.egg.NodeServiceGrpc;
import com.webank.ai.fate.core.utils.RuntimeUtils;
import com.webank.ai.fate.eggroll.egg.node.manager.ProcessorManager;
import io.grpc.stub.StreamObserver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.ArrayList;

@Component
@Scope("prototype")
public class NodeServiceImpl extends NodeServiceGrpc.NodeServiceImplBase {
    @Autowired
    private ProcessorManager processorManager;
    @Autowired
    private RuntimeUtils runtimeUtils;

    @Override
    public void getProcessor(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoint> responseObserver) {
        int port = processorManager.get();

        BasicMeta.Endpoint.Builder resultBuilder = BasicMeta.Endpoint.newBuilder()
                .setIp(runtimeUtils.getMySiteLocalAddress())
                .setPort(port);

        responseObserver.onNext(resultBuilder.build());
        responseObserver.onCompleted();
    }

    @Override
    // todo: cache result when it is not changed
    public void getAllPossibleProcessors(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoints> responseObserver) {
        ArrayList<Integer> allAvailables = processorManager.getAllAvailable();

        String mySiteLocalAddress = runtimeUtils.getMySiteLocalAddress();

        BasicMeta.Endpoint.Builder endpointBuilder = BasicMeta.Endpoint.newBuilder().setIp(mySiteLocalAddress);
        BasicMeta.Endpoints.Builder resultBuilder = BasicMeta.Endpoints.newBuilder();

        for (Integer port : allAvailables) {
            BasicMeta.Endpoint endpoint = endpointBuilder.clone().setPort(port).build();
            resultBuilder.addEndpoints(endpoint);
        }

        responseObserver.onNext(resultBuilder.build());
        responseObserver.onCompleted();
    }

    @Override
    public void killProcessor(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoint> responseObserver) {
        super.killProcessor(request, responseObserver);
    }

    @Override
    public void killAllProcessors(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoints> responseObserver) {
        super.killAllProcessors(request, responseObserver);
    }
}
