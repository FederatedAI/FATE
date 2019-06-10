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

package com.webank.ai.eggroll.framework.egg.api.grpc.server;

import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.framework.egg.NodeServiceGrpc;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.utils.RuntimeUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.egg.node.manager.ProcessorManager;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    @Autowired
    private ToStringUtils toStringUtils;

    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public void getProcessor(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoint> responseObserver) {
        LOGGER.info("[EGG][NODEMANAGER] getProcessor. request: {}", toStringUtils.toOneLineString(request));
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            int port = processorManager.get();

            BasicMeta.Endpoint.Builder resultBuilder = BasicMeta.Endpoint.newBuilder()
                    .setIp(runtimeUtils.getMySiteLocalAddress())
                    .setPort(port);

            LOGGER.info("[EGG][NODEMANAGER] getProcessor. result: {}", toStringUtils.toOneLineString(resultBuilder.build()));
            responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        });
    }

    @Override
    // todo: cache result when it is not changed
    public void getAllPossibleProcessors(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoints> responseObserver) {
        LOGGER.info("[EGG][NODEMANAGER] getAllPossibleProcessors. request: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            ArrayList<Integer> allAvailables = processorManager.getAllPossible();

            String mySiteLocalAddress = runtimeUtils.getMySiteLocalAddress();

            BasicMeta.Endpoint.Builder endpointBuilder = BasicMeta.Endpoint.newBuilder().setIp(mySiteLocalAddress);
            BasicMeta.Endpoints.Builder resultBuilder = BasicMeta.Endpoints.newBuilder();

            for (Integer port : allAvailables) {
                BasicMeta.Endpoint endpoint = endpointBuilder.clone().setPort(port).build();
                resultBuilder.addEndpoints(endpoint);
            }

            responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        });
    }

    @Override
    public void killProcessor(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoint> responseObserver) {
        LOGGER.info("[EGG][NODEMANAGER] killProcessor. request: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            int port = request.getPort();
            processorManager.kill(port);

            BasicMeta.Endpoint.Builder resultBuilder = BasicMeta.Endpoint.newBuilder()
                    .setIp(runtimeUtils.getMySiteLocalAddress())
                    .setPort(port);

            responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        });
    }

    @Override
    public void killAllProcessors(BasicMeta.Endpoint request, StreamObserver<BasicMeta.Endpoints> responseObserver) {
        LOGGER.info("[EGG][NODEMANAGER] killAllProcessor. request: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            ArrayList<Integer> allAvailables = processorManager.getAllPossible();

            String mySiteLocalAddress = runtimeUtils.getMySiteLocalAddress();

            BasicMeta.Endpoint.Builder endpointBuilder = BasicMeta.Endpoint.newBuilder().setIp(mySiteLocalAddress);
            BasicMeta.Endpoints.Builder resultBuilder = BasicMeta.Endpoints.newBuilder();

            for (Integer port : allAvailables) {
                BasicMeta.Endpoint endpoint = endpointBuilder.clone().setPort(port).build();
                resultBuilder.addEndpoints(endpoint);
            }

            processorManager.killAll();

            responseObserver.onNext(resultBuilder.build());
            responseObserver.onCompleted();
        });
    }
}
