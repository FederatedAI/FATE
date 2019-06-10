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

package com.webank.ai.eggroll.framework.roll.api.grpc.server;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.computing.processor.ProcessServiceGrpc;
import com.webank.ai.eggroll.api.computing.processor.Processor;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.api.storage.StorageBasic;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerRunnable;
import com.webank.ai.eggroll.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.eggroll.core.error.exception.MultipleRuntimeThrowables;
import com.webank.ai.eggroll.core.error.exception.StorageNotExistsException;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.core.model.NodeType;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.core.utils.NetworkingUtils;
import com.webank.ai.eggroll.core.utils.RandomUtils;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.EggProcessServiceClient;
import com.webank.ai.eggroll.framework.roll.factory.RollGrpcObserverFactory;
import com.webank.ai.eggroll.framework.roll.factory.RollModelFactory;
import com.webank.ai.eggroll.framework.roll.helper.NodeHelper;
import com.webank.ai.eggroll.framework.roll.service.async.processor.*;
import com.webank.ai.eggroll.framework.roll.service.handler.ProcessServiceResultHandler;
import com.webank.ai.eggroll.framework.roll.util.RollServerUtils;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFuture;

import javax.annotation.PostConstruct;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;


@Component
@Scope("prototype")
public class RollProcessServiceImpl extends ProcessServiceGrpc.ProcessServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private RandomUtils randomUtils;
    @Autowired
    private NetworkingUtils networkingUtils;
    @Autowired
    private ThreadPoolTaskExecutor asyncThreadPool;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    @Autowired
    private RollGrpcObserverFactory rollGrpcObserverFactory;
    @Autowired
    private RollModelFactory rollModelFactory;
    @Autowired
    private EggProcessServiceClient eggProcessServiceClient;
    @Autowired
    private RollServerUtils rollServerUtils;
    @Autowired
    private NodeHelper nodeHelper;

    @PostConstruct
    public void init() {
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());
    }

    @Override
    public void map(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Map] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        MapServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    @Override
    public void mapValues(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][MapValues] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        MapValuesServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    @Override
    public void join(Processor.BinaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Join] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        JoinServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    @Override
    public void reduce(Processor.UnaryProcess request, StreamObserver<Kv.Operand> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Reduce] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        ReduceServiceProcessor.class,
                        rollModelFactory.createReduceProcessServiceResultHandler()));
    }

    @Override
    public void mapPartitions(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][MapPartitions] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        MapPartitionsServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    @Override
    public void glom(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Glom] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        GlomServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    @Override
    public void sample(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Sample] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        SampleServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    @Override
    public void subtractByKey(Processor.BinaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][SubtractByKey] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        SubtractByKeyServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));

    }

    @Override
    public void filter(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Filter] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        FilterServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));

    }

    @Override
    public void union(Processor.BinaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][Union] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        UnionServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));

    }

    @Override
    public void flatMap(Processor.UnaryProcess request, StreamObserver<StorageBasic.StorageLocator> responseObserver) {
        LOGGER.info("[ROLL][PROCESS][FlatMap] request received: {}", toStringUtils.toOneLineString(request));

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver,
                new ProcessServiceTemplate<>(request,
                        responseObserver,
                        FlatMapServiceProcessor.class,
                        rollModelFactory.createProcessServiceStorageLocatorResultHandler()));
    }

    private Map<String, List<Node>> getEggTargetToNodes() {
        Map<String, List<Node>> result = Maps.newConcurrentMap();

        Node healthyEggNode = new Node();
        healthyEggNode.setType(NodeType.EGG.name());
        healthyEggNode.setStatus(NodeStatus.HEALTHY.name());

        List<Node> healthyEggs = storageMetaClient.getNodes(healthyEggNode);

        for (Node node : healthyEggs) {
            String target = networkingUtils.getIpOrHost(node);

            if (!result.containsKey(target)) {
                result.put(target, Lists.newArrayList());
            }
            List<Node> targetList = result.get(target);
            targetList.add(node);
        }

        return result;
    }

    /**
     * @param <R> grpc calleR type
     * @param <E> grpc calleE type
     * @param <I> Intermediate type for results. generally R -> I and then I -> E. in most cases E == I
     */
    public class ProcessServiceTemplate<R, E, I> implements GrpcServerRunnable {
        private R request;
        private StreamObserver<E> responseObserver;
        private Processor.UnaryProcess.Builder unaryProcessBuilder;
        private Processor.BinaryProcess.Builder binaryProcessBuilder;
        private StorageBasic.StorageLocator.Builder storageLocatorBuilder;
        private StorageBasic.StorageLocator.Builder storageLocatorBuilderTemplate;
        private Processor.TaskInfo.Builder taskInfoBuilder;
        private Class<? extends BaseProcessServiceProcessor<R, I>> processServiceProcessorClass;
        private ProcessServiceResultHandler<I, E> resultHandler;

        public ProcessServiceTemplate(R request,
                                      StreamObserver<E> responseObserver,
                                      Class<? extends BaseProcessServiceProcessor<R, I>> processServiceProcessorClass,
                                      ProcessServiceResultHandler<I, E> resultHandler) {
            if (!(request instanceof Processor.UnaryProcess) && !(request instanceof Processor.BinaryProcess)) {
                throw new IllegalArgumentException("request type error: " + request.getClass().getCanonicalName());
            }

            this.request = request;
            this.responseObserver = responseObserver;
            this.processServiceProcessorClass = processServiceProcessorClass;
            this.resultHandler = resultHandler;

            unaryProcessBuilder = Processor.UnaryProcess.newBuilder();
            binaryProcessBuilder = Processor.BinaryProcess.newBuilder();
            storageLocatorBuilder = StorageBasic.StorageLocator.newBuilder();
            storageLocatorBuilderTemplate = StorageBasic.StorageLocator.newBuilder();
            taskInfoBuilder = Processor.TaskInfo.newBuilder();
        }

        @Override
        public void run() throws Throwable {
            StorageBasic.StorageLocator requestStorageLocator = getStorageLocatorFromRequest(request);
            LOGGER.info("[ROLL][PROCESS][ProcessServiceTemplate] requestStorageLocator: {}",
                    toStringUtils.toOneLineString(requestStorageLocator));
            StoreInfo storeInfo = StoreInfo.fromStorageLocator(requestStorageLocator);
            storageLocatorBuilderTemplate.mergeFrom(requestStorageLocator);

            Dtable dtable = storageMetaClient.getTable(storeInfo);
            if (dtable == null) {
                throw new StorageNotExistsException(storeInfo);
            }

            List<ListenableFuture<I>> resultFutures = Lists.newArrayList();
            final List<I> results = Collections.synchronizedList(Lists.newArrayList());

            Long tableId = dtable.getTableId();
            Map<Long, Node> storageNodeIdToNode = nodeHelper.getNodeIdToStorageNodesOfTable(tableId);
            Map<String, List<Node>> eggTargetToNodes = getEggTargetToNodes();

            final List<Fragment> fragments = storageMetaClient.getFragmentsByTableId(tableId);
            final CountDownLatch finishLatch = new CountDownLatch(fragments.size());
            final List<Throwable> subTaskThrowables = Collections.synchronizedList(Lists.newArrayList());

            for (Fragment fragment : fragments) {
                StoreInfo storeInfoWithFragment = StoreInfo.copy(storeInfo);
                storeInfoWithFragment.setFragment(fragment.getFragmentOrder());

                Long fragmentNodeId = fragment.getNodeId();
                Node storageNode = storageNodeIdToNode.get(fragmentNodeId);

                String target = networkingUtils.getIpOrHost(storageNode);

                LOGGER.info("[ROLL][PROCESS][ProcessServiceTemplate] storeInfoWithFragment: {}, fragmentNodeId: {}, storageNode: {}, target: {}",
                        storeInfoWithFragment, fragmentNodeId, storageNode, target);

                // get egg nodes whose ip is the same as storage

                /*List<Node> eggPossibleNodes = eggTargetToNodes.get(target);
                if (eggPossibleNodes == null || eggPossibleNodes.isEmpty()) {
                    throw new ProcessorStateException("no valid egg for storeInfo: " + storeInfoWithFragment);
                }

                int i = 0;
                if (eggPossibleNodes.size() > 1) {
                    i = randomUtils.nextInt(0, eggPossibleNodes.size());
                }

                Node selectedEggNode = eggPossibleNodes.get(i);*/

                // todo: scheduler: should become a module later
                BasicMeta.Endpoint selectedEggProcessor = nodeHelper.getProcessorEndpoint(target);

                // fill the fragment into the parameter
                R dispatchRequest = buildDispatchRequest(request, fragment);

                // create processor
                BaseProcessServiceProcessor<R, I> processor
                        = rollModelFactory.createBaseProcessServiceProcessor(
                        processServiceProcessorClass, eggProcessServiceClient, dispatchRequest, selectedEggProcessor);

                ListenableFuture<I> resultFuture = asyncThreadPool.submitListenable(processor);
                resultFuture.addCallback(rollModelFactory
                        .createDefaultRollProcessListenableCallback(
                                results, subTaskThrowables, finishLatch, selectedEggProcessor.getIp(), selectedEggProcessor.getPort()));
                resultFutures.add(resultFuture);
            }

            int waitSec = 0;
            while (!finishLatch.await(10, TimeUnit.SECONDS)) {
                LOGGER.info("[ROLL][PROCESS][ProcessServiceTemplate] storeInfo: {}, processType: {}, latch count: {}",
                        storeInfo, processServiceProcessorClass.getSimpleName(),  finishLatch.getCount());
                ++waitSec;
            }

            // LOGGER.info("result size: {}", results.size());
            if (subTaskThrowables.isEmpty()) {                      // no error
                LOGGER.info("[ROLL][PROCESS][ProcessServiceTemplate] valid result. ready to return storeInfo: {}, processType: {}",
                        storeInfo, processServiceProcessorClass.getSimpleName());

                resultHandler.handle(responseObserver, results);
                try {
                    responseObserver.onCompleted();
                } catch (IllegalStateException ignored) {
                    LOGGER.warn("duplicate close from requestObserver");
                }
            } else {                                                // error
                MultipleRuntimeThrowables multipleRuntimeThrowables
                        = new MultipleRuntimeThrowables("error occured in sub tasks. processor type: "
                                + processServiceProcessorClass.getSimpleName(),
                        subTaskThrowables);

                throw multipleRuntimeThrowables;
            }
        }

        private StorageBasic.StorageLocator getStorageLocatorFromRequest(R request) {
            StorageBasic.StorageLocator result = null;

            if (request instanceof Processor.UnaryProcess) {
                result = ((Processor.UnaryProcess) request).getOperand();
            } else if (request instanceof Processor.BinaryProcess) {
                result = ((Processor.BinaryProcess) request).getLeft();
            }

            return result;
        }

        private R buildDispatchRequest(R request, Fragment fragment) {
            R result = null;
            if (request instanceof Processor.UnaryProcess) {
                result = buildUnaryDispatchRequest(request, fragment);
            } else if (request instanceof Processor.BinaryProcess) {
                result = buildBinaryDispatchRequest(request, fragment);
            }

            return result;
        }

        private R buildUnaryDispatchRequest(R request, Fragment fragment) {
            if (!(request instanceof Processor.UnaryProcess)) {
                throw new IllegalArgumentException("request type error: " + request.getClass().getCanonicalName());
            }

            R result = null;
            Processor.UnaryProcess typedRequest = (Processor.UnaryProcess) request;

            storageLocatorBuilder.clear()
                    .mergeFrom(typedRequest.getOperand())
                    .setFragment(fragment.getFragmentOrder());

            unaryProcessBuilder.clear();
            unaryProcessBuilder.mergeFrom(typedRequest)
                    .setOperand(storageLocatorBuilder.build());

            result = (R) unaryProcessBuilder.build();

            return result;
        }

        private R buildBinaryDispatchRequest(R request, Fragment fragment) {
            if (!(request instanceof Processor.BinaryProcess)) {
                throw new IllegalArgumentException("request type error: " + request.getClass().getCanonicalName());
            }

            R result = null;
            Processor.BinaryProcess typedRequest = (Processor.BinaryProcess) request;

            storageLocatorBuilder.clear()
                    .mergeFrom(typedRequest.getLeft())
                    .setFragment(fragment.getFragmentOrder());
            StorageBasic.StorageLocator left = storageLocatorBuilder.build();

            storageLocatorBuilder.clear()
                    .mergeFrom(typedRequest.getRight())
                    .setFragment(fragment.getFragmentOrder());
            StorageBasic.StorageLocator right = storageLocatorBuilder.build();

            binaryProcessBuilder.clear()
                    .mergeFrom(typedRequest)
                    .setLeft(left)
                    .setRight(right);

            result = (R) binaryProcessBuilder.build();

            return result;
        }

    }
}
