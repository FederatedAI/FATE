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

package com.webank.ai.fate.eggroll.roll.api.grpc.server;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.eggroll.storage.KVServiceGrpc;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.api.eggroll.storage.StorageBasic;
import com.webank.ai.fate.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.fate.core.api.grpc.server.GrpcServerWrapper;
import com.webank.ai.fate.core.constant.ModelConstants;
import com.webank.ai.fate.core.constant.StringConstants;
import com.webank.ai.fate.core.error.exception.CrudException;
import com.webank.ai.fate.core.error.exception.MultipleRuntimeThrowables;
import com.webank.ai.fate.core.error.exception.StorageNotExistsException;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.core.model.DtableStatus;
import com.webank.ai.fate.core.model.FragmentStatus;
import com.webank.ai.fate.core.utils.ErrorUtils;
import com.webank.ai.fate.core.utils.ToStringUtils;
import com.webank.ai.fate.core.utils.TypeConversionUtils;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Dtable;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Fragment;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Node;
import com.webank.ai.fate.eggroll.roll.api.grpc.client.StorageServiceClient;
import com.webank.ai.fate.eggroll.roll.api.grpc.observer.kv.roll.RollKvPutAllServerRequestStreamObserver;
import com.webank.ai.fate.eggroll.roll.factory.DispatchPolicyFactory;
import com.webank.ai.fate.eggroll.roll.factory.DispatcherFactory;
import com.webank.ai.fate.eggroll.roll.factory.RollGrpcObserverFactory;
import com.webank.ai.fate.eggroll.roll.factory.RollModelFactory;
import com.webank.ai.fate.eggroll.roll.helper.NodeHelper;
import com.webank.ai.fate.eggroll.roll.service.async.storage.CountProcessor;
import com.webank.ai.fate.eggroll.roll.service.async.storage.IterateProcessor;
import com.webank.ai.fate.eggroll.roll.service.model.DispatchResult;
import com.webank.ai.fate.eggroll.roll.service.model.OperandBroker;
import com.webank.ai.fate.eggroll.roll.strategy.DispatchPolicy;
import com.webank.ai.fate.eggroll.roll.strategy.Dispatcher;
import com.webank.ai.fate.eggroll.roll.util.RollServerUtils;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.springframework.util.concurrent.ListenableFuture;
import org.springframework.util.concurrent.ListenableFutureCallback;

import javax.annotation.PostConstruct;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

@Component
@Scope("prototype")
public class RollKvServiceImpl extends KVServiceGrpc.KVServiceImplBase {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private StorageServiceClient storageServiceClient;
    @Autowired
    private TypeConversionUtils typeConversionUtils;
    @Autowired
    private ToStringUtils toStringUtils;
    @Autowired
    private ErrorUtils errorUtils;
    @Autowired
    private ThreadPoolTaskExecutor asyncThreadPool;
    @Autowired
    private GrpcServerWrapper grpcServerWrapper;
    @Autowired
    private DispatcherFactory dispatcherFactory;
    @Autowired
    private DispatchPolicyFactory dispatchPolicyFactory;
    @Autowired
    private RollGrpcObserverFactory rollGrpcObserverFactory;
    @Autowired
    private RollModelFactory rollModelFactory;
    @Autowired
    private RollServerUtils rollServerUtils;
    @Autowired
    private NodeHelper nodeHelper;

    @PostConstruct
    public void init() {
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());
    }

    @Override
    public synchronized void createIfAbsent(Kv.CreateTableInfo request, StreamObserver<Kv.CreateTableInfo> responseObserver) {
        LOGGER.info("Kv.createIfAbsent request received. request: {}", toStringUtils.toOneLineString(request));
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StorageBasic.StorageLocator storageLocator = request.getStorageLocator();

            List<Fragment> fragments = null;
            Dtable createResult = null;
            Dtable createTemplate = storageMetaClient.getTable(storageLocator.getNamespace(), storageLocator.getName());

            // todo: add transaction control
            if (createTemplate == null) {
                createTemplate = typeConversionUtils.toDtable(request);

                createResult = storageMetaClient.createTable(createTemplate);

                if (createResult != null) {
                    fragments = storageMetaClient.createFragmentsForTable(createResult);
                }
            } else {
                fragments = storageMetaClient.getFragmentsByTableId(createTemplate.getTableId());
                createResult = createTemplate;
            }

            Kv.CreateTableInfo result = null;
            // todo: add more result check
            if (!fragments.isEmpty()) {
                // createResult = storageMetaClient.createTable(createTemplate);
                result = typeConversionUtils.toCreateTableInfo(createResult);
            }

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void put(Kv.Operand request, StreamObserver<Kv.Empty> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.put request received. storeInfo: {}", toStringUtils.toOneLineString(storeInfo));

            DispatchResult dispatchResult = dispatchInternal(storeInfo, request.getKey());
            storageServiceClient.put(request, dispatchResult.getStoreInfo(), dispatchResult.getNode());

            responseObserver.onNext(ModelConstants.EMPTY);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void putIfAbsent(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.putIfAbsent request received. storeInfo: {}", storeInfo);

            DispatchResult dispatchResult = dispatchInternal(storeInfo, request.getKey());
            Kv.Operand result = storageServiceClient.putIfAbsent(request, dispatchResult.getStoreInfo(), dispatchResult.getNode());

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public StreamObserver<Kv.Operand> putAll(StreamObserver<Kv.Empty> responseObserver) {
        StoreInfo storeInfo = StoreInfo.fromGrpcContext();
        LOGGER.info("Kv.putAll request received: {}", toStringUtils.toOneLineString(storeInfo));

        final ServerCallStreamObserver<Kv.Empty> serverCallStreamObserver
                = (ServerCallStreamObserver<Kv.Empty>) responseObserver;
        serverCallStreamObserver.disableAutoInboundFlowControl();

        final AtomicBoolean wasReady = new AtomicBoolean(false);

        serverCallStreamObserver.setOnReadyHandler(() -> {
            if (serverCallStreamObserver.isReady() && wasReady.compareAndSet(false, true)) {
                serverCallStreamObserver.request(1);
            }
        });

        RollKvPutAllServerRequestStreamObserver requestObserver
                = rollGrpcObserverFactory.createRollKvPutAllServerRequestStreamObserver(responseObserver, storeInfo, wasReady);

        return requestObserver;
    }

    @Override
    public void delOne(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.delete request received. storeInfo: {}", toStringUtils.toOneLineString(storeInfo));

            DispatchResult dispatchResult = dispatchInternal(storeInfo, request.getKey());
            Kv.Operand result = storageServiceClient.delete(request, dispatchResult.getStoreInfo(), dispatchResult.getNode());

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void get(Kv.Operand request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.get request received. storeInfo: {}", toStringUtils.toOneLineString(storeInfo));

            DispatchResult dispatchResult = dispatchInternal(storeInfo, request.getKey());
            Kv.Operand result = storageServiceClient.get(request, dispatchResult.getStoreInfo(), dispatchResult.getNode());

            responseObserver.onNext(result);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void iterate(Kv.Range request, StreamObserver<Kv.Operand> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.iterate request received: {}", toStringUtils.toOneLineString(storeInfo));
            OperandBroker sortedBroker = rollModelFactory.createOperandBroker();

            final List<Throwable> errorContainer = Collections.synchronizedList(Lists.newLinkedList());
            IterateProcessor iterateProcessor
                    = rollModelFactory.createIterateProcessor(request, storeInfo, sortedBroker);

            ListenableFuture<OperandBroker> iterateProcessorListenableFuture
                    = asyncThreadPool.submitListenable(iterateProcessor);
            iterateProcessorListenableFuture.addCallback(new ListenableFutureCallback<OperandBroker>() {
                @Override
                public void onFailure(Throwable throwable) {
                    LOGGER.error("[ROLL][KV][ITERATE] error in iterate processor: {}", errorUtils.getStackTrace(throwable));
                    sortedBroker.setFinished();
                    errorContainer.add(throwable);
                }

                @Override
                public void onSuccess(OperandBroker operandBroker) {
                    LOGGER.info("[ROLL][KV][ITERATE] finished without error. storeInfo: {}, request: {}",
                            storeInfo, toStringUtils.toOneLineString(request));
                    sortedBroker.setFinished();
                }
            });

            int totalIterated = 0;
            List<Kv.Operand> sortedOperands = Lists.newLinkedList();
            int spinCount = 0;
            while (!sortedBroker.isClosable()) {
                sortedBroker.awaitLatch(50, TimeUnit.MILLISECONDS);

                if (sortedBroker.isReady()) {
                    sortedOperands.clear();
                    totalIterated += sortedBroker.drainTo(sortedOperands);

                    for (Kv.Operand next : sortedOperands) {
                        responseObserver.onNext(next);
                    }
                    spinCount = 0;
                } else {
                    ++spinCount;
                    if (spinCount == 1000) {
                        LOGGER.info("[ROLL][KV][ITERATE] waiting for sortedBroker to finish storeInfo: {}. closable: {}, queueSize: {}, finished: {}, peek: {}",
                                storeInfo, sortedBroker.isClosable(), sortedBroker.getQueueSize(), sortedBroker.isFinished(), sortedBroker.peek());
                        spinCount = 0;
                    }
                }
            }

            if (!errorContainer.isEmpty()) {
                throw new MultipleRuntimeThrowables("[ROLL][KV][ITERATE] error in iterate. storeInfo: " + storeInfo, errorContainer);
            }

            LOGGER.info("[ROLL][KV][ITERATE] roll iterate successfully. totalIterated: {}, storeInfo: {}, request: {}",
                    totalIterated, storeInfo, toStringUtils.toOneLineString(request));
            responseObserver.onCompleted();
        });
    }

    @Override
    public void destroy(Kv.Empty request, StreamObserver<Kv.Empty> responseObserver) {
        LOGGER.info("Kv.destroy request received");
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();

            Dtable dtable = storageMetaClient.getTable(storeInfo.getNameSpace(), storeInfo.getTableName());

            if (dtable != null && DtableStatus.NORMAL.name().equals(dtable.getStatus())) {
                Map<Long, Node> nodeIdToNode = nodeHelper.getNodeIdToStorageNodesOfTable(dtable.getTableId());
                List<Fragment> fragments = storageMetaClient.getFragmentsByTableId(dtable.getTableId());

                // destroy all fragments in all nodes
                for (Fragment fragment : fragments) {
                    fragment.setStatus(FragmentStatus.DELETED.name());
                    storageMetaClient.updateFragment(fragment);

                    Node node = nodeIdToNode.get(fragment.getNodeId());
                    storageServiceClient.destroy(request, storeInfo, node);
                }

                // update metadata
                dtable.setStatus(DtableStatus.DELETED.name());
                dtable.setTableName(dtable.getTableName() + StringConstants.DASH + System.currentTimeMillis());
                Dtable result = storageMetaClient.updateTable(dtable);

                if (result == null) {
                    throw new CrudException(103, "Failed to destroy table: " + storeInfo);
                }
            }

            responseObserver.onNext(ModelConstants.EMPTY);
            responseObserver.onCompleted();
        });
    }

    // todo: eliminate duplicate codes
    @Override
    public void destroyAll(Kv.Empty request, StreamObserver<Kv.Empty> responseObserver) {
        // LOGGER.info("Kv.destroyAll request received");

        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.destroyAll request received. storeInfo: {}", storeInfo);

            List<Dtable> dtables = storageMetaClient.getTables(storeInfo);
            List<Dtable> destroyedDtables = Lists.newArrayListWithExpectedSize(dtables.size());

            for (Dtable dtable : dtables) {
                if (dtable != null && DtableStatus.NORMAL.name().equals(dtable.getStatus())) {
                    Map<Long, Node> nodeIdToNode = nodeHelper.getNodeIdToStorageNodesOfTable(dtable.getTableId());

                    List<Fragment> fragments = storageMetaClient.getFragmentsByTableId(dtable.getTableId());

                    StoreInfo storeInfoWithExactTableName = StoreInfo.fromDtable(dtable);

                    // destroy all fragments in all nodes
                    for (Fragment fragment : fragments) {
                        fragment.setStatus(FragmentStatus.DELETED.name());
                        storageMetaClient.updateFragment(fragment);

                        Node node = nodeIdToNode.get(fragment.getNodeId());
                        if (node == null) {
                            continue;
                        }

                        StoreInfo storeInfoWithExactTableNameAndFragment = StoreInfo.copy(storeInfoWithExactTableName);
                        storeInfoWithExactTableNameAndFragment.setFragment(fragment.getFragmentOrder());
                        storageServiceClient.destroy(request, storeInfoWithExactTableNameAndFragment, node);
                    }

                    // update metadata
                    dtable.setStatus(DtableStatus.DELETED.name());
                    dtable.setTableName(dtable.getTableName() + StringConstants.DASH + System.currentTimeMillis());
                    Dtable result = storageMetaClient.updateTable(dtable);

                    if (result == null) {
                        throw new CrudException(103, "Failed to destroy table: " + storeInfo);
                    } else {
                        destroyedDtables.add(dtable);
                    }
                }
            }

            LOGGER.info("Kv.destroyAll result: {}", destroyedDtables);

            responseObserver.onNext(ModelConstants.EMPTY);
            responseObserver.onCompleted();
        });
    }

    @Override
    public void count(Kv.Empty request, StreamObserver<Kv.Count> responseObserver) {
        grpcServerWrapper.wrapGrpcServerRunnable(responseObserver, () -> {
            long result = 0L;
            StoreInfo storeInfo = StoreInfo.fromGrpcContext();
            LOGGER.info("Kv.count request received. storeInfo: {}", storeInfo);

            Dtable dtable = storageMetaClient.getTable(storeInfo.getNameSpace(), storeInfo.getTableName());
            if (dtable != null && DtableStatus.NORMAL.name().equals(dtable.getStatus())) {
                List<Fragment> fragments = storageMetaClient.getFragmentsByTableId(dtable.getTableId());

                List<Node> healthyNodes = storageMetaClient.getStorageNodesByTableId(dtable.getTableId());
                Map<Long, Node> nodeIdToNode = Maps.newHashMap();

                for (Node node : healthyNodes) {
                    nodeIdToNode.put(node.getNodeId(), node);
                }

                AtomicLong countValueResult = new AtomicLong(0);
                List<Throwable> throwables = Collections.synchronizedList(Lists.newLinkedList());
                CountDownLatch finishLatch = new CountDownLatch(fragments.size());

                for (Fragment fragment : fragments) {
                    Node node = nodeIdToNode.get(fragment.getNodeId());
                    StoreInfo storeInfoWithFragment = StoreInfo.copy(storeInfo);
                    storeInfoWithFragment.setFragment(fragment.getFragmentOrder());

                    CountProcessor countProcessor = rollModelFactory.createCountProcessor(request, storeInfoWithFragment, node);
                    ListenableFuture<Kv.Count> countListenableFuture = asyncThreadPool.submitListenable(countProcessor);
                    countListenableFuture.addCallback(
                            rollModelFactory.createCountProcessListenableFutureCallback(
                                    countValueResult, throwables, finishLatch, node.getIp(), node.getPort()));
                }

                boolean awaitResult = false;
                while (!awaitResult) {
                    awaitResult = finishLatch.await(1, TimeUnit.SECONDS);
                }

                if (!throwables.isEmpty()) {
                    throw new MultipleRuntimeThrowables("error in getting counts.", throwables);
                } else {
                    result = countValueResult.get();
                }
            }
            Kv.Count countResult = Kv.Count.newBuilder().setValue(result).build();
            LOGGER.info("[ROLL][COUNT][EGG] result: {}", result);
            responseObserver.onNext(countResult);
            responseObserver.onCompleted();
        });
    }

    private DispatchResult dispatchInternal(StoreInfo storeInfo, ByteString dataKey) {
        Dtable dtable = storageMetaClient.getTable(storeInfo.getNameSpace(), storeInfo.getTableName());
        if (dtable == null) {
            throw new StorageNotExistsException(storeInfo);
        }

        Dispatcher dispatcher = dispatcherFactory.createDispatcher(dtable.getDispatcher());
        DispatchPolicy dispatchPolicy = dispatchPolicyFactory.createDefaultModDispatchPolicy();

        int fragmentResult = dispatchPolicy.executePolicy(storeInfo, dataKey);
        Node nodeResult = dispatcher.dispatch(storeInfo, dataKey);

        return dispatcherFactory.createDispatchResult(nodeResult, storeInfo, fragmentResult);
    }
}
