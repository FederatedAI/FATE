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

package com.webank.ai.fate.eggroll.roll.api.grpc.observer.kv.roll;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.fate.core.api.grpc.observer.BaseCalleeRequestStreamObserver;
import com.webank.ai.fate.core.constant.ModelConstants;
import com.webank.ai.fate.core.constant.RuntimeConstants;
import com.webank.ai.fate.core.error.exception.MultipleRuntimeThrowables;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Dtable;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Fragment;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Node;
import com.webank.ai.fate.eggroll.roll.factory.RollModelFactory;
import com.webank.ai.fate.eggroll.roll.helper.NodeHelper;
import com.webank.ai.fate.eggroll.roll.service.async.storage.PutAllProcessor;
import com.webank.ai.fate.eggroll.roll.service.model.OperandBroker;
import com.webank.ai.fate.eggroll.roll.strategy.DispatchPolicy;
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

import javax.annotation.PostConstruct;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

@Component
@Scope("prototype")
public class RollKvPutAllServerRequestStreamObserver extends BaseCalleeRequestStreamObserver<Kv.Operand, Kv.Empty> {
    private static final Logger LOGGER = LogManager.getLogger();
    private final Object fragmentOrderToOperandBrokerLock = new Object();

    @Autowired
    private ThreadPoolTaskExecutor asyncThreadPool;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private DispatchPolicy dispatchPolicy;
    @Autowired
    private RollModelFactory rollModelFactory;
    @Autowired
    private RollServerUtils rollServerUtils;
    @Autowired
    private NodeHelper nodeHelper;

    private final ServerCallStreamObserver<Kv.Empty> serverCallStreamObserver;
    private final StoreInfo storeInfo;
    private final AtomicBoolean wasReady;

    private Map<Long, Node> nodeIdToNodes;
    private Map<Integer, Node> fragmentOrderToNodes;
    private Map<Integer, OperandBroker> fragmentOrderToOperandBroker;
    private List<BasicMeta.ReturnStatus> resultContainer;
    private CountDownLatch eggPutAllFinishLatch;
    private List<Throwable> errorContainer;
    private List<ListenableFuture<BasicMeta.ReturnStatus>> listenableFutures;
    private Set<Integer> finishedFragmentSet;
    private int tableFragmentCount;
    private long totalCount;
    private volatile boolean inited;

    public RollKvPutAllServerRequestStreamObserver(StreamObserver<Kv.Empty> callerNotifier, StoreInfo storeInfo, AtomicBoolean wasReady) {
        super(callerNotifier);
        this.serverCallStreamObserver = (ServerCallStreamObserver<Kv.Empty>) callerNotifier;
        this.storeInfo = storeInfo;
        this.wasReady = wasReady;

        this.fragmentOrderToOperandBroker = Maps.newConcurrentMap();

        this.resultContainer = Collections.synchronizedList(Lists.newLinkedList());
        this.errorContainer = Collections.synchronizedList(Lists.newLinkedList());
        this.listenableFutures = Collections.synchronizedList(Lists.newLinkedList());
        this.finishedFragmentSet = Collections.synchronizedSet(Sets.newConcurrentHashSet());
        this.totalCount = 0;
    }

    @PostConstruct
    public synchronized void init() {
        if (inited) {
            return;
        }
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());
        Dtable dtable = storageMetaClient.getTable(storeInfo);

        long tableId = dtable.getTableId();

        nodeIdToNodes = nodeHelper.getNodeIdToStorageNodesOfTable(tableId);
        List<Fragment> fragments = nodeHelper.getFragmentListOfTable(tableId);
        fragmentOrderToNodes = nodeHelper.getFragmentOrderToStorageNodesOfTable(tableId);

        tableFragmentCount = fragments.size();
        eggPutAllFinishLatch = new CountDownLatch(fragments.size());

        inited = true;
    }

    @Override
    public void onNext(Kv.Operand operand) {
        if (!inited) {
            init();
        }
        // perform dispatch
        int dispatchedFragment = dispatchPolicy.executePolicy(storeInfo, operand.getKey());

        // init
        if (!fragmentOrderToOperandBroker.containsKey(dispatchedFragment)) {
            boolean newlyCreated = false;

            OperandBroker operandBroker = null;
            synchronized (fragmentOrderToOperandBrokerLock) {
                if (!fragmentOrderToOperandBroker.containsKey(dispatchedFragment)) {
                    operandBroker = rollModelFactory.createOperandBroker(500_000);
                    fragmentOrderToOperandBroker.put(dispatchedFragment, operandBroker);
                    newlyCreated = true;
                }
            }

            if (newlyCreated) {
                StoreInfo storeInfoWithFragment = StoreInfo.copy(storeInfo);
                storeInfoWithFragment.setFragment(dispatchedFragment);
                operandBroker = fragmentOrderToOperandBroker.get(dispatchedFragment);

                PutAllProcessor callable
                        = createStoragePutAllRequest(operandBroker, storeInfoWithFragment);

                Node node = fragmentOrderToNodes.get(dispatchedFragment);

                // todo: add error tracking
                ListenableFuture<BasicMeta.ReturnStatus> listenableFuture = asyncThreadPool.submitListenable(callable);
                listenableFuture.addCallback(
                        rollModelFactory.createPutAllProcessorListenableFutureCallback(
                                resultContainer, errorContainer, eggPutAllFinishLatch, node.getIp(), node.getPort(), storeInfoWithFragment, finishedFragmentSet));

                listenableFutures.add(listenableFuture);
            }
        }

        fragmentOrderToOperandBroker.get(dispatchedFragment).put(operand);
        ++totalCount;

        // todo: implement this in framework
        if (serverCallStreamObserver.isReady()) {
            serverCallStreamObserver.request(1);
        } else {
            LOGGER.warn("[PUTALL][SERVER][FLOWCONTROL] not ready");
            wasReady.set(false);
        }
    }

    @Override
    public void onError(Throwable throwable) {
        super.onError(throwable);
        LOGGER.error("[ROLL][KV][PUTALL] put all onError: {}", errorUtils.getStackTrace(throwable));
    }

    @Override
    public void onCompleted() {
        for (Map.Entry<Integer, OperandBroker> entry : fragmentOrderToOperandBroker.entrySet()) {
            entry.getValue().setFinished();
        }

        int dispatchedFragmentCount = fragmentOrderToOperandBroker.size();
        if (dispatchedFragmentCount < tableFragmentCount) {
            LOGGER.info("[ROLL][KV][PUTALL] adjusting lock. dispatchedFragmentCount < tableFragmentCount: {} < {}, storeInfo: {}",
                    dispatchedFragmentCount, tableFragmentCount, storeInfo);
            for (int i = dispatchedFragmentCount; i < tableFragmentCount; ++i) {
                eggPutAllFinishLatch.countDown();
            }
        }

        boolean awaitResult = false;
        try {
            while (!awaitResult) {
                awaitResult = eggPutAllFinishLatch.await(RuntimeConstants.DEFAULT_WAIT_TIME, TimeUnit.SECONDS);

                long currentLatchCount = eggPutAllFinishLatch.getCount();
                long finishedEggCount = tableFragmentCount - currentLatchCount;

                LOGGER.info("[ROLL][KV][PUTALL] waiting put all to finish. storeInfo: {}, current latch count: {}, finished count: {}",
                        storeInfo, currentLatchCount, finishedEggCount);

/*                for (Map.Entry<Integer, OperandBroker> entry : fragmentOrderToOperandBroker.entrySet()) {
                    Integer fragmentOrder = entry.getKey();
                    OperandBroker broker = entry.getValue();
                    // todo: investigate set conflict issue here
                    if (broker.isClosable()) {
                        if (!finishedFragmentSet.contains(fragmentOrder)) {
                            LOGGER.warn("[ROLL][KV][PUTALL] closing unclosed broker for storeInfo: {}, latch count before: {}",
                                    storeInfo, eggPutAllFinishLatch.getCount());
                            finishedFragmentSet.add(fragmentOrder);
                            eggPutAllFinishLatch.countDown();
                        }
                    }
                }*/
            }

            if (awaitResult) {
                if (errorContainer.isEmpty()) {
                    LOGGER.info("[ROLL][PROCESS][PUTALL] put all completed. storeInfo: {}, totalCount: {}",
                            storeInfo, totalCount);
                    callerNotifier.onNext(ModelConstants.EMPTY);
                    super.onCompleted();
                } else {
                    MultipleRuntimeThrowables multipleRuntimeThrowables = new MultipleRuntimeThrowables(
                            "[ROLL][PROCESS][PUTALL] error occured in put all sub tasks", errorContainer);
                    throw multipleRuntimeThrowables;
                }
            } else {
                throw new TimeoutException("[ROLL][PROCESS][PUTALL] put all observer latch wait timeout");
            }
        } catch (Throwable t) {
            onError(t);
        }
    }

    private PutAllProcessor createStoragePutAllRequest(OperandBroker operandBroker, StoreInfo storeInfo) {
        PutAllProcessor result =
                rollModelFactory.createPutAllProcessor(operandBroker, storeInfo, fragmentOrderToNodes.get(storeInfo.getFragment()));

        return result;
    }
}
