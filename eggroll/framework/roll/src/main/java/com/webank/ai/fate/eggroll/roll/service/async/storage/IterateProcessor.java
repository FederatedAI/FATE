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

package com.webank.ai.fate.eggroll.roll.service.async.storage;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.webank.ai.fate.api.eggroll.storage.Kv;
import com.webank.ai.fate.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.fate.core.constant.RuntimeConstants;
import com.webank.ai.fate.core.io.StoreInfo;
import com.webank.ai.fate.core.model.Bytes;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Dtable;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Fragment;
import com.webank.ai.fate.eggroll.meta.service.dao.generated.model.Node;
import com.webank.ai.fate.eggroll.roll.api.grpc.client.StorageServiceClient;
import com.webank.ai.fate.eggroll.roll.helper.NodeHelper;
import com.webank.ai.fate.eggroll.roll.service.model.OperandBroker;
import com.webank.ai.fate.eggroll.roll.util.RollServerUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.concurrent.GuardedBy;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

@Component
@Scope("prototype")
public class IterateProcessor implements Callable<OperandBroker> {
    private static final long DEFAULT_MIN_CHUNK_SIZE = 4 << 20;
    private static final long DEFAULT_MAX_CHUNK_SIZE = 64 << 20;
    private static final Logger LOGGER = LogManager.getLogger();

    private final OperandBroker result;
    private final Kv.Range range;
    private final StoreInfo storeInfo;

    @Autowired
    private StorageServiceClient storageServiceClient;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private RollServerUtils rollServerUtils;
    @Autowired
    private NodeHelper nodeHelper;

    private final Object eggBrokersLock;
    private final Object isEggFinishedLock;
    private final Object eggRangesLock;

    private int[] loserTree;
    private @GuardedBy("eggBrokersLock") ArrayList<OperandBroker> eggBrokers;
    private @GuardedBy("isEggFinishedLock") ArrayList<Boolean> isEggFinished;
    private @GuardedBy("eggRangesLock") ArrayList<Kv.Range> eggRanges;
    private AtomicInteger eggFinishedCount;
    private List<Fragment> fragments;
    private Map<Long, Node> nodeIdToNodes;
    private ArrayList<StoreInfo> storeInfosWithFragments;
    private int totalFragments;
    private Dtable dtable;
    private long curChunkSize;
    private long minChunkSize;

    public IterateProcessor(Kv.Range range, StoreInfo storeInfo, OperandBroker operandBroker) {
        this.range = range;
        this.storeInfo = storeInfo;
        this.result = operandBroker;

        this.eggFinishedCount = new AtomicInteger(0);

        this.curChunkSize = 0;
        minChunkSize = range.getMinChunkSize();
        if (minChunkSize < 0) {
           minChunkSize = Long.MAX_VALUE;
        }

        this.eggBrokersLock = new Object();
        this.isEggFinishedLock = new Object();
        this.eggRangesLock = new Object();
    }

    @PostConstruct
    public void init() {
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());
    }

    @Override
    public OperandBroker call() throws Exception {
        dtable = storageMetaClient.getTable(storeInfo);

        long tableId = dtable.getTableId();

        nodeIdToNodes = nodeHelper.getNodeIdToStorageNodesOfTable(tableId);

        fragments = nodeHelper.getFragmentListOfTable(tableId);
        totalFragments = fragments.size();

        if (minChunkSize == 0) {
            minChunkSize = Math.max(totalFragments * ((1 << 20) + (768 << 10)), DEFAULT_MIN_CHUNK_SIZE);
            minChunkSize = Math.min(minChunkSize, DEFAULT_MAX_CHUNK_SIZE);
        }

        LOGGER.info("[ROLL][ITERATOR][PROCESSOR] final minChunkSize: {}", minChunkSize);

        storeInfosWithFragments = Lists.newArrayListWithCapacity(totalFragments);
        synchronized (isEggFinishedLock) {
            isEggFinished = Lists.newArrayListWithCapacity(totalFragments);
        }

        synchronized (eggBrokersLock) {
            eggBrokers = Lists.newArrayListWithCapacity(totalFragments);
        }

        synchronized (eggRangesLock) {
            eggRanges = Lists.newArrayListWithCapacity(totalFragments);
        }

        // construct and init loser tree
        loserTree = new int[totalFragments];
        Arrays.fill(loserTree, -1);

        // data preparation
        for (int i = 0; i < totalFragments; ++i) {
            synchronized (isEggFinishedLock) {
                isEggFinished.add(false);
            }

            storeInfosWithFragments.add(null);

            synchronized (eggBrokersLock) {
                eggBrokers.add(null);
            }

            synchronized (eggRangesLock) {
                eggRanges.add(range);
            }

            refillBroker(i);
        }

        if (eggFinishedCount.get() >= totalFragments) {
            result.setFinished();
            return result;
        }

        // init adjust
        for (int i = totalFragments - 1; i >= 0; --i) {
            adjust(i);
        }

        // adjust until length exceeds minChunkSize
        int curSortedIndex = -1;
        int keySize = 0;
        int valueSize = 0;
        OperandBroker curSortedBroker = null;
        Kv.Operand curSortedOperand = null;

        while (curChunkSize < minChunkSize && eggFinishedCount.get() < totalFragments) {
            curSortedIndex = loserTree[0];
            synchronized (eggBrokersLock) {
                curSortedBroker = eggBrokers.get(curSortedIndex);
            }
            while (!curSortedBroker.isReady()) {
                curSortedBroker.awaitLatch(1, TimeUnit.SECONDS);
                LOGGER.info("[ROLL][KV][ITERATE] waiting to get. size: {}, storeInfo: {}", curSortedBroker.getQueueSize(), storeInfo);

                if (curSortedBroker.isClosable()) {
                    if (eggFinishedCount.get() >= totalFragments) {
                        LOGGER.info("[ROLL][KV][ITERATE] all finished. storeInfo: {}", storeInfo);
                        result.setFinished();

                        return result;
                    }

                    LOGGER.info("[ROLL][KV][ITERATE] closable not not hit limit. storeInfo: {}", storeInfo);
                    adjust(curSortedIndex);
                    break;
                }
            }

            curSortedOperand = curSortedBroker.get();

            // breaks when null occurs because it is the maximum value
            if (curSortedOperand == null) {
                LOGGER.info("[ROLL][KV][ITERATE] null value occured. storeInfo: {}", storeInfo);

                break;
            }

            keySize = curSortedOperand.getKey() == null ? 0 : curSortedOperand.getKey().size();
            valueSize = curSortedOperand.getValue() == null ? 0 : curSortedOperand.getValue().size();

            curChunkSize += keySize + valueSize;
            result.put(curSortedOperand);

            // if closable after get, then update range info
            if (curSortedBroker.isClosable()) {
                synchronized (eggRangesLock) {
                    Kv.Range lastRange = eggRanges.get(curSortedIndex);
                    eggRanges.set(curSortedIndex, lastRange.toBuilder().setStart(curSortedOperand.getKey()).build());
                }
            }
            adjust(curSortedIndex);
        }

        result.setFinished();

        return result;
    }

    private OperandBroker refillBroker(int fragmentOrder) {
        synchronized (isEggFinishedLock) {
            if (isEggFinished.get(fragmentOrder)) {
                return null;
            }
        }
        Preconditions.checkArgument(fragmentOrder >= 0 && fragmentOrder < totalFragments,
                "fragmentOrder must >= 0 and < totalFragments");

        StoreInfo storeInfoWithFragment = storeInfosWithFragments.get(fragmentOrder);
        if (storeInfoWithFragment == null) {
            storeInfoWithFragment = StoreInfo.copy(storeInfo);
            storeInfoWithFragment.setFragment(fragmentOrder);
            storeInfosWithFragments.add(fragmentOrder, storeInfoWithFragment);
        }

        Map<Integer, Fragment> fragmentOrderToFragment = Maps.newConcurrentMap();
        for (Fragment fragment : fragments) {
            fragmentOrderToFragment.put(fragment.getFragmentOrder(), fragment);
        }

        Fragment fragment = fragmentOrderToFragment.get(fragmentOrder);
        if (fragment == null) {
            throw new IllegalStateException("fragment is null. should not get here. fragment: " + storeInfoWithFragment);
        }

        Node fragmentToNode = nodeIdToNodes.get(fragment.getNodeId());
        if (fragmentToNode == null) {
            throw new IllegalStateException("fragmentToNode is null. should not get here. fragment: " + storeInfoWithFragment);
        }
        Kv.Range range = null;
        synchronized (eggRangesLock) {
             range = eggRanges.get(fragmentOrder);
        }
        OperandBroker result = storageServiceClient.iterate(range, storeInfoWithFragment, fragmentToNode);

        try {
            boolean awaitResult = false;
            while (!awaitResult) {
                LOGGER.info("[ROLL][KV][ITERATE][PROCESSOR] waiting latch for: {}", storeInfoWithFragment);

                awaitResult = result.awaitLatch(500, TimeUnit.MILLISECONDS);

                if (result.isReady() || result.isClosable()) {
                    LOGGER.info("[ROLL][KV][ITERATE][PROCESSOR] broker: {}, closable: {}, ready: {}",
                            storeInfoWithFragment, result.isClosable(), result.isReady());
                    break;
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        }
        LOGGER.info("[ROLL][KV][ITERATE][PROCESSOR] data arrived. size: {}, fragment order: {}, node address: {}:{}, range start: {}, range end: {}",
                result.getQueueSize(), fragmentOrder, fragmentToNode.getIp(), fragmentToNode.getPort(), range.getStart().toStringUtf8(), range.getEnd().toStringUtf8());
        if (result.isClosable()) {
            synchronized (isEggFinishedLock) {
                isEggFinished.set(fragmentOrder, true);
            }
            eggFinishedCount.incrementAndGet();
            result = null;
        }

        synchronized (eggBrokersLock) {
            OperandBroker oldBroker = eggBrokers.set(fragmentOrder, result);
            if (oldBroker != null && !oldBroker.isClosable()) {
                LOGGER.warn("[ROLL][KV][ITERATE] removing old broker which is not closable yet. tableId: {}, fragmentOrder: {}, nodeId: {}, node address: {}:{}",
                        dtable.getTableId(), fragmentOrder, fragmentToNode.getNodeId(), fragmentToNode.getIp(), fragmentToNode.getPort());
            }
        }

        return result;
    }

    private void adjust(int curIndex) {
        Preconditions.checkArgument(curIndex >= 0 && curIndex < totalFragments,
                "curIndex must >= 0 and < totalFragments");

        for (int parentIndex = (curIndex + totalFragments) >> 1; parentIndex > 0; parentIndex >>= 1) {
            if (curIndex < 0) {
                return;
            }
            Kv.Operand curOperand = peekOperand(curIndex);
            Kv.Operand parentOperand = peekOperand(loserTree[parentIndex]);

            Bytes curKey = curOperand == null ? null : Bytes.wrap(curOperand.getKey());
            Bytes parentKey = parentOperand == null ? null : Bytes.wrap(parentOperand.getKey());

            /* records loser (key with less value wins):
             *
             * 1. if curKey is null, lose
             * 2. parent is not set yet, set it (for init occasion)
             * 3. parent is set but parent < cur, parent wins. switch current and parent
             *
             * After init, loserTree[0] has the minimum value, but other nodes may not fulfill the condition
             * where parent < child.
             * But if any adjust on those branches occur, the condition above will be true.
             *
             * We can always get the minimum value in loserTree[0], as the 'finale' always occurs during and after init.
             */
            if (curKey == null || (loserTree[parentIndex] == -1 || (parentKey != null && curKey.compareTo(parentKey) > 0))) {
                loserTree[parentIndex] ^= curIndex;
                curIndex ^= loserTree[parentIndex];
                loserTree[parentIndex] ^= curIndex;
            }
        }

        loserTree[0] = curIndex;
    }

    private Kv.Operand peekOperand(int index) {
        Kv.Operand result = null;

        if (index < 0) {
            return result;
        }

        while (result == null) {
            OperandBroker curBroker = null;

            synchronized (eggBrokersLock) {
                curBroker = eggBrokers.get(index);
            }

            if (curBroker == null || curBroker.isClosable()) {
                curBroker = refillBroker(index);
                if (curBroker == null) {
                    break;
                }
            }
            result = curBroker.peek();
        }

        return result;
    }
}
