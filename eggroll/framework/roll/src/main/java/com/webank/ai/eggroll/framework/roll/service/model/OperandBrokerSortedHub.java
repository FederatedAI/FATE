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

package com.webank.ai.eggroll.framework.roll.service.model;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Queues;
import com.google.common.collect.Sets;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.storage.Kv;
import com.webank.ai.eggroll.core.io.KeyValue;
import com.webank.ai.eggroll.core.model.Bytes;
import com.webank.ai.eggroll.core.serdes.impl.POJOUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.PriorityBlockingQueue;

@Component
@Scope("prototype")
public class OperandBrokerSortedHub extends BaseOperandBrokerHub {
    private Map<Bytes, Kv.Operand> keyToOperand;
    private Map<Bytes, OperandBroker> keyToBranchBroker;
    private PriorityBlockingQueue<Bytes> sortedKeys;
    private static final Logger LOGGER = LogManager.getLogger();
    private Set<OperandBroker> closedBroker;

    public OperandBrokerSortedHub() {
        this.keyToOperand = Maps.newConcurrentMap();
        this.keyToBranchBroker = Maps.newConcurrentMap();
        this.sortedKeys = Queues.newPriorityBlockingQueue();
        this.closedBroker = Sets.newConcurrentHashSet();
    }

    @Override
    public BasicMeta.ReturnStatus call() throws Exception {
        BasicMeta.ReturnStatus result = returnStatusFactory.createSucessful(this.className + " completed");
        try {
            List<Kv.Operand> operands = Lists.newLinkedList();

            LinkedList<Integer> indexToRemove = Lists.newLinkedList();
            Integer index = 0;
            // branchBroker should not be removed during iteration. can be addLast only
            while (!branchBrokers.isEmpty()) {
                index = 0;

                for (OperandBroker curBroker : branchBrokers) {
                    // init
                    operands.clear();
                    indexToRemove.clear();

                    // before check - in case finished is set between drain and next iteration
                    if (curBroker.isClosable()) {
                        indexToRemove.addLast(index);
                        continue;
                    }

                    // get at least one to ensure there is something to be sorted
                    Kv.Operand one = curBroker.get();
                    operands.add(one);

                    curBroker.drainTo(operands);

                    for (Kv.Operand cur : operands) {
                        KeyValue<Bytes, byte[]> kv = POJOUtils.buildKeyValue(cur);
                        Bytes key = kv.key;
                        keyToOperand.put(key, cur);
                        keyToBranchBroker.put(key, curBroker);
                        sortedKeys.add(key);
                    }

                    // after check - if it is closable then there is no need to perform one more iteration
                    if (curBroker.isClosable()) {
                        indexToRemove.addLast(index);
                    }

                    ++index;
                }

                putToMergedBroker();
                Integer offset = 0;
                for (Integer curIndex : indexToRemove) {
                    OperandBroker deletedBroker = branchBrokers.remove(curIndex - offset++);
                    closedBroker.add(deletedBroker);
                }
            }

            mergedBroker.setFinished();
        } catch (Exception e) {
            LOGGER.error("error in OperandBrokerHub: {}", errorUtils.getStackTrace(e));
            result = returnStatusFactory.create(205, "error in operandBrokerHub: " + errorUtils.getStackTrace(e));
            throw e;
        }

        return result;
    }

    private void putToMergedBroker() {
        while (!sortedKeys.isEmpty()) {
            Bytes key = sortedKeys.poll();
            Kv.Operand operand = keyToOperand.remove(key);
            mergedBroker.put(operand);

            keyToBranchBroker.remove(key);
        }
    }
}
