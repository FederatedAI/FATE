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

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.api.storage.Kv;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.LinkedList;
import java.util.List;

@Component
@Scope("prototype")
public class OperandBrokerUnSortedHub extends BaseOperandBrokerHub {
    private static final Logger LOGGER = LogManager.getLogger();

    @Override
    public BasicMeta.ReturnStatus call() throws Exception {
        BasicMeta.ReturnStatus result = returnStatusFactory.createSucessful(this.className + " completed");
        LinkedList<Integer> finishedBroker = Lists.newLinkedList();
        List<Kv.Operand> operands = Lists.newLinkedList();
        int index = 0;

        LOGGER.info("branch broker size: {}", branchBrokers.size());

        try {
            while (finishedBroker.size() < branchBrokers.size()) {
                index = 0;
                // todo: change to pub/sub mode
                for (OperandBroker curBroker : branchBrokers) {
                    Preconditions.checkNotNull(curBroker, "curBroker is null");
                    operands.clear();

                    curBroker.drainTo(operands);
                    // LOGGER.info("[UNSORTEDHUB]drainTo size: {}", operands.size());
                    mergedBroker.addAll(operands);

                    if (curBroker.isFinished()) {
                        finishedBroker.add(index);
                    }

                    ++index;
                }
            }

            mergedBroker.setFinished();
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
            throw e;
        }

        return result;
    }
}
