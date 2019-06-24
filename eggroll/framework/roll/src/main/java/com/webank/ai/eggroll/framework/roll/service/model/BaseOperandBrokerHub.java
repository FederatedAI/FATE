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
import com.webank.ai.eggroll.core.factory.ReturnStatusFactory;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.framework.roll.factory.RollModelFactory;
import org.springframework.beans.factory.annotation.Autowired;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.concurrent.Callable;

public abstract class BaseOperandBrokerHub implements Callable<BasicMeta.ReturnStatus> {
    protected final String className;
    @Autowired
    protected RollModelFactory rollModelFactory;
    @Autowired
    protected ReturnStatusFactory returnStatusFactory;
    @Autowired
    protected ErrorUtils errorUtils;
    protected List<OperandBroker> branchBrokers;
    protected OperandBroker mergedBroker;

    public BaseOperandBrokerHub() {
        this.branchBrokers = Lists.newArrayList();
        this.className = this.getClass().getSimpleName();
    }

    @PostConstruct
    public void init() {
        this.mergedBroker = rollModelFactory.createOperandBroker();
    }

    public void addBranchBroker(OperandBroker operandBroker) {
        Preconditions.checkNotNull(operandBroker, "branch broker cannot be null");
        this.branchBrokers.add(operandBroker);
    }

    public OperandBroker getMergedBroker() {
        return mergedBroker;
    }
}
