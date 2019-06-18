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

import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.Dispatchers;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.service.model.DispatchResult;
import com.webank.ai.eggroll.framework.roll.strategy.DispatchPolicy;
import com.webank.ai.eggroll.framework.roll.strategy.Dispatcher;
import com.webank.ai.eggroll.framework.roll.strategy.impl.DefaultDispatcher;
import com.webank.ai.eggroll.framework.roll.strategy.impl.DefaultModDispatchPolicy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

@Component
public class DispatcherFactory {
    private static final String DEFAULT = "DEFAULT";
    @Autowired
    private ApplicationContext applicationContext;

    public Dispatcher createDispatcher(String dispatcherName) {
        Dispatcher result = null;
        DispatchPolicy dispatchPolicy = null;

        if (Dispatchers.MOD.name().equals(dispatcherName) || DEFAULT.equals(dispatcherName)) {
            dispatchPolicy = applicationContext.getBean(DefaultModDispatchPolicy.class);
        }
        result = applicationContext.getBean(DefaultDispatcher.class, dispatchPolicy);

        return result;
    }

    public DispatchResult createDispatchResult(Node node, StoreInfo storeInfo) {
        return applicationContext.getBean(DispatchResult.class, node, storeInfo);
    }

    public DispatchResult createDispatchResult(Node node, StoreInfo original, int fragment) {
        return applicationContext.getBean(DispatchResult.class, node, original, fragment);
    }
}
