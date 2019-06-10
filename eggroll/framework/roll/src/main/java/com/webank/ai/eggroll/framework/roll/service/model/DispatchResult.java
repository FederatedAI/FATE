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

import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.concurrent.Immutable;

@Component
@Scope("prototype")
@Immutable
public class DispatchResult {
    private final Node node;
    private final StoreInfo storeInfo;

    public DispatchResult(Node node, StoreInfo storeInfo) {
        this.node = node;
        this.storeInfo = storeInfo;
    }

    public DispatchResult(Node node, StoreInfo original, int fragment) {
        this.node = node;
        storeInfo = StoreInfo.copy(original);
        storeInfo.setFragment(fragment);
    }

    public Node getNode() {
        return node;
    }

    public StoreInfo getStoreInfo() {
        return storeInfo;
    }
}
