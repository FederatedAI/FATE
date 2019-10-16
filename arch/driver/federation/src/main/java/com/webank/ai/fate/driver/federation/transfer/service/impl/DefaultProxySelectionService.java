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

package com.webank.ai.fate.driver.federation.transfer.service.impl;

import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.core.model.NodeType;
import com.webank.ai.eggroll.core.utils.ToStringUtils;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.fate.driver.federation.transfer.service.ProxySelectionService;
import com.webank.ai.fate.driver.federation.utils.FederationServerUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.Node;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service("proxySelectionService")
public class DefaultProxySelectionService implements ProxySelectionService {
    @Autowired
    private FederationServerUtils federationServerUtils;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private TypeConversionUtils typeConversionUtils;
    @Autowired
    private ToStringUtils toStringUtils;

    private List<BasicMeta.Endpoint> proxies;

    private static final Logger LOGGER = LogManager.getLogger();

    public DefaultProxySelectionService() {
        proxies = Lists.newArrayList();
    }

    public synchronized void loadProxy() {
        LOGGER.info("[FEDERATION][PROXY][SELECTION] loading");
        storageMetaClient.init(federationServerUtils.getMetaServiceEndpoint());

        proxies.clear();

        Node node = new Node();
        node.setType(NodeType.PROXY.name());
        node.setStatus(NodeStatus.HEALTHY.name());

        List<Node> nodes = storageMetaClient.getNodes(node);

        BasicMeta.Endpoint endpoint = null;
        for (Node cur : nodes) {
            endpoint = typeConversionUtils.toEndpoint(cur);
            proxies.add(endpoint);
        }
    }

    @Override
    public BasicMeta.Endpoint select() {
        if (proxies.isEmpty()) {
            loadProxy();
        }
        // todo: make this more flexible

        BasicMeta.Endpoint result = proxies.get(0);
        LOGGER.info("[FEDERATION][PROXY][SELECTION] result: {}", toStringUtils.toOneLineString(result));

        return result;
    }
}
