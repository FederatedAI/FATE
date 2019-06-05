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

package com.webank.ai.eggroll.framework.roll.helper;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Maps;
import com.webank.ai.eggroll.api.core.BasicMeta;
import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.utils.TypeConversionUtils;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.api.grpc.client.EggNodeManagerClient;
import com.webank.ai.eggroll.framework.roll.util.RollServerUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Component
@Scope("prototype")
public class NodeHelper {
    @Autowired
    private RollServerUtils rollServerUtils;
    @Autowired
    private StorageMetaClient storageMetaClient;
    @Autowired
    private TypeConversionUtils typeConversionUtils;
    @Autowired
    private EggNodeManagerClient eggNodeManagerClient;

    private LoadingCache<Long, Map<Long, Node>> nodeIdToStorageNodeCache;
    private LoadingCache<Long, List<Fragment>> fragmentsCache;
    private LoadingCache<String, Node> ipToNodeManager;

    @PostConstruct
    public void init() {
        storageMetaClient.init(rollServerUtils.getMetaServiceEndpoint());

        nodeIdToStorageNodeCache = CacheBuilder.newBuilder()
                .expireAfterWrite(5, TimeUnit.MINUTES)
                .softValues()
                .build(new CacheLoader<Long, Map<Long, Node>>() {
                    @Override
                    public Map<Long, Node> load(Long tableId) throws Exception {
                        Map<Long, Node> result = Maps.newConcurrentMap();
                        List<Node> nodes = storageMetaClient.getStorageNodesByTableId(tableId);
                        for (Node node : nodes) {
                            result.put(node.getNodeId(), node);
                        }

                        return result;
                    }
                });

        fragmentsCache = CacheBuilder.newBuilder()
                .expireAfterWrite(5, TimeUnit.MINUTES)
                .softValues()
                .build(new CacheLoader<Long, List<Fragment>>() {
                    @Override
                    public List<Fragment> load(Long tableId) throws Exception {
                        return storageMetaClient.getFragmentsByTableId(tableId);
                    }
                });

        ipToNodeManager = CacheBuilder.newBuilder()
                .expireAfterAccess(5, TimeUnit.MINUTES)
                .softValues()
                .build(new CacheLoader<String, Node>() {
                    @Override
                    public Node load(String ip) throws Exception {
                        Node result = storageMetaClient.getEggNodeManagerByIp(ip);

                        return result;
                    }
                });
    }

    public Map<Long, Node> getNodeIdToStorageNodesOfTable(long tableId) {
        return nodeIdToStorageNodeCache.getUnchecked(tableId);
    }

    public List<Fragment> getFragmentListOfTable(long tableId) {
        return fragmentsCache.getUnchecked(tableId);
    }

    public Map<Integer, Node> getFragmentOrderToStorageNodesOfTable(long tableId) {
        Map<Integer, Node> result = Maps.newConcurrentMap();
        Map<Long, Node> nodeIdToNode = getNodeIdToStorageNodesOfTable(tableId);
        List<Fragment> fragments = getFragmentListOfTable(tableId);

        for (Fragment fragment : fragments) {
            result.put(fragment.getFragmentOrder(), nodeIdToNode.get(fragment.getNodeId()));
        }

        return result;
    }


    public Node getNodeManager(String ip) {
        return ipToNodeManager.getUnchecked(ip);
    }

    public BasicMeta.Endpoint getProcessorEndpoint(String ip) {
        BasicMeta.Endpoint result = null;
        Node nodeManager = getNodeManager(ip);

        if (nodeManager == null) {
            return result;
        }

        return eggNodeManagerClient.getProcessor(typeConversionUtils.toEndpoint(nodeManager));
    }
}
