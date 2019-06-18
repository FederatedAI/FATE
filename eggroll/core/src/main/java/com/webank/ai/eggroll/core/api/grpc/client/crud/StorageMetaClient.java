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

package com.webank.ai.eggroll.core.api.grpc.client.crud;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.webank.ai.eggroll.api.framework.meta.service.StorageMetaServiceGrpc;
import com.webank.ai.eggroll.core.io.StoreInfo;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Scope("prototype")
public class StorageMetaClient extends BaseCrudClient<StorageMetaServiceGrpc.StorageMetaServiceStub> {
    public Dtable createTable(Dtable dtable) {
        return doCrudRequest(dtable,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::createTable,
                Dtable.class);
    }

    public Dtable createTableIfAbsent(Dtable dtable) {
        return doCrudRequest(dtable,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::createTableIfAbsent,
                Dtable.class);
    }

    public Dtable updateTable(Dtable dtable) {
        return doCrudRequest(
                dtable,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::updateTable,
                Dtable.class);
    }

    public Fragment updateFragment(Fragment fragment) {
        return doCrudRequest(
                fragment,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::updateFragment,
                Fragment.class);
    }

    public Dtable getTableById(Long tableId) {
        return doCrudRequest(tableId,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getTableById,
                Dtable.class);
    }

    public Dtable getTable(String namespace, String tableName) {
        Dtable dtable = new Dtable();
        dtable.setNamespace(namespace);
        dtable.setTableName(tableName);

        return getTable(dtable);
    }

    public Dtable getTable(Dtable dtable) {
        return doCrudRequest(dtable,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getTable,
                Dtable.class);
    }

    public Dtable getTable(StoreInfo storeInfo) {
        Dtable dtable = new Dtable();
        dtable.setTableType(storeInfo.getType());
        dtable.setNamespace(storeInfo.getNameSpace());
        dtable.setTableName(storeInfo.getTableName());

        return getTable(dtable);
    }

    public List<Dtable> getTables(Dtable dtable) {
        List<Dtable> result = Lists.newArrayList();
        result = doCrudRequest(dtable,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                StorageMetaServiceGrpc.StorageMetaServiceStub::getTables,
                result.getClass());

        return result;
    }

    public List<Dtable> getTables(String namespace, String tableName) {
        Dtable dtable = new Dtable();
        dtable.setNamespace(namespace);
        dtable.setTableName(tableName);

        return getTables(dtable);
    }

    public List<Dtable> getTables(StoreInfo storeInfo) {
        Dtable dtable = new Dtable();
        dtable.setTableType(storeInfo.getType());
        dtable.setNamespace(storeInfo.getNameSpace());
        dtable.setTableName(storeInfo.getTableName());

        return getTables(dtable);
    }

    public Node getNodeByNodeId(Long nodeId) {
        return doCrudRequest(
                nodeId,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getNodeById,
                Node.class);
    }

    public Fragment getFragmentByFragmentId(Long fragmentId) {
        return doCrudRequest(
                fragmentId,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getFragmentById,
                Fragment.class);
    }

    public Node getNodeByFragmentId(Long fragmentId) {
        Node result = null;
        Fragment fragment = getFragmentByFragmentId(fragmentId);

        if (fragment == null) {
            return result;
        }

        Long nodeId = fragment.getNodeId();
        if (nodeId != null) {
            result = getNodeByNodeId(nodeId);
        }

        return result;
    }

    public List<Fragment> getFragmentsByTableId(Long tableId) {
        List<Fragment> result = Lists.newArrayList();

        result = doCrudRequest(
                tableId,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getFragmentsByTableId,
                result.getClass());
        return result;
    }

    public List<Node> getStorageNodesByTableId(Long nodeId) {
        Preconditions.checkNotNull(nodeId);

        List<Node> result = Lists.newArrayList();

        result = doCrudRequest(
                nodeId,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getStorageNodesByTableId,
                result.getClass());

        return result;
    }

    public List<Node> getNodesByIds(List<Long> nodeIds) {
        List<Node> result = Lists.newArrayList();

        result = doCrudRequest(
                nodeIds,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getNodesByIds,
                result.getClass());

        return result;
    }

    public List<Node> getNodesOfStatus(NodeStatus nodeStatus) {
        List<Node> result = Lists.newArrayList();

        result = doCrudRequest(nodeStatus,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getNodesOfStatus,
                result.getClass());

        return result;
    }

    public List<Fragment> createFragmentsForTable(Dtable dtable) {
        List<Fragment> result = Lists.newArrayList();

        result = doCrudRequest(dtable,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::createFragmentsForTable,
                result.getClass());

        return result;
    }

    public List<Node> getNodes(Node node) {
        List<Node> result = Lists.newArrayList();

        result = doCrudRequest(node,
                (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getNodes,
                result.getClass());

        return result;
    }

    public Node getEggNodeManagerByIp(String ip) {
        Node result = new Node();

        result = doCrudRequest(ip, (CrudRequestProcessor<StorageMetaServiceGrpc.StorageMetaServiceStub>)
                        StorageMetaServiceGrpc.StorageMetaServiceStub::getEggNodeManagerByIp,
                result.getClass());

        return result;
    }
}
