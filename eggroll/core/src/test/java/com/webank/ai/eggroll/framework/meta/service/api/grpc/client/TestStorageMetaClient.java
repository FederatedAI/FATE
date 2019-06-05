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

package com.webank.ai.eggroll.framework.meta.service.api.grpc.client.client;

import com.webank.ai.eggroll.core.api.grpc.client.crud.StorageMetaClient;
import com.webank.ai.eggroll.core.model.FragmentStatus;
import com.webank.ai.eggroll.core.model.NodeStatus;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Dtable;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.LinkedList;
import java.util.List;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-core.xml"})
public class TestStorageMetaClient {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    private StorageMetaClient storageMetaClient;
    private Dtable dtable;
    private Fragment fragment;

    public TestStorageMetaClient() {
        dtable = new Dtable();
        dtable.setNamespace("client_ns");
        dtable.setTableName("client_tn");
        dtable.setTableType("client_tt");
        dtable.setTotalFragments(0);
        dtable.setSerdes("client_normal");
        dtable.setStatus("client_unknown");
        dtable.setStorageVersion(1);

        fragment = new Fragment();
        fragment.setFragmentId(13L);
        fragment.setFragmentOrder(0);
        fragment.setTableId(1L);
        fragment.setNodeId(3L);
        fragment.setStatus(FragmentStatus.DELETED.name());
    }

    @Before
    public void testAbstractInit() {
        storageMetaClient.init("127.0.0.1", 8590);
        LOGGER.info("inited");
    }

    @Test
    public void testCreateTable() {

        Dtable result = storageMetaClient.createTable(dtable);

        LOGGER.info("result: {} ", result);
    }

    @Test
    public void testUpdateFragment() {
        Fragment result = storageMetaClient.updateFragment(fragment);
        LOGGER.info("result: {}", result);
    }

    @Test
    public void testGetTableById() {
        Dtable result = storageMetaClient.getTableById(3L);
        result = storageMetaClient.getTableById(2L);
        result = storageMetaClient.getTableById(1L);
        LOGGER.info(result);
    }

    @Test
    public void testGetTable() {
        String namespace = "client_ns";
        String tableName = "client_tn1";

        Dtable result = storageMetaClient.getTable(namespace, tableName);
        LOGGER.info("result: {}", result);
    }

    @Test
    public void testGetTables() {
        String namespace = "ce46817e-13bf-11e9-8d62-4a00003fc630";
        String tableName = "*x*";

        List<Dtable> result = storageMetaClient.getTables(namespace, tableName);
        LOGGER.info("result: count: {}, value: {}", result.size(), result);
    }

    @Test
    public void testGetNodeByNodeId() {
        Node result = storageMetaClient.getNodeByNodeId(4L);
        LOGGER.info(result);
    }

    @Test
    public void testGetFragmentByFragmentId() {
        Fragment result = storageMetaClient.getFragmentByFragmentId(1L);
        LOGGER.info(result);
    }

    @Test
    public void testGetStorageNodesByTableId() {
        List<Node> result = storageMetaClient.getStorageNodesByTableId(3L);
        LOGGER.info(result);
    }

    @Test
    public void testGetEggNodeManagerByIp() {
        Node result = storageMetaClient.getEggNodeManagerByIp("127.0.0.1");

        LOGGER.info(result);
    }

    @Test
    public void testGetNodeByFragmentId() {
        Node result = storageMetaClient.getNodeByFragmentId(1L);
        LOGGER.info(result);
    }

    @Test
    public void testGetFragmentByTableId() {
        List<Fragment> result = storageMetaClient.getFragmentsByTableId(1L);
        LOGGER.info("result: {}", result);
    }

    @Test
    public void testGetNodesByNodeIds() {
        List<Long> nodeIds = new LinkedList<>();

        nodeIds.add(3L);
        nodeIds.add(4L);

        List<Node> results = storageMetaClient.getNodesByIds(nodeIds);

        LOGGER.info(results);
    }

    @Test
    public void testGetNodeOfStatus() {
        List<Node> results = storageMetaClient.getNodesOfStatus(NodeStatus.WARNING);

        LOGGER.info(results);
    }

    @Test
    public void testCreateFragmentsForTable() {
        Dtable input = new Dtable();
        input.setTableId(1L);
        input.setTotalFragments(3);

        List<Fragment> results = storageMetaClient.createFragmentsForTable(input);

        LOGGER.info("testCreateFragmentsForTable: {}", results);
    }

    @Test
    public void testGetNodes() {
        Node node = new Node();
        node.setIp("127.0.0.1");

        List<Node> results = storageMetaClient.getNodes(node);

        LOGGER.info("testGetNodes: {}", results);
    }

    @Test
    public void testCreateTableIfAbsent() {
        Dtable result = storageMetaClient.createTableIfAbsent(dtable);

        LOGGER.info("result: {} ", result);
    }
}
