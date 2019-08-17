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

package com.webank.ai.eggroll.framework.roll.api.grpc.client;

import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Fragment;
import com.webank.ai.eggroll.framework.meta.service.dao.generated.model.model.Node;
import com.webank.ai.eggroll.framework.roll.helper.NodeHelper;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.List;
import java.util.Map;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath*:applicationContext-roll.xml"})
public class TestNodeHelper {
    private static final Logger LOGGER = LogManager.getLogger();

    @Autowired
    private NodeHelper nodeHelper;

    @Test
    public void testNodeIdToStorageNodes() {
        Map<Long, Node> result = nodeHelper.getNodeIdToStorageNodesOfTable(500L);

        System.out.println(result);
    }

    @Test
    public void testGetFragmentListOfTable() {
        List<Fragment> result = nodeHelper.getFragmentListOfTable(500L);

        System.out.println(result);
    }

    @Test
    public void testGetFragmentOrderToStorageNodesOfTable() {
        Map<Integer, Node> result = nodeHelper.getFragmentOrderToStorageNodesOfTable(500L);

        System.out.println(result);
    }
}
