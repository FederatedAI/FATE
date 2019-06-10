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

package com.webank.ai.eggroll.framework.egg;


import com.webank.ai.eggroll.core.factory.GrpcServerFactory;
import com.webank.ai.eggroll.core.server.ServerConf;
import com.webank.ai.eggroll.core.utils.RuntimeUtils;
import com.webank.ai.eggroll.framework.egg.node.sandbox.ProcessorOperator;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-egg.xml"})
public class TestProcessorOperator {
    @Autowired
    private ProcessorOperator processorOperator;
    @Autowired
    private ServerConf serverConf;
    @Autowired
    private GrpcServerFactory grpcServerFactory;
    @Autowired
    private RuntimeUtils runtimeUtils;

    @Before
    public void init() throws Exception {
        serverConf = grpcServerFactory.parseConfFile("/Users/max-webank/git/FATE/arch/eggroll/egg/src/main/resources/egg.properties");
    }

    @Test
    public void testStartProcessor() throws Exception {
        Process processor = processorOperator.start(7889);

        System.out.println(processor);
    }

    @Test
    public void testStopProcessor() throws Exception {
        processorOperator.stop(7889);
        System.out.println();
    }

    @Test
    public void testGetCpuCores() {
        System.out.println(Runtime.getRuntime().availableProcessors());
    }

    @Test
    public void testGetRuntimeIp() {
        String myIpAndPort = runtimeUtils.getMySiteLocalIpAndPort();
        String myIp = runtimeUtils.getMySiteLocalAddress();

        System.out.println(myIpAndPort);
        System.out.println(myIp);
    }
}
