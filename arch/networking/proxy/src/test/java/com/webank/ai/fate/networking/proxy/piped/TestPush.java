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

package com.webank.ai.fate.networking.proxy.piped;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.factory.DefaultPipeFactory;
import com.webank.ai.fate.networking.proxy.grpc.client.DataTransferPipedClient;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.service.FdnRouter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Scope;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.stereotype.Component;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.concurrent.CountDownLatch;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-proxy.xml"})
public class TestPush {
    private static final Logger LOGGER = LogManager.getLogger(TestPipedClient.class);
    @Autowired
    private DataTransferPipedClient dataTransferPipedClient;
    @Autowired
    private DefaultPipeFactory defaultPipeFactory;
    @Autowired
    private FdnRouter fdnRouter;
    @Autowired
    private ThreadPoolTaskExecutor asyncThreadPool;
    private Proxy.Topic topic1;
    private Proxy.Topic topic2;
    private Proxy.Topic topic10000;
    private Proxy.Topic topic9999;
    private int serverPort = 8888;
    private String localIp = "127.0.0.1";
    private String localhost = "localhost";
    private String operator = "testOperator";
    private Proxy.Task task;
    private BasicMeta.Endpoint proxy1;
    private BasicMeta.Endpoint proxy2;
    private BasicMeta.Endpoint proxy10000;

    @Before
    public void init() {
        Proxy.Task.Builder taskBuilder = Proxy.Task.newBuilder();
        task = taskBuilder.setTaskId("testTask").build();

        BasicMeta.Endpoint.Builder endPointBuilder = BasicMeta.Endpoint.newBuilder();
        proxy1 = endPointBuilder.setIp(localIp).setPort(8888).build();
        proxy2 = endPointBuilder.setIp(localIp).setPort(9999).build();

        Proxy.Topic.Builder topicBuilder = Proxy.Topic.newBuilder();
        topic1 = topicBuilder.setName("topic").setPartyId("webank").setRole("guest").build();
        topic2 = topicBuilder.setName("topic").setPartyId("webank").setRole("host").setCallback(proxy1).build();

        topic10000 = topicBuilder.setName("topic").setPartyId("10000").setRole("guest").setCallback(proxy1).build();
        topic9999 = topicBuilder.setName("topic").setPartyId("9999").setRole("host").build();
    }

    @Test
    public void testConcurrentPush() throws Exception {
        int concurrentCount = 5;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch endLatch = new CountDownLatch(concurrentCount);

        for (int i = 0; i < concurrentCount; ++i) {
            asyncThreadPool.submit(new PushRunnable(startLatch, endLatch));
        }

        startLatch.countDown();

        endLatch.await();

        // Thread.sleep(1000);
    }

    @Component
    @Scope("prototype")
    public class PushRunnable implements Runnable {
        private DataTransferPipedClient client = new DataTransferPipedClient();
        private CountDownLatch startLatch;
        private CountDownLatch endLatch;

        public PushRunnable(CountDownLatch startLatch, CountDownLatch endLatch) {
            this.startLatch = startLatch;
            this.endLatch = endLatch;
        }

        @Override
        public void run() {
            Proxy.Metadata header = Proxy.Metadata.newBuilder()
                    .setTask(task)
                    .setDst(topic1)
                    .setSrc(topic2)
                    .setOperator(operator)
                    .build();

            try {
                InputStream inputStream = new FileInputStream("/Users/max-webank/Downloads/software/IDiskForMac.dmg");
                Pipe pipe = defaultPipeFactory.createInputStreamToPacketUnidirectionalPipe(inputStream, header);
                client.setEndpoint(BasicMeta.Endpoint.newBuilder().setHostname("localhost").setPort(8888).build());

                startLatch.await();

                client.push(header, pipe);
                inputStream.close();

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                LOGGER.info("finished");
                endLatch.countDown();
            }
        }
    }
}
