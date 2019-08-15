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

import com.google.protobuf.ByteString;
import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.factory.DefaultPipeFactory;
import com.webank.ai.fate.networking.proxy.grpc.client.DataTransferPipedClient;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.model.ServerConf;
import com.webank.ai.fate.networking.proxy.service.FdnRouter;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.CountDownLatch;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:applicationContext-proxy.xml"})
public class TestPipedClient {
    private static final Logger LOGGER = LogManager.getLogger(TestPipedClient.class);
    @Autowired
    private DataTransferPipedClient dataTransferPipedClient;
    @Autowired
    private DefaultPipeFactory defaultPipeFactory;
    @Autowired
    private FdnRouter fdnRouter;
    @Autowired
    private ServerConf serverConf;
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
        task = taskBuilder.setTaskId("testProxy.Task").build();

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
    public void testPush() throws Exception {
        Proxy.Metadata header = Proxy.Metadata.newBuilder()
                .setTask(task)
                .setDst(topic9999)
                .setSrc(topic10000)
                .setOperator(operator)
                .build();

        //fdnRouter.setRouteTable("src/main/resources/route_tables/route_table1.json");
//office_wifi_office365.pkg, StarUML-3.0.1.dmg, IDiskForMac.dmg
        serverConf.setSecureClient(true);
        serverConf.setCaCrtPath("/Users/max-webank/Documents/zmodem/ca.crt");
        serverConf.setCoordinator("10000");

        InputStream inputStream = new FileInputStream("/Users/max-webank/Downloads/software/IDiskForMac.dmg");
        Pipe pipe = defaultPipeFactory.createInputStreamToPacketUnidirectionalPipe(inputStream, header);

        // dataTransferPipedClient.setEndpoint(Endpoint.newBuilder().setHostname("localhost").setPort(8888).build());
        dataTransferPipedClient.setEndpoint(BasicMeta.Endpoint.newBuilder().setIp("127.0.0.1").setPort(8888).build());
        dataTransferPipedClient.push(header, pipe);

    }

    @Test
    public void testPull() throws Exception {
        Proxy.Metadata header = Proxy.Metadata.newBuilder()
                .setTask(task)
                .setDst(topic2)
                .setSrc(topic1)
                .setOperator(operator)
                .build();
        //fdnRouter.setRouteTable("src/main/resources/route_tables/route_table1.json");
        OutputStream os = new FileOutputStream("/tmp/testPullOut");
        Pipe pipe = defaultPipeFactory.createPacketToOutputStreamUnidirectionalPipe(os);

        dataTransferPipedClient.setEndpoint(BasicMeta.Endpoint.newBuilder().setHostname("127.0.0.1").setPort(8888).build());
        dataTransferPipedClient.pull(header, pipe);

        System.out.println("done");
    }

    @Test
    public void testUnaryCall() throws Exception {
        Proxy.Metadata header = Proxy.Metadata.newBuilder()
                .setTask(task)
                .setDst(topic1)
                .setSrc(topic2)
                .setOperator(operator)
                .build();


        Proxy.Data data = Proxy.Data.newBuilder().setValue(ByteString.copyFromUtf8("hello")).build();

        Proxy.Packet packet = Proxy.Packet.newBuilder().setHeader(header).setBody(data).build();
        dataTransferPipedClient.setEndpoint(BasicMeta.Endpoint.newBuilder().setHostname("localhost").setPort(8888).build());
        dataTransferPipedClient.unaryCall(packet, defaultPipeFactory.create());
    }

    @Test
    public void testPushAndPull() throws Exception {
/*
        Proxy.Metadata pullHeader = Proxy.Metadata.newBuilder()
                .setProxy.Task(task)
                .setDst(proxy1)
                .setSrc(proxy2)
                .setOperator(operator)
                .build();

        Proxy.Metadata pushHeader = Proxy.Metadata.newBuilder()
                .setProxy.Task(task)
                .setDst(proxy2)
                .setSrc(proxy1)
                .setOperator(operator)
                .build();
*/

        CountDownLatch latch = new CountDownLatch(1);

        Thread pushThread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    System.out.println("push thread waiting ...");
                    latch.await();
                    Thread.sleep(1000);
                    System.out.println("push thread running ...");
                    testPush();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, "pushThread");

        Thread pullThread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    System.out.println("pull thread waiting ...");
                    latch.await();
                    testPull();
                    System.out.println("pull thread running ...");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }, "pullThread");

        pullThread.start();
        pushThread.start();

        System.out.println("ready to countdown");

        latch.countDown();


        pushThread.join();
        pullThread.join();

    }
}
