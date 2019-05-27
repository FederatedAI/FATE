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

package com.webank.ai.fate.networking.proxy;

import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.api.networking.proxy.Proxy;
import com.webank.ai.fate.networking.proxy.factory.DefaultPipeFactory;
import com.webank.ai.fate.networking.proxy.factory.LocalBeanFactory;
import com.webank.ai.fate.networking.proxy.factory.PipeFactory;
import com.webank.ai.fate.networking.proxy.grpc.service.DataTransferPipedServerImpl;
import com.webank.ai.fate.networking.proxy.infra.Pipe;
import com.webank.ai.fate.networking.proxy.service.ConfFileBasedFdnRouter;
import com.webank.ai.fate.networking.proxy.service.FdnRouter;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;


public class Main0Insecure {
    private static final Logger LOGGER = LogManager.getLogger(Main0Insecure.class);

    public static void main(String[] args) throws Exception {

        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-proxy.xml");
        LocalBeanFactory localBeanFactory = context.getBean(LocalBeanFactory.class);
        localBeanFactory.setApplicationContext(context);

        // DataTransferServiceImpl dataTransferService = context.getBean(DataTransferServiceImpl.class);

        DataTransferPipedServerImpl dataTransferPipedServer =
                context.getBean(DataTransferPipedServerImpl.class);

        String routeTableFile = "src/main/resources/route_tables/route_table2.json";
        FdnRouter fdnRouter = context.getBean(ConfFileBasedFdnRouter.class);
        fdnRouter.setRouteTable(routeTableFile);

        PipeFactory pipeFactory = context.getBean(DefaultPipeFactory.class);

        InputStream is = new ByteArrayInputStream("hello world from main".getBytes());
        OutputStream os = System.out;


        BasicMeta.Endpoint endpoint = BasicMeta.Endpoint.newBuilder().setIp("127.0.0.1").setPort(8888).build();
        Proxy.Metadata header = Proxy.Metadata.newBuilder()
                .setTask(Proxy.Task.newBuilder().setTaskId("123"))
                //.setDst(endpoint)
                // .setSrc(endpoint)
                .setOperator("operator")
                .build();

        Pipe pipe =
                ((DefaultPipeFactory) pipeFactory).createInputStreamOutputStreamNoStoragePipe(is, os, header);

        dataTransferPipedServer.setPipeFactory(pipeFactory);

        int port = 8888;

        File crt = new File("src/main/resources/server.crt");
        File key = new File("src/main/resources/server-private.pem");

        System.out.println(crt.getAbsolutePath());
        Server server = ServerBuilder
                .forPort(port)
                // .useTransportSecurity(crt, key)
                .addService(dataTransferPipedServer)
                .build()
                .start();

        LOGGER.info("Server started listening on port: {}", port);


        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                this.stop();
                System.err.println("*** server shut down");
            }
        });

        server.awaitTermination();
    }
}
