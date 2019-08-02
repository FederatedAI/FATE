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

package com.webank.ai.fate.serving;

import com.webank.ai.fate.core.network.grpc.client.GrpcClientPool;
import com.webank.ai.fate.serving.utils.HttpClientPool;
import com.webank.ai.fate.core.utils.Configuration;
import com.webank.ai.fate.serving.manger.InferenceWorkerManager;
import com.webank.ai.fate.serving.manger.ModelManager;
import com.webank.ai.fate.serving.service.InferenceService;
import com.webank.ai.fate.serving.service.ModelService;
import com.webank.ai.fate.serving.service.ProxyService;
import com.webank.ai.fate.serving.service.ServiceExceptionHandler;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class ServingServer {
    private static final Logger LOGGER = LogManager.getLogger();
    private Server server;
    private String confPath;

    public ServingServer(String confPath) {
        this.confPath = new File(confPath).getAbsolutePath();
    }

    private void start() throws IOException {
        this.initialize();

        int port = Integer.parseInt(Configuration.getProperty("port"));
        //TODO: Server custom configuration
        Executor    executor= Executors.newCachedThreadPool();

        server = ServerBuilder.forPort(port).executor(executor)
                .addService(ServerInterceptors.intercept(new InferenceService(), new ServiceExceptionHandler()))
                .addService(ServerInterceptors.intercept(new ModelService(), new ServiceExceptionHandler()))
                .addService(ServerInterceptors.intercept(new ProxyService(), new ServiceExceptionHandler()))
                .build();
        LOGGER.info("Server started listening on port: {}, use configuration: {}", port, this.confPath);

        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                LOGGER.info("*** shutting down gRPC server since JVM is shutting down");
                ServingServer.this.stop();
                LOGGER.info("*** server shut down");
            }
        });
    }

    private void stop() {
        if (server != null) {
            server.shutdown();
        }
    }

    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }


    private void initialize() {
        new Configuration(this.confPath).load();
        new ModelManager();
        this.initializeClientPool();
        HttpClientPool.initPool();
        InferenceWorkerManager.prestartAllCoreThreads();
    }

    private void initializeClientPool() {
        ArrayList<String> serverAddress = new ArrayList<>();
        serverAddress.add(Configuration.getProperty("proxy"));
        serverAddress.add(Configuration.getProperty("roll"));
        new Thread(new Runnable() {
            @Override
            public void run() {
                GrpcClientPool.initPool(serverAddress);
            }
        }).start();
        LOGGER.info("Finish init client pool");
    }

    public static void main(String[] args) {
        try {
            Options options = new Options();
            Option option = Option.builder("c")
                    .longOpt("config")
                    .argName("file")
                    .required()
                    .hasArg()
                    .numberOfArgs(1)
                    .desc("configuration file")
                    .build();
            options.addOption(option);
            CommandLineParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args);

            ServingServer a = new ServingServer(cmd.getOptionValue("c"));
            a.start();
            a.blockUntilShutdown();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
