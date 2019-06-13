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

package com.webank.ai.eggroll.driver;

import com.webank.ai.eggroll.core.factory.DefaultGrpcServerFactory;
import com.webank.ai.eggroll.core.server.BaseEggRollServer;
import com.webank.ai.eggroll.core.server.DefaultServerConf;
import com.webank.ai.eggroll.core.utils.ErrorUtils;
import com.webank.ai.eggroll.driver.clustercomm.transfer.communication.TransferJobScheduler;
import com.webank.ai.eggroll.driver.clustercomm.transfer.api.grpc.server.ProxyServiceImpl;
import com.webank.ai.eggroll.driver.clustercomm.transfer.api.grpc.server.TransferSubmitServiceImpl;
import com.webank.ai.eggroll.framework.storage.service.server.ObjectStoreServicer;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import org.apache.commons.cli.CommandLine;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;
import org.springframework.util.concurrent.ListenableFuture;
import org.springframework.util.concurrent.ListenableFutureCallback;

public class ClusterComm extends BaseEggRollServer {
    private static final Logger LOGGER = LogManager.getLogger();
    public static void main(String[] args) throws Exception {
        String confFilePath = null;
        CommandLine cmd = parseArgs(args);

        if (cmd == null) {
            return;
        }

        confFilePath = cmd.getOptionValue("c");

        ApplicationContext context = new ClassPathXmlApplicationContext("applicationContext-clustercomm.xml");
        ThreadPoolTaskExecutor clusterCommAsyncThreadPool = (ThreadPoolTaskExecutor) context.getBean("clusterCommAsyncThreadPool");
        ErrorUtils errorUtils = context.getBean(ErrorUtils.class);

        TransferJobScheduler transferJobScheduler = context.getBean(TransferJobScheduler.class);
        ListenableFuture<?> schedulerListenableFuture = clusterCommAsyncThreadPool.submitListenable(transferJobScheduler);

        schedulerListenableFuture.addCallback(new ListenableFutureCallback<Object>() {
            @Override
            public void onFailure(Throwable throwable) {
                LOGGER.fatal("[CLUSTERCOMM][MAIN][FATAL] job scheduler failed: {}", errorUtils.getStackTrace(throwable));
            }

            @Override
            public void onSuccess(Object o) {
                LOGGER.fatal("[CLUSTERCOMM][MAIN][FATAL] job scheduler 'return' successful");
            }
        });


        DefaultGrpcServerFactory serverFactory = context.getBean(DefaultGrpcServerFactory.class);
        DefaultServerConf serverConf = (DefaultServerConf) serverFactory.parseConfFile(confFilePath);

        ProxyServiceImpl proxyService = context.getBean(ProxyServiceImpl.class);
        TransferSubmitServiceImpl transferSubmitService = context.getBean(TransferSubmitServiceImpl.class);

        serverConf
                .addService(ServerInterceptors.intercept(proxyService, new ObjectStoreServicer.KvStoreInterceptor()))
                .addService(ServerInterceptors.intercept(transferSubmitService, new ObjectStoreServicer.KvStoreInterceptor()));

        Server server = serverFactory.createServer(serverConf);

        server.start();
        server.awaitTermination();
    }
}
