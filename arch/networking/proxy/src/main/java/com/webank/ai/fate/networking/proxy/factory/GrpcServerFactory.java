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

package com.webank.ai.fate.networking.proxy.factory;

import com.google.common.net.InetAddresses;
import com.webank.ai.fate.networking.proxy.grpc.service.DataTransferPipedServerImpl;
import com.webank.ai.fate.networking.proxy.grpc.service.RouteServerImpl;
import com.webank.ai.fate.networking.proxy.manager.ServerConfManager;
import com.webank.ai.fate.networking.proxy.model.ServerConf;
import com.webank.ai.fate.networking.proxy.service.FdnRouter;
import io.grpc.Server;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.ClientAuth;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContext;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.apache.logging.log4j.core.config.Configurator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.core.task.TaskExecutor;
import org.springframework.stereotype.Component;

import javax.net.ssl.SSLException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import java.util.concurrent.TimeUnit;


@Component
public class GrpcServerFactory {
    private static final Logger LOGGER = LogManager.getLogger(GrpcServerFactory.class);
    @Autowired
    private ApplicationContext applicationContext;
    @Autowired
    private ServerConfManager serverConfManager;
    @Autowired
    private FdnRouter fdnRouter;
    @Autowired
    private PipeFactory pipeFactory;
    @Autowired
    private DataTransferPipedServerImpl dataTransferPipedServer;
    @Autowired
    private RouteServerImpl routeServer;

    public Server createServer(ServerConf serverConf) {
        this.serverConfManager.setServerConf(serverConf);

        String routeTablePath = serverConf.getRouteTablePath();
        fdnRouter.setRouteTable(routeTablePath);

        if (serverConf.getPipe() != null) {
            dataTransferPipedServer.setDefaultPipe(serverConf.getPipe());
        } else if (serverConf.getPipeFactory() != null) {
            dataTransferPipedServer.setPipeFactory(serverConf.getPipeFactory());
        } else {
            dataTransferPipedServer.setPipeFactory(pipeFactory);
        }

        NettyServerBuilder serverBuilder = null;

        if (StringUtils.isBlank(serverConf.getIp())) {
            LOGGER.info("server build on port only :{}", serverConf.getPort());
            // LOGGER.warn("this may cause trouble in multiple network devices. you may want to consider binding to a ip");
            serverBuilder = NettyServerBuilder.forPort(serverConf.getPort());
        } else {
            LOGGER.info("server build on address {}:{}", serverConf.getIp(), serverConf.getPort());
            InetSocketAddress inetSocketAddress = new InetSocketAddress(
                    InetAddresses.forString(serverConf.getIp()), serverConf.getPort());

            LOGGER.info(inetSocketAddress);
            SocketAddress addr =
                    new InetSocketAddress(
                            InetAddresses.forString(serverConf.getIp()), serverConf.getPort());
            serverBuilder = NettyServerBuilder.forAddress(addr);

        }

        serverBuilder.addService(dataTransferPipedServer)
                .addService(routeServer)
                .maxConcurrentCallsPerConnection(20000)
                .maxInboundMessageSize(32 << 20)
                .flowControlWindow(32 << 20)
                .keepAliveTime(6, TimeUnit.MINUTES)
                .keepAliveTimeout(24, TimeUnit.HOURS)
                .maxConnectionIdle(1, TimeUnit.HOURS)
                .permitKeepAliveTime(1, TimeUnit.SECONDS)
                .permitKeepAliveWithoutCalls(true)
                .executor((TaskExecutor) applicationContext.getBean("grpcServiceExecutor"))
                .maxConnectionAge(24, TimeUnit.HOURS)
                .maxConnectionAgeGrace(24, TimeUnit.HOURS);

        if (serverConf.isSecureServer()) {
            String serverCrtPath = serverConf.getServerCrtPath();
            String serverKeyPath = serverConf.getServerKeyPath();
            String caCrtPath = serverConf.getCaCrtPath();

            File serverCrt = new File(serverCrtPath);
            File serverKey = new File(serverKeyPath);
            File caCrt = new File(caCrtPath);

            try {
                SslContextBuilder sslContextBuilder = GrpcSslContexts.forServer(serverCrt, serverKey)
                        .trustManager(caCrt);

                if (serverConf.isNeighbourInsecureChannelEnabled()) {
                    sslContextBuilder.clientAuth(ClientAuth.OPTIONAL);
                } else {
                    sslContextBuilder.clientAuth(ClientAuth.REQUIRE);
                }

                SslContext sslContext = sslContextBuilder
                        .sessionTimeout(3600 << 4)
                        .sessionCacheSize(65536)
                        .build();

                serverBuilder.sslContext(sslContext);
            } catch (SSLException e) {
                throw new SecurityException(e);
            }


            LOGGER.info("running in secure mode. server crt path: {}, server key path: {}, ca crt path: {}",
                    serverCrtPath, serverKeyPath, caCrtPath);
        } else {
            LOGGER.info("running in insecure mode");
        }

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                this.stop();
                System.err.println("*** server shut down");
            }
        });

        return serverBuilder.build();
    }

    public Server createServer(String confPath) throws IOException {
        ServerConf serverConf = applicationContext.getBean(ServerConf.class);
        Properties properties = new Properties();

        Path absolutePath = Paths.get(confPath).toAbsolutePath();
        String finalConfPath = absolutePath.toString();

        try (InputStream is = new FileInputStream(finalConfPath)) {
            properties.load(is);

            String coordinator = properties.getProperty("coordinator", null);
            if (coordinator == null) {
                throw new IllegalArgumentException("coordinator cannot be null");
            } else {
                serverConf.setCoordinator(coordinator);
            }

            String ipString = properties.getProperty("ip", null);
            serverConf.setIp(ipString);

            String portString = properties.getProperty("port", null);
            if (portString == null) {
                throw new IllegalArgumentException("port cannot be null");
            } else {
                int port = Integer.valueOf(portString);
                serverConf.setPort(port);
            }

            String routeTablePath = properties.getProperty("route.table", null);
            if (routeTablePath == null) {
                throw new IllegalArgumentException("route table cannot be null");
            } else {
                serverConf.setRouteTablePath(routeTablePath);
            }

            String serverCrt = properties.getProperty("server.crt");
            String serverKey = properties.getProperty("server.key");

            serverConf.setServerCrtPath(serverCrt);
            serverConf.setServerKeyPath(serverKey);

            if (StringUtils.isBlank(serverCrt) && StringUtils.isBlank(serverKey)) {
                serverConf.setSecureServer(false);
            } else {
                serverConf.setSecureServer(true);
            }

            String caCrt = properties.getProperty("ca.crt");
            serverConf.setCaCrtPath(caCrt);

            if (StringUtils.isBlank(caCrt)) {
                serverConf.setSecureClient(false);
            } else {
                serverConf.setSecureClient(true);
            }

            String logPropertiesPath = properties.getProperty("log.properties");
            if (StringUtils.isNotBlank(logPropertiesPath)) {
                File logConfFile = new File(logPropertiesPath);
                if (logConfFile.exists() && logConfFile.isFile()) {
                    try {
                        ConfigurationSource configurationSource =
                                new ConfigurationSource(new FileInputStream(logConfFile), logConfFile);
                        Configurator.initialize(null, configurationSource);

                        serverConf.setLogPropertiesPath(logPropertiesPath);
                        LOGGER.info("using log conf file: {}", logPropertiesPath);
                    } catch (Exception e) {
                        LOGGER.warn("failed to set log conf file at {}. using default conf", logPropertiesPath);
                    }
                }
            }

            String isAuditEnabled = properties.getProperty("audit.enabled");
            if (StringUtils.isNotBlank(isAuditEnabled)
                    && ("true".equals(isAuditEnabled.toLowerCase()))
                    || ("1".equals(isAuditEnabled))) {
                serverConf.setAuditEnabled(true);
            } else {
                serverConf.setAuditEnabled(false);
            }

            String isNeighbourInsecureChannelEnabled = properties.getProperty("neighbour.insecure.channel.enabled");
            if (StringUtils.isNotBlank(isNeighbourInsecureChannelEnabled)
                    && ("true".equals(isNeighbourInsecureChannelEnabled.toLowerCase()))
                    || ("1".equals(isNeighbourInsecureChannelEnabled))) {
                serverConf.setNeighbourInsecureChannelEnabled(true);
            } else {
                serverConf.setNeighbourInsecureChannelEnabled(false);
            }

            String isDebugEnabled = properties.getProperty("debug.enabled");
            if (StringUtils.isNotBlank(isDebugEnabled)
                    && ("true".equals(isDebugEnabled.toLowerCase()))
                    || ("1".equals(isDebugEnabled))) {
                serverConf.setDebugEnabled(true);
            } else {
                serverConf.setDebugEnabled(false);
            }
        } catch (Exception e) {
            LOGGER.error(e);
            throw e;
        }
        return createServer(serverConf);
    }
}
