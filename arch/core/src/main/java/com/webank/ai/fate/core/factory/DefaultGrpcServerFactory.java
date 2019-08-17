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

package com.webank.ai.fate.core.factory;

import com.google.common.net.InetAddresses;
import com.webank.ai.fate.core.server.DefaultServerConf;
import com.webank.ai.fate.core.server.ServerConf;
import com.webank.ai.fate.core.utils.ErrorUtils;
import io.grpc.BindableService;
import io.grpc.Server;
import io.grpc.ServerServiceDefinition;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.apache.logging.log4j.core.config.Configurator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.core.task.TaskExecutor;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

@Component
public class DefaultGrpcServerFactory implements GrpcServerFactory {
    private static final Logger LOGGER = LogManager.getLogger();
    @Autowired
    protected ApplicationContext applicationContext;
    @Autowired
    private ErrorUtils errorUtils;

    @Override
    public Server createServer(ServerConf serverConf) {
        NettyServerBuilder serverBuilder = null;

        DefaultServerConf defaultServerConf = (DefaultServerConf) serverConf;

        if (StringUtils.isBlank(defaultServerConf.getIp())) {
            LOGGER.info("server build on port only :{}", defaultServerConf.getPort());
            // LOGGER.warn("this may cause trouble in multiple network devices. you may want to consider binding to a ip");
            serverBuilder = NettyServerBuilder.forPort(defaultServerConf.getPort());
        } else {
            LOGGER.info("server build on address {}:{}", defaultServerConf.getIp(), defaultServerConf.getPort());
            InetSocketAddress inetSocketAddress = new InetSocketAddress(
                    InetAddresses.forString(defaultServerConf.getIp()), defaultServerConf.getPort());

            LOGGER.info(inetSocketAddress);
            SocketAddress addr =
                    new InetSocketAddress(
                            InetAddresses.forString(defaultServerConf.getIp()), defaultServerConf.getPort());
            serverBuilder = NettyServerBuilder.forAddress(addr);

        }

        for (BindableService service : defaultServerConf.getBindableServices()) {
            serverBuilder.addService(service);
        }

        for (ServerServiceDefinition service : defaultServerConf.getServerServiceDefinitions()) {
            serverBuilder.addService(service);
        }

        serverBuilder.executor((TaskExecutor) applicationContext.getBean("grpcServiceExecutor"));

        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public void run() {
                // Use stderr here since the logger may have been reset by its JVM shutdown hook.
                System.err.println("*** shutting down gRPC server since JVM is shutting down");
                this.stop();
                System.err.println("*** server shut down");
            }
        });

        serverBuilder.maxInboundMessageSize(32 << 20)
                .flowControlWindow(16 << 20);


        return serverBuilder.build();
    }

    @Override
    public ServerConf parseConfFile(String confFilePath) throws IOException {
        DefaultServerConf defaultServerConf = applicationContext.getBean(DefaultServerConf.class);
        return parseConfFile(confFilePath, defaultServerConf);
    }

    @Override
    public ServerConf parseConfFile(String confFilePath, ServerConf serverConf) throws IOException {
        DefaultServerConf defaultServerConf = null;
        if (serverConf != null) {
            defaultServerConf = (DefaultServerConf) serverConf;
        } else {
            defaultServerConf = applicationContext.getBean(DefaultServerConf.class);
        }

        Path absolutePath = Paths.get(confFilePath).toAbsolutePath();
        String finalConfPath = absolutePath.toString();

        LOGGER.info("final conf path: {}", finalConfPath);

        try (InputStream is = new FileInputStream(finalConfPath)) {
            Properties properties = new Properties();
            properties.load(is);

            defaultServerConf.setProperties(properties);

            String partyId = properties.getProperty("party.id", null);
            if (partyId == null) {
                throw new IllegalArgumentException("partyId cannot be null");
            } else {
                defaultServerConf.setPartyId(partyId);
            }

            String ipString = properties.getProperty("service.ip", null);
            defaultServerConf.setIp(ipString);

            String portString = properties.getProperty("service.port", null);
            if (portString == null) {
                throw new IllegalArgumentException("service.port cannot be null");
            } else {
                int port = Integer.valueOf(portString);
                defaultServerConf.setPort(port);
            }

            String serverCrt = properties.getProperty("server.crt");
            defaultServerConf.setServerCrtPath(serverCrt);

            String serverKey = properties.getProperty("server.key");
            defaultServerConf.setServerKeyPath(serverKey);

            if (StringUtils.isBlank(serverCrt) && StringUtils.isBlank(serverKey)) {
                defaultServerConf.setSecureServer(false);
            } else {
                defaultServerConf.setSecureServer(true);
            }

            String caCrt = properties.getProperty("ca.crt");
            defaultServerConf.setCaCrtPath(caCrt);

            if (StringUtils.isBlank(caCrt)) {
                defaultServerConf.setSecureClient(false);
            } else {
                defaultServerConf.setSecureClient(true);
            }

            String logPropertiesPath = properties.getProperty("log.properties");
            if (StringUtils.isNotBlank(logPropertiesPath)) {
                File logConfFile = new File(logPropertiesPath);
                if (logConfFile.exists() && logConfFile.isFile()) {
                    try (FileInputStream logFis = new FileInputStream(logConfFile)) {
                        ConfigurationSource configurationSource = new ConfigurationSource(logFis, logConfFile);
                        Configurator.initialize(null, configurationSource);

                        defaultServerConf.setLogPropertiesPath(logPropertiesPath);
                        LOGGER.info("using log conf file: {}", logPropertiesPath);
                    } catch (Exception e) {
                        LOGGER.warn("failed to set log conf file at {}. using default conf", logPropertiesPath);
                    }
                }
            }

            defaultServerConf.setDebugEnabled(stringConfValueToBoolean(properties.getProperty("debug.enabled")));
        } catch (Exception e) {
            LOGGER.error(errorUtils.getStackTrace(e));
            throw e;
        }

        return defaultServerConf;
    }

    public boolean stringConfValueToBoolean(String confValue) {
        return StringUtils.isNotBlank(confValue)
                && ("true".equals(confValue.toLowerCase()) || ("1".equals(confValue)));
    }
}
