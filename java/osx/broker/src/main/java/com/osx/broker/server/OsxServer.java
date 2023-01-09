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
package com.osx.broker.server;
import com.osx.broker.ServiceContainer;
import com.osx.broker.grpc.ContextPrepareInterceptor;
import com.osx.broker.grpc.ServiceExceptionHandler;
import com.osx.broker.http.DispatchServlet;
import com.osx.core.config.MetaInfo;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.ClientAuth;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslProvider;
import org.apache.commons.lang3.StringUtils;
import org.eclipse.jetty.server.*;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.SSLException;
import java.io.File;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static com.osx.core.config.MetaInfo.PROPERTY_OPEN_GRPC_TLS_SERVER;

/**
 *  http1.X  + grpc
 */
public class OsxServer {


    Logger logger = LoggerFactory.getLogger(OsxServer.class);
    io.grpc.Server server;
    io.grpc.Server tlsServer;
    org.eclipse.jetty.server.Server  httpServer;

    private void init() {
        server = buildServer();
        if(MetaInfo.PROPERTY_OPEN_HTTP_SERVER) {
            httpServer = buildHttpServer();
        }

        //  tlsServer = buildTlsServer();
    }

    public Server  buildHttpServer(){
        Server server = new Server();
        try {
            int acceptors = 1;
            int selectors = 1;
            ServerConnector connector = new ServerConnector(server, acceptors, selectors, new HttpConnectionFactory());
           // logger.info("http server try to start listen port {}", MetaInfo.PROPERTY_HTTP_PORT);
            connector.setPort(MetaInfo.PROPERTY_HTTP_PORT);
            connector.setHost("127.0.0.1");
            connector.setAcceptQueueSize(128);
            server.addConnector(connector);
            server.setHandler(buildServlet());
            return  server;
        } catch (Exception e) {
            logger.error("build http server error",e);
        }
        return null;
    }

    ServletContextHandler   buildServlet(){
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath(MetaInfo.PROPERTY_HTTP_CONTEXT_PATH);
        ServletHolder servletHolder = context.addServlet(DispatchServlet.class, MetaInfo.PROPERTY_HTTP_SERVLET_PATH);//"/*"
        return  context;
    }



    public boolean start() {

        init();
        try {
            server.start();
            logger.info("listen grpc port {} success", MetaInfo.PROPERTY_GRPC_PORT);
        } catch (Exception e) {
            if (e instanceof  java.net.BindException||e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_GRPC_PORT);
            }
            return false;
        }
        try{
            if(httpServer!=null){

                httpServer.start();
                logger.info("listen http port {} success", MetaInfo.PROPERTY_HTTP_PORT);
            }
        }
        catch (Exception e) {
            if (e instanceof  java.net.BindException||e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_GRPC_PORT);
            }
            return false;
        }
        try{
            if (tlsServer != null) {
                logger.info("grpc tls server try to start, listen port {}", MetaInfo.PROPERTY_GRPC_TLS_PORT);
                tlsServer.start();
                logger.info("listen grpc tls port {} success", MetaInfo.PROPERTY_GRPC_TLS_PORT);
            }
        } catch (Exception e) {
            if (e instanceof  java.net.BindException||e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_GRPC_TLS_PORT);
            }
            return false;
        }
        return true;
    }

    private io.grpc.Server buildTlsServer(){
        String certChainFilePath = MetaInfo.PROPERTY_SERVER_CERTCHAIN_FILE;
        String privateKeyFilePath = MetaInfo.PROPERTY_SERVER_PRIVATEKEY_FILE;
        String trustCertCollectionFilePath = MetaInfo.PROPERTY_SERVER_CA_FILE;

        if(PROPERTY_OPEN_GRPC_TLS_SERVER && StringUtils.isNotBlank(certChainFilePath)
                && StringUtils.isNotBlank(privateKeyFilePath) && StringUtils.isNotBlank(trustCertCollectionFilePath)) {
            try {
                int port = MetaInfo.PROPERTY_GRPC_TLS_PORT;
                NettyServerBuilder serverBuilder = (NettyServerBuilder) ServerBuilder.forPort(port);
                SslContextBuilder sslContextBuilder = GrpcSslContexts.forServer(new File(certChainFilePath), new File(privateKeyFilePath))
                        .trustManager(new File(trustCertCollectionFilePath))
                        .clientAuth(ClientAuth.REQUIRE)
                        .sessionTimeout(3600 << 4)
                        .sessionCacheSize(65536);
                GrpcSslContexts.configure(sslContextBuilder, SslProvider.OPENSSL);
                serverBuilder.sslContext(sslContextBuilder.build());
                logger.info("running in secure mode. server crt path: {}, server key path: {}, ca crt path: {}.",
                        certChainFilePath, privateKeyFilePath, trustCertCollectionFilePath);
                //serverBuilder.executor(executor);
                serverBuilder.addService(ServerInterceptors.intercept(ServiceContainer.proxyGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
                serverBuilder.addService(ServerInterceptors.intercept(ServiceContainer.pcpGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
                return  serverBuilder.build();
            } catch (SSLException e) {
                throw new SecurityException(e);
            }


        }
        return null;
    }


    private io.grpc.Server buildServer() {
        NettyServerBuilder nettyServerBuilder = (NettyServerBuilder) ServerBuilder.forPort(MetaInfo.PROPERTY_GRPC_PORT);
        nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.proxyGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
        nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.pcpGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
              nettyServerBuilder
                .executor(Executors.newCachedThreadPool())
                .maxConcurrentCallsPerConnection(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONCURRENT_CALL_PER_CONNECTION)
                .maxInboundMessageSize(MetaInfo.PROPERTY_GRPC_SERVER_MAX_INBOUND_MESSAGE_SIZE)
                .maxInboundMetadataSize(MetaInfo.PROPERTY_GRPC_SERVER_MAX_INBOUND_METADATA_SIZE)
                .flowControlWindow(MetaInfo.PROPERTY_GRPC_SERVER_FLOW_CONTROL_WINDOW);

        if (MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIME_SEC > 0)
            nettyServerBuilder.keepAliveTime(MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIMEOUT_SEC > 0)
            nettyServerBuilder.keepAliveTimeout(MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_TIMEOUT_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_SERVER_PERMIT_KEEPALIVE_TIME_SEC > 0) {

            nettyServerBuilder.permitKeepAliveTime(MetaInfo.PROPERTY_GRPC_SERVER_PERMIT_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
        }
        if (MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_WITHOUT_CALLS_ENABLED)
            nettyServerBuilder.permitKeepAliveWithoutCalls(MetaInfo.PROPERTY_GRPC_SERVER_KEEPALIVE_WITHOUT_CALLS_ENABLED);
        if (MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_IDLE_SEC > 0)
            nettyServerBuilder.maxConnectionIdle(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_IDLE_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_SEC > 0)
            nettyServerBuilder.maxConnectionAge(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_GRACE_SEC > 0)
            nettyServerBuilder.maxConnectionAgeGrace(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONNECTION_AGE_GRACE_SEC, TimeUnit.SECONDS);
        return nettyServerBuilder.build();
    }
}
