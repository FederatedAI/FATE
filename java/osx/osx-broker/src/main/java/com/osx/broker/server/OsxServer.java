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

import com.osx.broker.grpc.ContextPrepareInterceptor;
import com.osx.broker.grpc.PcpGrpcService;
import com.osx.broker.grpc.ProxyGrpcService;
import com.osx.broker.grpc.ServiceExceptionHandler;
import com.osx.broker.http.DispatchServlet;
import com.osx.core.config.MetaInfo;
import com.osx.core.utils.OSXCertUtils;
import com.osx.core.utils.OsxX509TrustManager;
import io.grpc.ServerInterceptors;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.ClientAuth;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslProvider;
import org.apache.commons.lang3.StringUtils;
import org.eclipse.jetty.server.HttpConnectionFactory;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.util.ssl.SslContextFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLException;
import javax.net.ssl.TrustManager;
import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.security.KeyStore;
import java.security.SecureRandom;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static com.osx.core.config.MetaInfo.PROPERTY_OPEN_GRPC_TLS_SERVER;

/**
 * http1.X  + grpc
 */
public class OsxServer {

    Logger logger = LoggerFactory.getLogger(OsxServer.class);
    io.grpc.Server server;
    io.grpc.Server tlsServer;
    org.eclipse.jetty.server.Server httpServer;
    org.eclipse.jetty.server.Server httpsServer;
    ProxyGrpcService proxyGrpcService;
    PcpGrpcService pcpGrpcService;

    private synchronized void init() {
        try {
            proxyGrpcService = new ProxyGrpcService();
            pcpGrpcService = new PcpGrpcService();
            server = buildServer();
            if (MetaInfo.PROPERTY_OPEN_HTTP_SERVER) {
                logger.info("prepare to create http server");
                httpServer = buildHttpServer();
                if (httpServer == null) {
                    System.exit(0);
                }
                if (MetaInfo.PROPERTY_HTTP_USE_TLS) {
                    logger.info("prepare to create http server with TLS");
                    httpsServer = buildHttpsServer();
                    if (httpsServer == null) {
                        System.exit(0);
                    }
                }
            }
            tlsServer = buildTlsServer();
        }catch(Exception e){
            logger.error("server init error ",e);
            e.printStackTrace();
        }
    }

    public Server buildHttpServer() {
        Server server = new Server();
        try {
            HttpConnectionFactory http11 = new HttpConnectionFactory();
            ServerConnector connector;
            connector = new ServerConnector(server, MetaInfo.PROPERTY_HTTP_SERVER_ACCEPTOR_NUM, MetaInfo.PROPERTY_HTTP_SERVER_SELECTOR_NUM, http11);
            // logger.info("http server try to start listen port {}", MetaInfo.PROPERTY_HTTP_PORT);
            connector.setPort(MetaInfo.PROPERTY_HTTP_PORT);
            connector.setHost(MetaInfo.PROPERTY_BIND_HOST);
            connector.setAcceptQueueSize(MetaInfo.PROPERTY_HTTP_RECEIVE_QUEUE_SIZE);
            connector.setAcceptedReceiveBufferSize(MetaInfo.PROPERTY_HTTP_ACCEPT_RECEIVE_BUFFER_SIZE);
            server.addConnector(connector);
            server.setHandler(buildServlet());
            return server;
        } catch (Exception e) {
            logger.error("build http server error", e);
        }
        return null;
    }

    public Server buildHttpsServer() {
        Server server = new Server();
        try {
            HttpConnectionFactory http11 = new HttpConnectionFactory();
            ServerConnector connector;
            SslContextFactory.Server sslServer = new SslContextFactory.Server();
//            //如果PROPERTY_HTTP_SSL_TRUST_STORE_PATH 为空， 则去读取证书套件，然后生成一个TRUST_STORE
            if (StringUtils.isNotBlank(MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PATH)) {
                sslServer.setTrustStoreType(MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_TYPE.toUpperCase());
                sslServer.setKeyStorePath(MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PATH);
                sslServer.setTrustStore(OSXCertUtils.getTrustStore(MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PATH, MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_TYPE));
                if (StringUtils.isAllBlank(MetaInfo.PROPERTY_HTTP_SSL_KEY_STORE_PASSWORD, MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PASSWORD)) {
                    throw new IllegalArgumentException("http.ssl.key.store.password/http.ssl.trust.store.password is not set,please check config file");
                }
                sslServer.setTrustStorePassword(StringUtils.firstNonBlank(MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PASSWORD, MetaInfo.PROPERTY_HTTP_SSL_KEY_STORE_PASSWORD));
                sslServer.setKeyStorePassword(StringUtils.firstNonBlank(MetaInfo.PROPERTY_HTTP_SSL_KEY_STORE_PASSWORD, MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PASSWORD));
                sslServer.setTrustStoreProvider(MetaInfo.PROPERTY_HTTP_SSL_TRUST_STORE_PROVIDER);
            } else {
                SSLContext sslContext = SSLContext.getInstance("SSL");
                KeyStore keyStore = OSXCertUtils.getKeyStore(MetaInfo.PROPERTY_SERVER_CA_FILE, MetaInfo.PROPERTY_SERVER_CERT_CHAIN_FILE, MetaInfo.PROPERTY_SERVER_PRIVATE_KEY_FILE);
                TrustManager[] tm = {OsxX509TrustManager.getInstance(keyStore)};
                // Load client certificate
                KeyManagerFactory kmf = KeyManagerFactory.getInstance("SunX509");
                kmf.init(keyStore, MetaInfo.PROPERTY_HTTP_SSL_KEY_STORE_PASSWORD.toCharArray());
                sslContext.init(kmf.getKeyManagers(), tm, new SecureRandom());
                sslServer.setSslContext(sslContext);
            }
            sslServer.setNeedClientAuth(true);
            sslServer.setSslSessionTimeout(MetaInfo.PROPERTY_HTTP_SSL_SESSION_TIME_OUT);
            SslConnectionFactory tls = new SslConnectionFactory(sslServer, http11.getProtocol());
            connector = new ServerConnector(server, MetaInfo.PROPERTY_HTTP_SERVER_ACCEPTOR_NUM, MetaInfo.PROPERTY_HTTP_SERVER_SELECTOR_NUM, tls, http11);
            // logger.info("http server try to start listen port {}", MetaInfo.PROPERTY_HTTP_PORT);
            connector.setPort(MetaInfo.PROPERTY_HTTPS_PORT);
            connector.setHost(MetaInfo.PROPERTY_BIND_HOST);
            connector.setAcceptQueueSize(MetaInfo.PROPERTY_HTTP_RECEIVE_QUEUE_SIZE);
            connector.setAcceptedReceiveBufferSize(MetaInfo.PROPERTY_HTTP_ACCEPT_RECEIVE_BUFFER_SIZE);
            server.addConnector(connector);
            server.setHandler(buildServlet());
//            new Thread(()->{
//                while (true){
//                    try {
//                        logger.info("========================= http连接数 = {}",server.getConnectors().length);
//                        Thread.sleep(5000);
//                    } catch (InterruptedException e) {
//                        e.printStackTrace();
//                    }
//                }
//            }).start();
            return server;
        } catch (Exception e) {
            logger.error("build https server error = {}", e.getMessage());
            e.printStackTrace();
        }
        return null;
    }

    ServletContextHandler buildServlet() {
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath(MetaInfo.PROPERTY_HTTP_CONTEXT_PATH);
        context.addServlet(DispatchServlet.class, MetaInfo.PROPERTY_HTTP_SERVLET_PATH);
        context.setMaxFormContentSize(Integer.MAX_VALUE);
        return context;
    }

    public boolean start() {
        init();
        //grpc
        try {
            server.start();
            logger.info("listen grpc port {} success", MetaInfo.PROPERTY_GRPC_PORT);
        } catch (Exception e) {
            if (e instanceof IOException || e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_GRPC_PORT);
            }
            e.printStackTrace();
            return false;
        }

        //http
        try {
            if (httpServer != null) {
                httpServer.start();
                logger.info("listen http port {} success", MetaInfo.PROPERTY_HTTP_PORT);
            }
        } catch (Exception e) {
            if (e instanceof java.net.BindException || e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_HTTP_PORT);
            }
            e.printStackTrace();
            return false;
        }

        //tls
        try {
            if (tlsServer != null) {
                logger.info("grpc tls server try to start, listen port {}", MetaInfo.PROPERTY_GRPC_TLS_PORT);
                tlsServer.start();
                logger.info("listen grpc tls port {} success", MetaInfo.PROPERTY_GRPC_TLS_PORT);
            }
        } catch (Exception e) {
            if (e instanceof java.net.BindException || e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_GRPC_TLS_PORT);
            }
            e.printStackTrace();
            return false;
        }

        //https
        try {
            if (httpsServer != null) {
                httpsServer.start();
                logger.info("listen https port {} success", MetaInfo.PROPERTY_HTTPS_PORT);
            }
        } catch (Exception e) {
            if (e instanceof java.net.BindException || e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_HTTPS_PORT);
            }
            e.printStackTrace();
            return false;
        }
        return true;
    }

    private io.grpc.Server buildTlsServer() {
        String certChainFilePath = MetaInfo.PROPERTY_SERVER_CERT_CHAIN_FILE;
        String privateKeyFilePath = MetaInfo.PROPERTY_SERVER_PRIVATE_KEY_FILE;
        String trustCertCollectionFilePath = MetaInfo.PROPERTY_SERVER_CA_FILE;
        if (PROPERTY_OPEN_GRPC_TLS_SERVER && StringUtils.isNotBlank(certChainFilePath)
                && StringUtils.isNotBlank(privateKeyFilePath) && StringUtils.isNotBlank(trustCertCollectionFilePath)) {
            try {
                SocketAddress address = new InetSocketAddress(MetaInfo.PROPERTY_BIND_HOST, MetaInfo.PROPERTY_GRPC_TLS_PORT);
                NettyServerBuilder nettyServerBuilder = NettyServerBuilder.forAddress(address);
                SslContextBuilder sslContextBuilder = GrpcSslContexts.forServer(new File(certChainFilePath), new File(privateKeyFilePath))
                        .trustManager(new File(trustCertCollectionFilePath))
                        .clientAuth(ClientAuth.REQUIRE)
                        .sessionTimeout(MetaInfo.PROPERTY_GRPC_SSL_SESSION_TIME_OUT)
                        .sessionCacheSize(MetaInfo.PROPERTY_HTTP_SSL_SESSION_CACHE_SIZE);
                logger.info("running in secure mode. server crt path: {}, server key path: {}, ca crt path: {}.",
                        certChainFilePath, privateKeyFilePath, trustCertCollectionFilePath);
                //serverBuilder.executor(executor);
                nettyServerBuilder.sslContext(GrpcSslContexts.configure(sslContextBuilder, SslProvider.OPENSSL).build());
                nettyServerBuilder.addService(ServerInterceptors.intercept(proxyGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
                nettyServerBuilder.addService(ServerInterceptors.intercept(pcpGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));


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
            } catch (SSLException e) {
                throw new SecurityException(e);
            }
        }
        return null;
    }


    private io.grpc.Server buildServer() {
        SocketAddress address = new InetSocketAddress(MetaInfo.PROPERTY_BIND_HOST, MetaInfo.PROPERTY_GRPC_PORT);
        NettyServerBuilder nettyServerBuilder = NettyServerBuilder.forAddress(address);
        nettyServerBuilder.addService(ServerInterceptors.intercept(proxyGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
        nettyServerBuilder.addService(ServerInterceptors.intercept(pcpGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
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
