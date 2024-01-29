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

package org.fedai.osx.core.frame;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslProvider;
import org.apache.commons.lang3.StringUtils;
import org.fedai.osx.core.config.GrpcChannelInfo;
import org.fedai.osx.core.config.MetaInfo;
import org.fedai.osx.core.exceptions.NoRouterInfoException;
import org.fedai.osx.core.exceptions.SysException;
import org.fedai.osx.core.router.RouterInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.net.ssl.KeyManagerFactory;
import javax.net.ssl.TrustManagerFactory;
import java.io.File;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.fedai.osx.core.config.MetaInfo.*;
import static org.fedai.osx.core.config.MetaInfo.PROPERTY_MAX_TRANSFER_QUEUE_SIZE;

public class GrpcConnectionFactory {

    private static final Logger logger = LoggerFactory.getLogger(GrpcConnectionFactory.class);
    static ConcurrentHashMap<String, ManagedChannel> managedChannelPool = new ConcurrentHashMap<>();
    static LoadingCache<String, ReentrantLock> keyLockCache = CacheBuilder.newBuilder()
            .expireAfterAccess(PROPERTY_MAX_QUEUE_LOCK_LIVE, TimeUnit.SECONDS)
            .concurrencyLevel(4)
            .maximumSize(PROPERTY_MAX_TRANSFER_QUEUE_SIZE)
            .build(new CacheLoader<String, ReentrantLock>() {
                @Override
                public ReentrantLock load(String s) throws Exception {
                    return new ReentrantLock();
                }
            });

    private static AtomicLong   historyCount = new AtomicLong(0);

    private static  boolean checkChannel(ManagedChannel channel){
        boolean shutdown = channel.isShutdown();
        boolean terminated = channel.isTerminated();
        ConnectivityState state = channel.getState(true);
        if (shutdown || terminated || state == ConnectivityState.SHUTDOWN||state==ConnectivityState.TRANSIENT_FAILURE){
            if(state==ConnectivityState.TRANSIENT_FAILURE){
                Thread  thread=    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            channel.shutdown();
                        }
                    });
                thread.start();
            }
            return false;
        }
        return  true;
    }

    static {
        // 创建守护线程
        Thread daemonThread = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(MetaInfo.PROPERTY_CHANNEL_POOL_INFO);
                    int activeNum = 0;
                    int total = managedChannelPool.size();
                    // 遍历并删除元素
                    Iterator<Map.Entry<String, ManagedChannel>> iterator = managedChannelPool.entrySet().iterator();
                    while (iterator.hasNext()) {
                        Map.Entry<String, ManagedChannel> entry = iterator.next();
                        String key = entry.getKey();
                        ReentrantLock  lock = keyLockCache.get(key);
                        try {
                            lock.lock();
                            ManagedChannel channel = entry.getValue();
                            if (!checkChannel(channel)) {
                                iterator.remove();
                            } else {
                                activeNum++;
                            }
                        }finally {
                            if(lock!=null){
                                lock.unlock();
                            }
                        }
                    }
                    logger.info("grpc pool info：history {} current：{}, active:{}", historyCount.get(),total, activeNum);
                } catch (Exception e) {
                    logger.error("exception：", e);
                }
            }
        });
        daemonThread.setDaemon(true);
        daemonThread.start();
    }


    public static  ManagedChannel createManagedChannel(RouterInfo routerInfo) {
        if (routerInfo == null) {
            throw new NoRouterInfoException("no router info");
        }
        ReentrantLock  lock = null;
        try {
            lock = keyLockCache.get(routerInfo.toKey());
            lock.lock();
            if (managedChannelPool.get(routerInfo.toKey()) != null) {
                ManagedChannel targetChannel = managedChannelPool.get(routerInfo.toKey());
                if (!checkChannel(targetChannel)) {
                    ManagedChannel managedChannel = createManagedChannel(routerInfo, buildDefaultGrpcChannelInfo());
                    if (managedChannel != null) {
                        managedChannelPool.put(routerInfo.toKey(), managedChannel);
                    }
                }
                return managedChannelPool.get(routerInfo.toKey());
            } else {
                ManagedChannel managedChannel = createManagedChannel(routerInfo, buildDefaultGrpcChannelInfo());
                if (managedChannel != null) {
                    managedChannelPool.put(routerInfo.toKey(), managedChannel);
                }
                return managedChannel;
            }
        } catch (ExecutionException e) {
            e.printStackTrace();
        } finally {
            if(lock!=null){
                lock.unlock();
            }
        }
        return  null;
    }

    public static  ManagedChannel createManagedChannelNoPool(RouterInfo routerInfo){
        ManagedChannel managedChannel = createManagedChannel(routerInfo, buildDefaultGrpcChannelInfo());
        return  managedChannel;
    }


    private static GrpcChannelInfo buildDefaultGrpcChannelInfo() {
        GrpcChannelInfo grpcChannelInfo = new GrpcChannelInfo();
        grpcChannelInfo.setKeepAliveTime(MetaInfo.PROPERTY_GRPC_CLIENT_KEEPALIVE_TIME_SEC);
        grpcChannelInfo.setKeepAliveTimeout(MetaInfo.PROPERTY_GRPC_CLIENT_KEEPALIVE_TIMEOUT_SEC);
        grpcChannelInfo.setKeepAliveWithoutCalls(MetaInfo.PROPERTY_GRPC_CLIENT_KEEPALIVE_WITHOUT_CALLS_ENABLED);
        grpcChannelInfo.setFlowControlWindow(MetaInfo.PROPERTY_GRPC_CLIENT_FLOW_CONTROL_WINDOW);
        grpcChannelInfo.setMaxInboundMessageSize(MetaInfo.PROPERTY_GRPC_CLIENT_MAX_INBOUND_MESSAGE_SIZE);
        grpcChannelInfo.setRetryBufferSize(MetaInfo.PROPERTY_GRPC_CLIENT_RETRY_BUFFER_SIZE);
        grpcChannelInfo.setIdelTimeOut(MetaInfo.PROPERTY_GRPC_CLIENT_MAX_CONNECTION_IDLE_SEC);
        grpcChannelInfo.setPerRpcBufferLimit(MetaInfo.PROPERTY_GRPC_CLIENT_PER_RPC_BUFFER_LIMIT);
        return grpcChannelInfo;

    }


    public static synchronized ManagedChannel createManagedChannel(RouterInfo routerInfo, GrpcChannelInfo channelInfo) {
        try {
            if (channelInfo == null) {
                throw new SysException("grpc channel info is null");
            }
            NettyChannelBuilder channelBuilder = NettyChannelBuilder
                    .forAddress(routerInfo.getHost(), routerInfo.getPort())
                    .keepAliveTime(channelInfo.getKeepAliveTime(), TimeUnit.MINUTES)
                    .keepAliveTimeout(channelInfo.getKeepAliveTimeout(), TimeUnit.MINUTES)
                    .keepAliveWithoutCalls(channelInfo.isKeepAliveWithoutCalls())
                    .idleTimeout(channelInfo.getIdelTimeOut(), TimeUnit.MINUTES)
                    .perRpcBufferLimit(channelInfo.getPerRpcBufferLimit())
                    .flowControlWindow(channelInfo.getFlowControlWindow())
                    .maxInboundMessageSize(channelInfo.getMaxInboundMessageSize())
                    .enableRetry()
                    .retryBufferSize(channelInfo.getRetryBufferSize())
                    .intercept(ContextPrepareInterceptor.INTERCEPTOR)
                    .maxRetryAttempts(channelInfo.getMaxRetryAttemps());

            if (routerInfo.isUseSSL()) {
                if (routerInfo.isUseKeyStore()) {

                    // Load the truststore file
                    KeyStore trustStore = loadKeyStore(routerInfo.getTrustStoreFilePath(), routerInfo.getTrustStorePassword());
                    // Create a TrustManagerFactory and initialize it with the truststore
                    TrustManagerFactory trustManagerFactory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
                    trustManagerFactory.init(trustStore);
                    // Load the keystore file
                    KeyStore keyStore = loadKeyStore(routerInfo.getKeyStoreFilePath(), routerInfo.getKeyStorePassword());
                    // Create a keyManagerFactory and initialize it with the keystore
                    KeyManagerFactory keyManagerFactory = KeyManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm());
                    keyManagerFactory.init(keyStore, routerInfo.getKeyStorePassword().toCharArray());

                    SslContextBuilder sslContextBuilder = GrpcSslContexts.forClient()
                            .keyManager(keyManagerFactory)
                            .trustManager(trustManagerFactory)
                            .sessionTimeout(PROPERTY_GRPC_TLS_SESSION_TIMEOUT)
                            .sessionCacheSize(PROPERTY_GRPC_TLS_SESSION_SIZE)
                            .sslProvider(SslProvider.OPENSSL);
                    channelBuilder.negotiationType(NegotiationType.TLS).sslContext(sslContextBuilder.build()).useTransportSecurity();

                } else if (StringUtils.isNotBlank(routerInfo.getCertChainFile()) && StringUtils.isNotBlank(routerInfo.getPrivateKeyFile()) && StringUtils.isNotBlank(routerInfo.getCaFile())) {
                    SslContextBuilder sslContextBuilder = GrpcSslContexts.forClient()
                            .keyManager(new File(routerInfo.getCertChainFile()), new File(routerInfo.getPrivateKeyFile()))
                            .trustManager(new File(routerInfo.getCaFile()))
                            .sessionTimeout(PROPERTY_GRPC_TLS_SESSION_TIMEOUT)
                            .sessionCacheSize(PROPERTY_GRPC_TLS_SESSION_SIZE);
                    channelBuilder.negotiationType(NegotiationType.TLS).sslContext(sslContextBuilder.build()).useTransportSecurity().overrideAuthority(routerInfo.getHost());
                }
            } else {
                channelBuilder.usePlaintext();
            }
            historyCount.addAndGet(1);
            return channelBuilder.build();
        } catch (Exception e) {
            logger.error("create channel to {} error : ", routerInfo, e);
            //e.printStackTrace();
        }
        return null;
    }

    private static KeyStore loadKeyStore(String keyStorePath, String keyStorePassword) throws Exception {
        try (FileInputStream fis = new FileInputStream(keyStorePath)) {
            KeyStore keyStore = KeyStore.getInstance("JKS");
            keyStore.load(fis, keyStorePassword.toCharArray());
            return keyStore;
        }
    }

}