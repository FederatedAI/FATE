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

package com.osx.core.frame;
import com.osx.core.config.*;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.osx.core.router.RouterInfo;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public class GrpcConnectionFactory {

    private static final Logger logger = LoggerFactory.getLogger(GrpcConnectionFactory.class);

    private GrpcConnectionFactory() {
    }

    static ConcurrentHashMap<String,ManagedChannel> managedChannelPool = new ConcurrentHashMap<>();

//
//    public static synchronized ManagedChannel createManagedChannel(RouterInfo  routerInfo){
//
//
//
//    }
    static GrpcChannelInfo  defaultfGrpcChannelInfo;

    public static synchronized ManagedChannel createManagedChannel(RouterInfo  routerInfo) {
        if(managedChannelPool.get(routerInfo.toKey())!=null){
            return  managedChannelPool.get(routerInfo.toKey());
        }else {
            ManagedChannel managedChannel =  createManagedChannel(routerInfo, defaultfGrpcChannelInfo);
            managedChannelPool.put(routerInfo.toKey(),managedChannel);
            return  managedChannel;
        }
    }











    public static synchronized ManagedChannel createManagedChannel(RouterInfo  routerInfo, GrpcChannelInfo channelInfo) {
        try {
            if (logger.isDebugEnabled()) {
                logger.debug("create ManagedChannel");
            }

            NettyChannelBuilder channelBuilder = NettyChannelBuilder
                    .forAddress(routerInfo.getHost(), routerInfo.getPort())
                    .keepAliveTime(60, TimeUnit.MINUTES)
                    .keepAliveTimeout(60, TimeUnit.MINUTES)
                    .keepAliveWithoutCalls(true)
                    .idleTimeout(60, TimeUnit.MINUTES)
                    .perRpcBufferLimit(128 << 20)
                    .flowControlWindow(32 << 20)
                    .maxInboundMessageSize(32 << 20)
                    .enableRetry()
                    .retryBufferSize(16 << 20)
                    .maxRetryAttempts(20);

            if (routerInfo != null && NegotiationType.TLS.name().equals(routerInfo.getNegotiationType())
                    && StringUtils.isNotBlank(routerInfo.getCertChainFile())
                    && StringUtils.isNotBlank(routerInfo.getPrivateKeyFile())
                    && StringUtils.isNotBlank(routerInfo.getTrustCertCollectionFile())) {
                SslContextBuilder sslContextBuilder = GrpcSslContexts.forClient()
                        .keyManager(new File(routerInfo.getCertChainFile()), new File(routerInfo.getPrivateKeyFile()))
                        .trustManager(new File(routerInfo.getTrustCertCollectionFile()))
                        .sessionTimeout(3600 << 4)
                        .sessionCacheSize(65536);
                channelBuilder.sslContext(sslContextBuilder.build()).useTransportSecurity();


            } else {
                channelBuilder.usePlaintext();
            }
            return channelBuilder.build();
        }
        catch (Exception e) {
            logger.error("create channel error : " ,e);
            //e.printStackTrace();
        }
        return null;
    }



}