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

import com.osx.api.router.RouterInfo;
import com.osx.core.config.GrpcChannelInfo;
import com.osx.core.exceptions.NoRouterInfoException;
import com.osx.core.exceptions.SysException;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static com.osx.core.config.MetaInfo.*;

public class GrpcConnectionFactory {

    private static final Logger logger = LoggerFactory.getLogger(GrpcConnectionFactory.class);
    static ConcurrentHashMap<String, ManagedChannel> managedChannelPool = new ConcurrentHashMap<>();

    static GrpcChannelInfo defaultfGrpcChannelInfo;

    private GrpcConnectionFactory() {
    }

    public static synchronized ManagedChannel createManagedChannel(RouterInfo routerInfo,boolean usePooled) {
        if(routerInfo==null){
            throw new NoRouterInfoException("no router info");
        }
        if(usePooled) {
            if (managedChannelPool.get(routerInfo.toKey()) != null) {
                ManagedChannel targetChannel = managedChannelPool.get(routerInfo.toKey());
               // logger.info("channel  is shutdown : {} isTerminated {}",targetChannel.isShutdown() ,targetChannel.isTerminated() ,targetChannel.getState(true));
                return managedChannelPool.get(routerInfo.toKey());
            } else {
                ManagedChannel managedChannel = createManagedChannel(routerInfo, buildDefaultGrpcChannelInfo());
                if(managedChannel!=null) {
                    managedChannelPool.put(routerInfo.toKey(), managedChannel);
                }
                return managedChannel;
            }
        }else{
            ManagedChannel managedChannel = createManagedChannel(routerInfo, buildDefaultGrpcChannelInfo());
            return  managedChannel;
        }
    }


    private static  GrpcChannelInfo  buildDefaultGrpcChannelInfo(){
        GrpcChannelInfo  grpcChannelInfo = new  GrpcChannelInfo();
        grpcChannelInfo.setKeepAliveTime(PROPERTY_GRPC_CLIENT_KEEPALIVE_TIME_SEC);
        grpcChannelInfo.setKeepAliveTimeout(PROPERTY_GRPC_CLIENT_KEEPALIVE_TIMEOUT_SEC);
        grpcChannelInfo.setKeepAliveWithoutCalls(PROPERTY_GRPC_CLIENT_KEEPALIVE_WITHOUT_CALLS_ENABLED);
        grpcChannelInfo.setFlowControlWindow(PROPERTY_GRPC_CLIENT_FLOW_CONTROL_WINDOW);
        grpcChannelInfo.setMaxInboundMessageSize(PROPERTY_GRPC_CLIENT_MAX_INBOUND_MESSAGE_SIZE);
        grpcChannelInfo.setRetryBufferSize(PROPERTY_GRPC_CLIENT_RETRY_BUFFER_SIZE);
        grpcChannelInfo.setIdelTimeOut(PROPERTY_GRPC_CLIENT_MAX_CONNECTION_IDLE_SEC);
        grpcChannelInfo.setPerRpcBufferLimit(PROPERTY_GRPC_CLIENT_PER_RPC_BUFFER_LIMIT);
        return grpcChannelInfo;

    }


    public static synchronized ManagedChannel createManagedChannel(RouterInfo routerInfo, GrpcChannelInfo channelInfo) {
        try {
            if(channelInfo==null){
                throw  new SysException("grpc channel info is null");
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
                    .maxRetryAttempts(channelInfo.getMaxRetryAttemps());
            if (routerInfo.isUseSSL() && NegotiationType.TLS.name().equals(routerInfo.getNegotiationType()) && StringUtils.isNotBlank(routerInfo.getCertChainFile()) && StringUtils.isNotBlank(routerInfo.getPrivateKeyFile()) && StringUtils.isNotBlank(routerInfo.getCaFile())) {
                SslContextBuilder sslContextBuilder = GrpcSslContexts.forClient()
                        .keyManager(new File(routerInfo.getCertChainFile()), new File(routerInfo.getPrivateKeyFile()))
                        .trustManager(new File(routerInfo.getCaFile()))
                        .sessionTimeout(3600 << 4)
                        .sessionCacheSize(65536);

                channelBuilder.sslContext(sslContextBuilder.build()).useTransportSecurity().overrideAuthority(routerInfo.getHost());
            } else {
                channelBuilder.usePlaintext();
            }
            return channelBuilder.build();
        } catch (Exception e) {
            logger.error("create channel to {} error : ",routerInfo, e);
            //e.printStackTrace();
        }
        return null;
    }


}