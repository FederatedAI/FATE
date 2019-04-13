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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.webank.ai.fate.api.core.BasicMeta;
import com.webank.ai.fate.core.retry.RetryException;
import com.webank.ai.fate.core.retry.Retryer;
import com.webank.ai.fate.core.retry.factory.AttemptOperations;
import com.webank.ai.fate.core.retry.factory.RetryerBuilder;
import com.webank.ai.fate.core.retry.factory.StopStrategies;
import com.webank.ai.fate.core.retry.factory.WaitTimeStrategies;
import com.webank.ai.fate.core.utils.ToStringUtils;
import io.grpc.ConnectivityState;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

@Component
public class GrpcChannelFactory {
    private static final Logger LOGGER = LogManager.getLogger(GrpcChannelFactory.class);
    @Autowired
    private ApplicationContext applicationContext;
    @Autowired
    private ToStringUtils toStringUtils;
    private LoadingCache<BasicMeta.Endpoint, ManagedChannel> channelCache;

    public GrpcChannelFactory() {
        channelCache = CacheBuilder.newBuilder()
                .maximumSize(100)
                .expireAfterAccess(20, TimeUnit.MINUTES)
                .recordStats()
                .weakValues()
                .removalListener(removalNotification -> {
                    BasicMeta.Endpoint endpoint = (BasicMeta.Endpoint) removalNotification.getKey();
                    ManagedChannel managedChannel = (ManagedChannel) removalNotification.getValue();

                    LOGGER.info("[COMMON][CHANNEL][REMOVE] removed for ip: {}, port: {}, hostname: {}. reason: {}",
                            endpoint.getIp(),
                            endpoint.getPort(),
                            endpoint.getHostname(),
                            removalNotification.getCause());
                    if (managedChannel != null) {
                        LOGGER.info("[COMMON][CHANNEL][REMOVE] removed channel state: isShutdown: {}, isTerminated: {}",
                                managedChannel.isShutdown(), managedChannel.isTerminated());
                        managedChannel.shutdown();

                        try {
                            boolean awaitResult = managedChannel.awaitTermination(10, TimeUnit.MILLISECONDS);
                            if (!awaitResult) {
                                LOGGER.warn("[COMMON][CHANNEL][REMOVE] channel await termination timeout");
                            }
                        } catch (InterruptedException e) {
                            LOGGER.warn("[COMMON][CHANNEL][REMOVE] channel await termination interrupted");
                            Thread.currentThread().interrupt();
                        }
                    } else {
                        LOGGER.info("[COMMON][CHANNEL][REMOVE] channel is null");
                    }
                })
                .build(new CacheLoader<BasicMeta.Endpoint, ManagedChannel>() {
                           @Override
                           public ManagedChannel load(BasicMeta.Endpoint endpoint) throws Exception {
                               Preconditions.checkNotNull(endpoint);
                               LOGGER.info("[COMMON][CHANNEL][CREATE] creating channel for endpoint: ip: {}, port: {}, hostname: {}",
                                       endpoint.getIp(), endpoint.getPort(), endpoint.getHostname());
                               return createChannel(endpoint);
                           }
                       }
                );
    }

    // todo: use retry framework
    public ManagedChannel getChannel(final BasicMeta.Endpoint endpoint) {
        ManagedChannel result = null;

        Retryer<ManagedChannel> retryer = RetryerBuilder.<ManagedChannel>newBuilder()
                .withWaitTimeStrategy(WaitTimeStrategies.fixedWaitTime(1000))
                .withStopStrategy(StopStrategies.stopAfterMaxAttempt(10))
                .withAttemptOperation(AttemptOperations.<ManagedChannel>fixedTimeLimit(3, TimeUnit.SECONDS))
                .retryIfAnyException()
                .build();

        final Callable<ManagedChannel> getUsableChannel = () -> getChannelInternal(endpoint);

        try {
            result = retryer.call(getUsableChannel);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        } catch (RetryException e) {
            LOGGER.error("[COMMON][CHANNEL][ERROR] Error getting ManagedChannel after retries");
        }

        return result;
    }

    private ManagedChannel getChannelInternal(BasicMeta.Endpoint endpoint) {
        ManagedChannel result = null;
        try {
            result = channelCache.get(endpoint);
            ConnectivityState state = result.getState(true);
            /*LOGGER.info("Managed channel state: isShutdown: {}, isTerminated: {}, state: {}",
                    result.isShutdown(), result.isTerminated(), state.name());*/

            if (result.isShutdown() || result.isTerminated()) {
                channelCache.invalidate(result);
                result = channelCache.get(endpoint);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        return result;
    }

    private ManagedChannel createChannel(BasicMeta.Endpoint endpoint) {
        String target = endpoint.getIp();
        if (Strings.isNullOrEmpty(target)) {
            target = endpoint.getHostname();
        }

        NettyChannelBuilder builder = NettyChannelBuilder
                .forAddress(target, endpoint.getPort())
                .executor((Executor) applicationContext.getBean("grpcClientExecutor"))
                .keepAliveTime(6, TimeUnit.MINUTES)
                .keepAliveTimeout(1, TimeUnit.HOURS)
                .keepAliveWithoutCalls(true)
                .idleTimeout(1, TimeUnit.HOURS)
                .perRpcBufferLimit(128 << 20)
                .flowControlWindow(32 << 20)
                .maxInboundMessageSize(32 << 20)
                .enableRetry()
                .retryBufferSize(16 << 20)
                .maxRetryAttempts(20);      // todo: configurable

        LOGGER.info("[COMMON][CHANNEL][CREATE] use insecure channel to {}", toStringUtils.toOneLineString(endpoint));
        builder.negotiationType(NegotiationType.PLAINTEXT)
                .usePlaintext();

        ManagedChannel managedChannel = builder
                .build();

        LOGGER.info("[COMMON][CHANNEL][CREATE] created channel to {}", toStringUtils.toOneLineString(endpoint));
        return managedChannel;
    }

}
