///*
// * Copyright 2019 The FATE Authors. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// */
//
//package com.osx.transfer.grpc;
//
//import com.google.common.base.Preconditions;
//import com.google.common.collect.Lists;
//import io.grpc.ConnectivityState;
//import io.grpc.ManagedChannel;
//import io.grpc.netty.shaded.io.grpc.netty.GrpcSslContexts;
//import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
//import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
//import io.grpc.netty.shaded.io.netty.handler.ssl.SslContextBuilder;
//import org.apache.commons.lang3.StringUtils;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.io.File;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.concurrent.ScheduledExecutorService;
//import java.util.concurrent.ScheduledThreadPoolExecutor;
//import java.util.concurrent.TimeUnit;
//import java.util.concurrent.atomic.AtomicLong;
//
//public class GrpcConnectionPool {
//
//    private static final Logger logger = LoggerFactory.getLogger(GrpcConnectionPool.class);
//
//    static private GrpcConnectionPool pool = new GrpcConnectionPool();
//
//    public ConcurrentHashMap<String, ChannelResource> poolMap = new ConcurrentHashMap<String, ChannelResource>();
//    Random r = new Random();
//    private int maxTotalPerAddress = Runtime.getRuntime().availableProcessors();
//    private long defaultLoadFactor = 10;
//    private ScheduledExecutorService scheduledExecutorService = new ScheduledThreadPoolExecutor(1);
//
//    private GrpcConnectionPool() {
//
//            scheduledExecutorService.scheduleAtFixedRate(
//                    new Runnable() {
//                        @Override
//                        public void run() {
//
//                            poolMap.forEach((k, v) -> {
//                                try {
//                                    //logger.info("grpc pool {} channel size {} req count {}", k, v.getChannels().size(), v.getRequestCount().get() - v.getPreCheckCount());
//
//                                    if (needAddChannel(v)) {
//                                        String[] ipPort = k.split(":");
//                                        String ip = ipPort[0];
//                                        int port = Integer.parseInt(ipPort[1]);
//                                        ManagedChannel managedChannel = createManagedChannel(ip, port, v.getNettyServerInfo());
//                                        v.getChannels().add(managedChannel);
//                                    }
//                                    v.getChannels().forEach(e -> {
//                                        try {
//                                            ConnectivityState state = e.getState(true);
//                                            if (state.equals(ConnectivityState.TRANSIENT_FAILURE) || state.equals(ConnectivityState.SHUTDOWN)) {
//                                                fireChannelError(k, state);
//                                            }
//                                        } catch (Exception ex) {
//                                            ex.printStackTrace();
//                                            logger.error("channel {} check status error", k);
//                                        }
//
//                                    });
//                                } catch (Exception e) {
//                                    logger.error("channel {} check status error", k);
//                                }
//
//                            });
//                        }
//                    },
//                    1000,
//                    10000,
//                    TimeUnit.MILLISECONDS);
//
//
//    }
//
//    static public GrpcConnectionPool getPool() {
//        return pool;
//    }
//
//    private void fireChannelError(String k, ConnectivityState status) {
//        logger.error("grpc channel {} status is {}", k, status);
//    }
//
//    private boolean needAddChannel(ChannelResource channelResource) {
//        long requestCount = channelResource.getRequestCount().longValue();
//        long preCount = channelResource.getPreCheckCount();
//        long latestTimestamp = channelResource.getLatestChecktimestamp();
//
//        int channelSize = channelResource.getChannels().size();
//        long now = System.currentTimeMillis();
//        long loadFactor = ((requestCount - preCount) * 1000) / (channelSize * (now - latestTimestamp));
//        channelResource.setLatestChecktimestamp(now);
//        channelResource.setPreCheckCount(requestCount);
//        if (channelSize > maxTotalPerAddress) {
//            return false;
//        }
//        if (latestTimestamp == 0) {
//            return false;
//        }
//        if (channelSize > 0) {
//
//            if (loadFactor > defaultLoadFactor) {
//                return true;
//            } else {
//                return false;
//            }
//        } else {
//            return true;
//        }
//    }
//
//    public ManagedChannel getManagedChannel(String key) {
//        return getAManagedChannel(key, new NettyServerInfo());
//    }
//
//    private ManagedChannel getAManagedChannel(String key, NettyServerInfo nettyServerInfo) {
//        ChannelResource channelResource = poolMap.get(key);
//        if (channelResource == null) {
//            return createInner(key, nettyServerInfo);
//        } else {
//            return getRandomManagedChannel(channelResource);
//        }
//    }
//
//    public ManagedChannel getManagedChannel(String ip,int port, NettyServerInfo nettyServerInfo) {
//        String key = new StringBuilder().append(ip).append(":").append(port).toString();
//        return this.getAManagedChannel(key, nettyServerInfo);
//    }
//
//    public ManagedChannel getManagedChannel(String ip, int port) {
//        String key = new StringBuilder().append(ip).append(":").append(port).toString();
//        return this.getManagedChannel(key);
//    }
//
//    private ManagedChannel getRandomManagedChannel(ChannelResource channelResource) {
//        List<ManagedChannel> list = channelResource.getChannels();
//        Preconditions.checkArgument(list != null && list.size() > 0);
//        int index = r.nextInt(list.size());
//        ManagedChannel result = list.get(index);
//        channelResource.getRequestCount().addAndGet(1);
//        return result;
//
//    }
//
//    private synchronized ManagedChannel createInner(String key, NettyServerInfo nettyServerInfo) {
//        ChannelResource channelResource = poolMap.get(key);
//        if (channelResource == null) {
//            String[] ipPort = key.split(":");
//            String ip = ipPort[0];
//            int port = Integer.parseInt(ipPort[1]);
//            ManagedChannel managedChannel = createManagedChannel(ip, port, nettyServerInfo);
//            List<ManagedChannel> managedChannelList = new ArrayList<ManagedChannel>();
//            managedChannelList.add(managedChannel);
//            channelResource = new ChannelResource(key, nettyServerInfo);
//            channelResource.setChannels(managedChannelList);
//            channelResource.getRequestCount().addAndGet(1);
//            poolMap.put(key, channelResource);
//            return managedChannel;
//        } else {
//            return getRandomManagedChannel(channelResource);
//        }
//
//    }
//
//    public synchronized ManagedChannel createManagedChannel(String ip, int port, NettyServerInfo nettyServerInfo) {
//        try {
//            if (logger.isDebugEnabled()) {
//                logger.debug("create ManagedChannel");
//            }
//
//            NettyChannelBuilder channelBuilder = NettyChannelBuilder
//                    .forAddress(ip, port)
//                    .keepAliveTime(60, TimeUnit.SECONDS)
//                    .keepAliveTimeout(60, TimeUnit.SECONDS)
//                    .keepAliveWithoutCalls(true)
//                    .idleTimeout(60, TimeUnit.SECONDS)
//                    .perRpcBufferLimit(128 << 20)
//                    .flowControlWindow(32 << 20)
//                    .maxInboundMessageSize(32 << 20)
//                    .enableRetry()
//                    .retryBufferSize(16 << 20)
//                    .maxRetryAttempts(20);
//
//            if (nettyServerInfo != null && nettyServerInfo.getNegotiationType() == NegotiationType.TLS
//                    && StringUtils.isNotBlank(nettyServerInfo.getCertChainFilePath())
//                    && StringUtils.isNotBlank(nettyServerInfo.getPrivateKeyFilePath())
//                    && StringUtils.isNotBlank(nettyServerInfo.getTrustCertCollectionFilePath())) {
//                SslContextBuilder sslContextBuilder = GrpcSslContexts.forClient()
//                        .keyManager(new File(nettyServerInfo.getCertChainFilePath()), new File(nettyServerInfo.getPrivateKeyFilePath()))
//                        .trustManager(new File(nettyServerInfo.getTrustCertCollectionFilePath()))
//                        .sessionTimeout(3600 << 4)
//                        .sessionCacheSize(65536);
//                channelBuilder.sslContext(sslContextBuilder.build()).useTransportSecurity();
//
//                logger.info("running in secure mode for endpoint {}:{}, client crt path: {}, client key path: {}, ca crt path: {}.",
//                        ip, port, nettyServerInfo.getCertChainFilePath(), nettyServerInfo.getPrivateKeyFilePath(),
//                        nettyServerInfo.getTrustCertCollectionFilePath());
//            } else {
//                channelBuilder.usePlaintext();
//            }
//            return channelBuilder.build();
//        }
//        catch (Exception e) {
//            logger.error("create channel error : " ,e);
//            //e.printStackTrace();
//        }
//        return null;
//    }
//
//    class ChannelResource {
//
//        String address;
//        List<ManagedChannel> channels = Lists.newArrayList();
//        AtomicLong requestCount = new AtomicLong(0);
//        long latestChecktimestamp = 0;
//        long preCheckCount = 0;
//        NettyServerInfo nettyServerInfo;
//
//        public ChannelResource(String address) {
//            this(address, null);
//        }
//
//        public NettyServerInfo getNettyServerInfo() {
//            return nettyServerInfo;
//        }
//
//        public  ChannelResource(String address, NettyServerInfo nettyServerInfo) {
//            this.address = address;
//            this.nettyServerInfo = nettyServerInfo;
//        }
//
//        public List<ManagedChannel> getChannels() {
//            return channels;
//        }
//
//        public void setChannels(List<ManagedChannel> channels) {
//            this.channels = channels;
//        }
//
//        public AtomicLong getRequestCount() {
//            return requestCount;
//        }
//
//        public void setRequestCount(AtomicLong requestCount) {
//            this.requestCount = requestCount;
//        }
//
//        public long getLatestChecktimestamp() {
//            return latestChecktimestamp;
//        }
//
//        public void setLatestChecktimestamp(long latestChecktimestamp) {
//            this.latestChecktimestamp = latestChecktimestamp;
//        }
//
//        public long getPreCheckCount() {
//            return preCheckCount;
//        }
//
//        public void setPreCheckCount(long preCheckCount) {
//            this.preCheckCount = preCheckCount;
//        }
//    }
//
//}