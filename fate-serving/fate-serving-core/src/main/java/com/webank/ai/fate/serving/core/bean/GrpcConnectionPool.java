package com.webank.ai.fate.serving.core.bean;

import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NegotiationType;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import org.apache.commons.pool2.BasePooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

public class GrpcConnectionPool {

    private  GrpcConnectionPool(){

    }

    static private   GrpcConnectionPool    pool= new GrpcConnectionPool();

    static public  GrpcConnectionPool  getPool(){
        return  pool;
    }


    private static final Logger LOGGER = LogManager.getLogger();

    ConcurrentHashMap<String, GenericObjectPool<ManagedChannel>> poolMap = new ConcurrentHashMap<String, GenericObjectPool<ManagedChannel>>();

    public void returnPool(ManagedChannel channel, String address) {
        try {

            poolMap.get(address).returnObject(channel);

        } catch (Exception e) {
            LOGGER.error("return to pool error", e);
        }
    }
    private Integer maxTotal = 64;

    private Integer maxIdle = 16;

    public ManagedChannel getManagedChannel(String key) throws Exception {

        GenericObjectPool<ManagedChannel> pool = poolMap.get(key);
        if (pool == null) {

            GenericObjectPoolConfig poolConfig = new GenericObjectPoolConfig();

            poolConfig.setMaxTotal(maxTotal);

            poolConfig.setMinIdle(0);

            poolConfig.setMaxIdle(maxIdle);

            poolConfig.setMaxWaitMillis(-1);

            poolConfig.setLifo(true);

            poolConfig.setMinEvictableIdleTimeMillis(1000L * 60L * 30L);

            poolConfig.setBlockWhenExhausted(true);

            poolConfig.setTestOnBorrow(true);

            String[] ipPort = key.split(":");
            String ip = ipPort[0];
            int port = Integer.parseInt(ipPort[1]);
            poolMap.putIfAbsent(key, new GenericObjectPool<ManagedChannel>
                    (new ManagedChannelFactory(ip, port), poolConfig));

        }

        return poolMap.get(key).borrowObject();
    }

    ;


    private class ManagedChannelFactory extends BasePooledObjectFactory<ManagedChannel> {

        private String ip;
        private int port;

        public ManagedChannelFactory(String ip, int port) {
            this.ip = ip;
            this.port = port;
        }

        @Override
        public ManagedChannel create() throws Exception {


            NettyChannelBuilder builder = NettyChannelBuilder
                    .forAddress(ip, port)
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


            builder.negotiationType(NegotiationType.PLAINTEXT)
                    .usePlaintext();

            return builder.build();


        }

        @Override
        public PooledObject<ManagedChannel> wrap(ManagedChannel managedChannel) {
            return new DefaultPooledObject<>(managedChannel);
        }

        @Override
        public void destroyObject(PooledObject<ManagedChannel> p) throws Exception {
            p.getObject().shutdown();
            super.destroyObject(p);
        }

        @Override
        public boolean validateObject(PooledObject<ManagedChannel> channel) {


            return !(channel.getObject().isShutdown() || channel.getObject().isTerminated());
        }


    }
}