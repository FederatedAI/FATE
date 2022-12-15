package com.osx.broker.server;

import com.osx.core.config.MetaInfo;
import com.osx.broker.ServiceContainer;
import com.osx.broker.grpc.ContextPrepareInterceptor;
import com.osx.broker.grpc.ServiceExceptionHandler;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class OsxServer  {
    Logger logger  = LoggerFactory.getLogger(OsxServer.class);
    Server  server;
    Server  tlsServer;

    private  void init() {
        server = buildServer();
      //  tlsServer = buildTlsServer();
    }

    public boolean start() {

        init();
        try {
            logger.info("try to start listen port {}",MetaInfo.PROPERTY_PORT);
            server.start();
            if(tlsServer!=null){
                tlsServer.start();
            }
            logger.info("listen port {} success",MetaInfo.PROPERTY_PORT);
        } catch (Exception  e) {
            if(e.getCause() instanceof  java.net.BindException){
                logger.error("port {}  already in use, please try to choose another one  !!!!",MetaInfo.PROPERTY_PORT);
            }
            return false;
        }
        return true;
    }

//    private Server buildTlsServer(){
//
//        String negotiationType = MetaInfo.PROPERTY_NEGOTIATIONTYPE;
//        String certChainFilePath = MetaInfo.PROPERTY_SERVER_CERTCHAIN_FILE;
//        String privateKeyFilePath = MetaInfo.PROPERTY_SERVER_PRIVATEKEY_FILE;
//        String trustCertCollectionFilePath = MetaInfo.PROPERTY_SERVER_CA_FILE;
//
//        if(NegotiationType.TLS == NegotiationType.valueOf(negotiationType) && StringUtils.isNotBlank(certChainFilePath)
//                && StringUtils.isNotBlank(privateKeyFilePath) && StringUtils.isNotBlank(trustCertCollectionFilePath)) {
//            try {
//                int port = MetaInfo.PROPERTY_TLS_PORT;
//                NettyServerBuilder serverBuilder = (NettyServerBuilder) ServerBuilder.forPort(port);
//                SslContextBuilder sslContextBuilder = GrpcSslContexts.forServer(new File(certChainFilePath), new File(privateKeyFilePath))
//                        .trustManager(new File(trustCertCollectionFilePath))
//                        .clientAuth(ClientAuth.REQUIRE)
//                        .sessionTimeout(3600 << 4)
//                        .sessionCacheSize(65536);
//                GrpcSslContexts.configure(sslContextBuilder, SslProvider.OPENSSL);
//                serverBuilder.sslContext(sslContextBuilder.build());
//                logger.info("running in secure mode. server crt path: {}, server key path: {}, ca crt path: {}.",
//                        certChainFilePath, privateKeyFilePath, trustCertCollectionFilePath);
//                //serverBuilder.executor(executor);
//                serverBuilder.addService(ServerInterceptors.intercept(ServiceContainer.proxyGrpcService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//                serverBuilder.addService(ServerInterceptors.intercept(, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//              //  serverBuilder.addService(ServerInterceptors.intercept(ServiceContainer.commonService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
//                              return  serverBuilder.build();
//            } catch (SSLException e) {
//                throw new SecurityException(e);
//            }
//
//
//        }
//        return null;
//    }


    private Server buildServer( ){
        NettyServerBuilder nettyServerBuilder = (NettyServerBuilder) ServerBuilder.forPort(MetaInfo.PROPERTY_PORT);
        nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.proxyGrpcService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
        nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.pcpGrpcService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
        //nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.commonService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
       //nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.clusterService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));


        // nettyServerBuilder.addService(ServerInterceptors.intercept(dLedgerServer,new ServiceExceptionHandler(), new ContextPrepareInterceptor() ));
        nettyServerBuilder
                .executor(Executors.newCachedThreadPool())
                .maxConcurrentCallsPerConnection(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONCURRENT_CALL_PER_CONNECTION)
                .maxInboundMessageSize(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_MESSAGE_SIZE)
                .maxInboundMetadataSize(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_INBOUND_METADATA_SIZE)
                .flowControlWindow(MetaInfo.PROPERTY_GRPC_CHANNEL_FLOW_CONTROL_WINDOW);


//        PERMIT_KEEPALIVE_TIME
//，默认 5 minutes
//                PERMIT_KEEPALIVE_WITHOUT_CALLS
//，默认 false
     //   nettyServerBuilder.withChildOption(c)
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC > 0) nettyServerBuilder.keepAliveTime(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC > 0) nettyServerBuilder.keepAliveTimeout(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_TIMEOUT_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC > 0) {

            nettyServerBuilder.permitKeepAliveTime(MetaInfo.PROPERTY_GRPC_CHANNEL_PERMIT_KEEPALIVE_TIME_SEC, TimeUnit.SECONDS);
        }
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED) nettyServerBuilder.permitKeepAliveWithoutCalls(MetaInfo.PROPERTY_GRPC_CHANNEL_KEEPALIVE_WITHOUT_CALLS_ENABLED);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC > 0) nettyServerBuilder.maxConnectionIdle(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_IDLE_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC > 0) nettyServerBuilder.maxConnectionAge(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_SEC, TimeUnit.SECONDS);
        if (MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC > 0) nettyServerBuilder.maxConnectionAgeGrace(MetaInfo.PROPERTY_GRPC_CHANNEL_MAX_CONNECTION_AGE_GRACE_SEC, TimeUnit.SECONDS);
       return  nettyServerBuilder.build();
    }
}
