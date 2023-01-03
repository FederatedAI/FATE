package com.osx.broker.server;

import com.osx.broker.ServiceContainer;
import com.osx.broker.grpc.ContextPrepareInterceptor;
import com.osx.broker.grpc.ServiceExceptionHandler;
import com.osx.broker.http.DispatchServlet;
import com.osx.core.config.MetaInfo;

import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import org.eclipse.jetty.server.*;
import org.eclipse.jetty.server.handler.AbstractHandler;
import org.eclipse.jetty.server.handler.ContextHandler;
import org.eclipse.jetty.server.handler.HandlerList;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

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
        httpServer =   buildHttpServer();

        //  tlsServer = buildTlsServer();
    }

    public Server  buildHttpServer(){
        Server server = new Server();
        try {
            // The number of acceptor threads.
            int acceptors = 1;
// The number of selectors.
            int selectors = 1;
// Create a ServerConnector instance.
            ServerConnector connector = new ServerConnector(server, acceptors, selectors, new HttpConnectionFactory());

// Configure TCP/IP parameters.

// The port to listen to.
            logger.info("http server try to start listen port {}", MetaInfo.PROPERTY_HTTP_PORT);
            connector.setPort(MetaInfo.PROPERTY_HTTP_PORT);
// The address to bind to.
            connector.setHost("127.0.0.1");
// The TCP accept queue size.
            connector.setAcceptQueueSize(128);
            server.addConnector(connector);
            server.setHandler(buildServlet());



            return  server;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    ServletContextHandler   buildServlet(){
        // Create a ServletContextHandler with contextPath.
        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath(MetaInfo.PROPERTY_HTTP_CONTEXT_PATH);
// Add the Servlet implementing the cart functionality to the context.
        ServletHolder servletHolder = context.addServlet(DispatchServlet.class, MetaInfo.PROPERTY_HTTP_SERVLET_PATH);//"/*"
// Configure the Servlet with init-parameters.
   //     servletHolder.setInitParameter("maxItems", "128");

// Add the CrossOriginFilter to protect from CSRF attacks.
  //      FilterHolder filterHolder = context.addFilter(CrossOriginFilter.class, "/*", EnumSet.of(DispatcherType.REQUEST));
// Configure the filter.
 //       filterHolder.setAsyncSupported(true);

        return  context;
    }

//    Handler  buildHandler(){
//        HandlerList handlerList = new HandlerList();
//// Create a ContextHandler with contextPath.
//        ServletContextHandler  servletContextHandler = this.buildServlet();
//        ContextHandler contextHandler = new ContextHandler();
//        contextHandler.setContextPath("/shop");
//      //  contextHandler.setHandler(new ShopHandler());
//        handlerList.addHandler(contextHandler);
//        return  handlerList;
//    }

    public boolean start() {

        init();
        try {
            logger.info("grpc server try to start, listen port {}", MetaInfo.PROPERTY_PORT);
            server.start();
            if(httpServer!=null){
                logger.info("http server try to start, listen port {}", MetaInfo.PROPERTY_HTTP_PORT);
                httpServer.start();

            }
            if (tlsServer != null) {
                tlsServer.start();
            }
            logger.info("listen http port {} success", MetaInfo.PROPERTY_HTTP_PORT);
            logger.info("listen grpc port {} success", MetaInfo.PROPERTY_PORT);
        } catch (Exception e) {
            if (e.getCause() instanceof java.net.BindException) {
                logger.error("port {}  already in use, please try to choose another one  !!!!", MetaInfo.PROPERTY_PORT);
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


    private io.grpc.Server buildServer() {
        NettyServerBuilder nettyServerBuilder = (NettyServerBuilder) ServerBuilder.forPort(MetaInfo.PROPERTY_PORT);
        nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.proxyGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
        nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.pcpGrpcService, new ServiceExceptionHandler(), new ContextPrepareInterceptor()));
        //nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.commonService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));
        //nettyServerBuilder.addService(ServerInterceptors.intercept(ServiceContainer.clusterService, new ServiceExceptionHandler(),new ContextPrepareInterceptor()));


        // nettyServerBuilder.addService(ServerInterceptors.intercept(dLedgerServer,new ServiceExceptionHandler(), new ContextPrepareInterceptor() ));
        nettyServerBuilder
                .executor(Executors.newCachedThreadPool())
                .maxConcurrentCallsPerConnection(MetaInfo.PROPERTY_GRPC_SERVER_MAX_CONCURRENT_CALL_PER_CONNECTION)
                .maxInboundMessageSize(MetaInfo.PROPERTY_GRPC_SERVER_MAX_INBOUND_MESSAGE_SIZE)
                .maxInboundMetadataSize(MetaInfo.PROPERTY_GRPC_SERVER_MAX_INBOUND_METADATA_SIZE)
                .flowControlWindow(MetaInfo.PROPERTY_GRPC_SERVER_FLOW_CONTROL_WINDOW);


//        PERMIT_KEEPALIVE_TIME
//，默认 5 minutes
//                PERMIT_KEEPALIVE_WITHOUT_CALLS
//，默认 false
        //   nettyServerBuilder.withChildOption(c)
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
